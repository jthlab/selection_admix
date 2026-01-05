"particle filter"

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from numba import njit


FWD_DEBUG = False


def forward_filter(z, obs, s, particles, log_weights, ref_path, mean_path, N_E, key):
    """
    Forward algorithm for the particle filter.
    obs: observations [T], 0/1 array
    thetas: [T, D] admixture loadings over D populations (rows sum to 1)
    s: selection matrix [T, D]
    t: [T] time indices
    pi: initial state distribution: particles, log_weights
    """
    N_E = jnp.array(N_E).astype(int)
    P, D = particles.shape
    T = len(z)

    def body(accum, seq):
        import os

        if FWD_DEBUG and os.path.exists("/tmp/bodybreak"):
            breakpoint()
            os.remove("/tmp/bodybreak")
        particles, log_weights, loglik, key = accum
        mp_t, ob_t, rp_t, s_t, z_t = seq

        old_particles = particles

        # transition and process observations
        # foll/owing pgas paper

        # line 5
        key, subkey = jax.random.split(key)
        # ess = 1 / jnp.sum(jax.nn.softmax(log_weights) ** 2)
        resample = True
        inds = jnp.where(
            resample,
            jax.random.categorical(subkey, shape=(P - 1,), logits=log_weights),
            jnp.arange(P - 1),
        )
        # jax.debug.print("ess:{} resample:{}", ess, resample)
        log_weights = jnp.where(
            resample,
            jnp.full(P, -jnp.log(P)),
            log_weights,
        )

        # line 6
        p = particles / 2 / N_E
        p_prime = (1 + s_t / 2) * p / (1 + s_t / 2 * p)
        p_prime = jnp.clip(p_prime, 0, 1)
        log_p = log_weights + jax.scipy.stats.binom.logpmf(rp_t, 2 * N_E, p_prime).sum(
            1
        )
        loglik += jsp.logsumexp(log_p)

        # if FWD_DEBUG:
        #     try:
        #         assert jnp.isfinite(log_p).any()
        #     except AssertionError:
        #         breakpoint()
        #         pass

        log_p = jnp.where(jnp.isinf(log_p).all(), jnp.zeros(P), log_p)
        key, subkey = jax.random.split(key)
        J = jax.random.categorical(subkey, logits=log_p)
        inds = jnp.append(inds, J)

        # this is \tilde{x}_{t-1} in paper
        particles = particles[inds]
        p = particles / 2 / N_E  # Update p with resampled particles

        # p' from current transition
        p_prime = (1 + s_t / 2) * p / (1 + s_t / 2 * p)
        p_prime = jnp.clip(p_prime, 0, 1)

        # line 7
        # p(x_t | x_{t-1}, y_t) proposal distn
        # p(x_t|x_t-1)p(y_t|x_t) \propto (x_t-1 / 2 / N_E)^{x_t} (1 - x_t / 2 / N_E)^{2N_E - x_t}  *
        # use that x / 2NE ->_d beta(2Np, 2N(1-p))
        n = ob_t[:, 0][:, None]
        d = ob_t[:, 1][:, None]

        def f0(key):
            particles = jax.random.binomial(key, 2 * N_E, p_prime).astype(jnp.int32)
            particles = jnp.concatenate([particles[:-1], rp_t[None]])
            log_weights = jnp.full(P, -jnp.log(P))
            return particles, log_weights

        def f1(key):
            a_prime = 2 * N_E * p_prime + jnp.sum(n * d * z_t, axis=0) + 1
            b_prime = 2 * N_E * (1 - p_prime) + jnp.sum(n * (1 - d) * z_t, axis=0) + 1
            key, *subkeys = jax.random.split(key, 5)
            p0 = jax.random.beta(subkeys[0], a_prime, b_prime)
            particles0 = jax.random.binomial(
                subkeys[1], shape=(P, D), n=2 * N_E, p=p0
            ).astype(jnp.int32)
            # to maintain absolute continuity, we mix with uniform distribution
            particles1 = jax.random.randint(
                subkeys[2], shape=(P, D), minval=0, maxval=2 * N_E + 1
            ).astype(jnp.int32)
            mixture_wts = jnp.array([1.0 - 1e-6, 1e-6])
            b = jax.random.choice(subkeys[3], 2, shape=(P,), p=mixture_wts)
            particles = jnp.stack([particles0, particles1], axis=0)[b, jnp.arange(P)]
            particles = jnp.concatenate([particles[:-1], rp_t[None]])

            p0 = jnp.isclose(p_prime, 0.0)
            p1 = jnp.isclose(p_prime, 1.0)
            particles = jnp.where(p0, 0, particles)
            particles = jnp.where(p1, 2 * N_E, particles)
            poly = ~(p0 | p1)

            # line 10
            p = particles / 2 / N_E
            r0 = (
                jax.scipy.stats.binom.logpmf(particles, 2 * N_E, p_prime).sum(1)
                + jsp.xlogy(n * d * z_t, p[:, None]).sum((1, 2))
                + jsp.xlog1py(n * (1 - d) * z_t, -p[:, None]).sum((1, 2))
            )
            # proposal distn is mixture:
            # eps * betabinom + (1 - eps) * uniform
            r10s = jax.scipy.stats.betabinom.logpmf(
                particles, 2 * N_E, a_prime, b_prime
            )
            r10 = jnp.sum(r10s * poly, 1)
            r11 = jnp.sum(
                jnp.full((P, D), -jnp.log(2 * N_E)) * poly, 1
            )  # uniform log pmf
            r1 = jsp.logsumexp(
                jnp.array([r10, r11]) + jnp.log(mixture_wts)[:, None], axis=0
            )
            log_weights = r0 - r1

            if FWD_DEBUG:
                panc = old_particles[inds]
                bad = ((panc == 0) & (particles > 0)) | (
                    (panc == 2 * N_E) & (particles < 2 * N_E)
                )
                if bad.any():
                    breakpoint()
                    pass

            return particles, log_weights

        key, subkey = jax.random.split(key)

        # if n.sum() = 0, then we can sample directly from p(x[t] | x[t-1])

        if FWD_DEBUG:
            f = f0 if n.sum() == 0 else f1
            particles, log_weights1 = f(subkey)

            try:
                assert jnp.isfinite(log_weights).any()
                assert (~jnp.isnan(log_weights)).all()
            except AssertionError:
                breakpoint()
                pass
        else:
            particles, log_weights1 = jax.lax.cond(n.sum() == 0, f0, f1, subkey)

        log_weights += log_weights1

        #  line 9
        return (particles, log_weights, loglik, key), (particles, inds)

    # d | x_0, z ~ binom(n, x0) => x0 | y0 ~ beta(...)
    key, subkey = jax.random.split(key)
    n, d = obs[0].T[..., None]
    a0 = jnp.sum(n * d * z[0], axis=0) + 1
    b0 = jnp.sum(n * (1 - d) * z[0], axis=0) + 1
    particles = (2 * N_E * jax.random.beta(subkey, a=a0, b=b0, shape=(P, D))).astype(
        jnp.int32
    )
    particles = particles.at[-1].set(ref_path[0])
    alpha0 = particles
    log_weights = jnp.full(P, -jnp.log(P))  # uniform weights
    loglik = 0.0

    if FWD_DEBUG:
        alpha = [alpha0]
        ancestors = []
        for t in range(1, T):
            (particles, log_weights, loglik, key), (_, inds) = body(
                (particles, log_weights, loglik, key),
                (t, mean_path[t], obs[t], ref_path[t], s[t], z[t]),
            )
            alpha.append(particles)
            ancestors.append(inds)
        alpha = jnp.array(alpha)
    else:
        (particles, log_weights, loglik, _), (alpha, ancestors) = jax.lax.scan(
            body,
            (particles, log_weights, loglik, key),
            (mean_path[1:], obs[1:], ref_path[1:], s[1:], z[1:]),
        )

    alpha = jnp.concatenate([alpha0[None], alpha])
    ancestors = jnp.array(ancestors)

    return alpha, log_weights, ancestors, loglik


if not FWD_DEBUG:
    forward_filter = jax.jit(forward_filter)


@njit(parallel=True)
def backward_trace(alpha, gamma, ancestors, seed):
    """
    Batched backward sampling with PGAS ancestor tracing using Numba on CPU.
    alpha: [N, P, D] particles
    ancestors: [N, P] ancestor indices from forward filter
    gamma: [N, P] log weights
    batch_size: number of paths to sample
    Returns sampled paths: [batch_size, N, D]
    """
    np.random.seed(seed)
    N, P, D = alpha.shape
    g = np.random.gumbel(0.0, 1.0, size=P)  # Sample Gumbel noise
    j = np.argmax(gamma + g)  # Sample initial index j at final time
    ret = np.empty_like(alpha[:, 0])
    ret[0] = alpha[N - 1, j]

    # Trace ancestors backward
    for i in range(N - 2, -1, -1):
        j = ancestors[i, j]
        ret[N - i - 1] = alpha[i, j]

    return ret
