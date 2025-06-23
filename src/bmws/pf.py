"particle filter"

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from numba import njit


FWD_DEBUG = False


def forward_filter(z, obs, s, particles, log_weights, ref_path, N_E, key):
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
        particles, log_weights, key = accum
        ob_t, rp_t, s_t, z_t = seq

        # transition and process observations
        # foll/owing pgas paper

        # line 5
        key, subkey = jax.random.split(key)
        inds = jax.random.categorical(subkey, shape=(P - 1,), logits=log_weights)
        # line 6
        p = particles / 2 / N_E
        p_prime = (1 + s_t / 2) * p / (1 + s_t / 2 * p)
        p_prime = jnp.clip(p_prime, 0, 1)
        log_p = log_weights + jax.scipy.stats.binom.logpmf(rp_t, 2 * N_E, p_prime).sum(
            1
        )
        key, subkey = jax.random.split(key)

        if FWD_DEBUG:
            try:
                assert jnp.isfinite(log_p).any()
            except AssertionError:
                breakpoint()
                pass

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
            eps = 1e-6
            b = jax.random.bernoulli(subkeys[3], eps, shape=(P, D))
            particles = jnp.where(
                b, particles1, particles0
            )  # Mix with random particles
            particles = jnp.concatenate([particles[:-1], rp_t[None]])

            # line 10
            p = particles / 2 / N_E
            r0 = (
                jax.scipy.stats.binom.logpmf(particles, 2 * N_E, p_prime).sum(1)
                + jsp.xlogy(n * d * z_t, p[:, None]).sum((1, 2))
                + jsp.xlog1py(n * (1 - d) * z_t, 1 - p[:, None]).sum((1, 2))
            )
            # proposal distn is mixture:
            # eps * betabinom + (1 - eps) * uniform
            r10 = jax.scipy.stats.betabinom.logpmf(
                particles, 2 * N_E, a_prime, b_prime
            ).sum(1)
            r11 = -jnp.log(2 * N_E)  # uniform log pmf
            # r1 = log(a1 * exp(r10) + (1 - a1) * exp(r11))
            r1 = jnp.logaddexp(r10 + jnp.log(eps), r11 + jnp.log1p(-eps))
            log_weights = r0 - r1
            return particles, log_weights

        key, subkey = jax.random.split(key)

        # if n.sum() = 0, then we can sample directly from p(x[t] | x[t-1])

        if FWD_DEBUG:
            f = f0 if n.sum() == 0 else f1
            particles, log_weights = f(subkey)

            try:
                assert jnp.isfinite(log_weights).any()
                assert (~jnp.isnan(log_weights)).all()
            except AssertionError:
                breakpoint()
                pass
        else:
            particles, log_weights = jax.lax.cond(n.sum() == 0, f0, f1, subkey)

        #  line 9
        return (particles, log_weights, key), (particles, inds)

    if FWD_DEBUG:
        alpha = []
        ancestors = []
        for t in range(T):
            (particles, log_weights, key), (_, inds) = body(
                (particles, log_weights, key), (obs[t], ref_path[t], s[t], z[t])
            )
            alpha.append(particles)
            ancestors.append(inds)
    else:
        (particles, log_weights, _), (alpha, ancestors) = jax.lax.scan(
            body, (particles, log_weights, key), (obs, ref_path, s, z)
        )

    alpha = jnp.array(alpha)
    ancestors = jnp.array(ancestors)

    return alpha, log_weights, ancestors[1:]


if not FWD_DEBUG:
    forward_filter = jax.jit(forward_filter)


@njit
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


def test_forward():
    """
    Test the forward algorithm.
    """
    # Test the forward algorithm
    np.seterr(divide="raise", over="raise", invalid="raise")
    obs = np.array([0, 1, 1, 0, 1])
    thetas = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
    s = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
    t = np.array([0, 1, 2, 3, 4])
    pi = np.random.beta(1, 100, (1_000, 2))
    seed = 42
    alpha = forward_filter(obs, thetas, s, t, pi, seed)
    print(alpha)


def test_bwd_sample():
    """
    Test the backward sampling algorithm.
    """
    # Test the backward sampling algorithm
    obs = np.array([0, 1, 1, 0, 1])
    thetas = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
    s = np.random.uniform(-0.1, 0.1, (5, 2))
    t = np.array([0, 1, 2, 3, 4])
    N_E = 10_000
    particles = (2 * N_E * np.random.beta(1, 100, (10_000, 2))).astype(int)
    log_weights = np.full((10_000,), -math.log(10_000), dtype=np.float32)
    alpha = np.zeros((5, 10_000, 2), dtype=np.int32)
    gamma = np.zeros((5, 10_000), dtype=np.float32)
    ref_path = np.zeros((5, 2), dtype=np.int32)
    ll_out = np.zeros(1)
    forward_filter(
        obs,
        thetas,
        s,
        t,
        particles,
        log_weights,
        ref_path,
        alpha,
        gamma,
        ll=ll_out,
        N_E=N_E,
        seed=1,
    )
    ret = backward_sample_batched(alpha, gamma, s, N_E, 1)
    print(ret)
