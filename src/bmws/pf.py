"particle filter"

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from numba import njit

from .timer import timer


@timer
@jax.jit
def forward_filter(z, obs, s, particles, log_weights, ref_path, mean_path, N_E, seed):
    """
    Forward algorithm for the particle filter.
    obs: observations [T], 0/1 array
    thetas: [T, D] admixture loadings over D populations (rows sum to 1)
    s: selection matrix [T, D]
    t: [T] time indices
    pi: initial state distribution: particles, log_weights
    """
    N_E = N_E.astype(int)
    P, D = particles.shape
    T = len(z)
    inds = jnp.empty(P, dtype=np.int32)
    key = jax.random.key(seed)

    # Initialize particles and log weights
    # y0|x0 ~ prod bernoulli(x0[z]) => x0 ~ beta(2Np, 2N(1-p))
    key, subkey = jax.random.split(key)
    n = obs[0, :, 0][:, None]
    d = obs[0, :, 1][:, None]  # [N, 1]
    a = 2 * N_E * jnp.sum(n * d * z[0], axis=0) + 1
    b = 2 * N_E * jnp.sum(n * (1 - d) * z[0], axis=0) + 1
    particles0 = (
        jax.random.beta(subkey, a, b, shape=(P - 1, 3)).astype(jnp.int32) * 2 * N_E
    )
    particles = jnp.concatenate(
        [particles0, ref_path[0][None]], axis=0
    )  # Add reference path
    log_weights = jnp.full((P,), -jnp.log(P), dtype=jnp.float32)  # uniform log weights
    alpha0 = particles  # Store initial particles

    def body(accum, carry):
        particles, log_weights = accum
        t, key = carry

        # transition and process observations
        # foll/owing pgas paper

        # line 5
        key, subkey = jax.random.split(key)
        inds = jax.random.categorical(subkey, shape=(P - 1,), logits=log_weights)

        # line 6
        p = particles / 2 / N_E
        # p' from _last_ transition
        p_prime = (1 + s[t - 1] / 2) * p / (1 + s[t - 1] / 2 * p)
        log_p = log_weights + jax.scipy.stats.binom.logpmf(
            ref_path[t], 2 * N_E, p_prime
        ).sum(1)
        key, subkey = jax.random.split(key)
        J = jax.random.categorical(subkey, logits=log_p)
        inds = jnp.append(inds, J)

        particles = particles[inds]
        p = particles / 2 / N_E  # Update p with resampled particles

        # p' from current transition
        p_prime = (1 + s[t] / 2) * p / (1 + s[t] / 2 * p)

        # line 7
        # p(x_t | x_{t-1}, y_t) proposal distn
        # p(x_t|x_t-1)p(y_t|x_t) \propto (x_t-1 / 2 / N_E)^{x_t} (1 - x_t / 2 / N_E)^{2N_E - x_t}  *
        # use that x / 2NE ->_d beta(2Np, 2N(1-p))
        n = obs[t, :, 0][:, None]
        d = obs[t, :, 1][:, None]
        a_prime = 2 * N_E * p_prime + jnp.sum(n * d * z[t], axis=0) + 1
        b_prime = 2 * N_E * (1 - p_prime) + jnp.sum(n * (1 - d) * z[t], axis=0) + 1
        key, *subkeys = jax.random.split(key, 5)
        p0 = jax.random.beta(subkeys[0], a_prime, b_prime)
        particles0 = jax.random.binomial(
            subkeys[1], shape=(P, 3), n=2 * N_E, p=p0
        ).astype(jnp.int32)
        particles1 = jax.random.randint(
            subkeys[2], shape=(P, 3), minval=0, maxval=2 * N_E + 1
        ).astype(jnp.int32)
        # line 8
        eps = 1e-6
        b = jax.random.bernoulli(subkeys[3], eps, shape=(P, 3))
        particles = jnp.where(b, particles1, particles0)  # Mix with random particles
        particles = jnp.concatenate([particles[:-1], ref_path[t][None]])

        # line 10
        p = particles / 2 / N_E
        r0 = (
            jax.scipy.stats.binom.logpmf(particles, 2 * N_E, p_prime).sum(1)
            + jsp.xlogy(n * d * z[t], p[:, None]).sum((1, 2))
            + jsp.xlog1py(n * (1 - d) * z[t], 1 - p[:, None]).sum((1, 2))
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

        #  line 9
        return (particles, log_weights), (particles, inds)

    (particles, log_weights), (alpha, ancestors) = jax.lax.scan(
        body, (alpha0, log_weights), (jnp.arange(1, T), jax.random.split(key, T - 1))
    )

    alpha = jnp.concatenate([alpha0[None], alpha], axis=0)  # Add initial particles
    return alpha, log_weights, ancestors


@njit
def backward_trace(alpha, gamma, ancestors, batch_size, seed):
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
    ret = np.empty((N, D), dtype=alpha.dtype)
    g = np.random.gumbel(0.0, 1.0, size=P)  # Sample Gumbel noise
    j = np.argmax(gamma + g)  # Sample initial index j at final time
    path_indices = np.empty(N, dtype=np.int32)
    path_indices[-1] = j

    # Trace ancestors backward
    for i in range(N - 2, -1, -1):
        j = ancestors[i, j]
        path_indices[i] = j

    for i in range(N):
        ret[N - i - 1, :] = alpha[i, path_indices[i], :]

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
