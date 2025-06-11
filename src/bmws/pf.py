"particle filter"

import math

import jax
import jax.numpy as jnp
import numba
import numpy as np
import scipy
from numba import njit, prange
from scipy.special import logsumexp


@njit
def random_binomial_large_N(n, p):
    if p == 0.0:
        return 0
    if p == 1.0:
        return n
    if p * n < 10.0:
        return np.random.poisson(n * p)
    elif p * n > n - 10.0:
        return n - np.random.poisson(n * (1 - p))
    z = np.random.normal(n * p, np.sqrt(n * p * (1 - p)))
    k = np.rint(z)
    # rarely, the gaussian sample can be outside the range [0, n]
    k = np.minimum(k, n)
    k = np.maximum(k, 0)
    return k


@njit(parallel=True)
def logsumexp(a):
    """
    Numerically stable logsumexp function.
    """
    a_max = np.max(a)
    if np.isneginf(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


@njit(parallel=True)
def resample(
    particles: np.ndarray,
    log_weights: np.ndarray,
    ll: np.ndarray,
    weights_are_uniform: bool = False,
) -> None:
    """
    Resample particles according to their log weights.
    particles: [P, D] array of particles
    log_weights: [P] array of log weights
    ll: [1] array to accumulate log likelihood
    """
    (P, D) = particles.shape
    p0 = np.copy(particles[0])  # Save the first particle
    lse = logsumexp(log_weights)
    ll[0] += lse - np.log(P)
    if weights_are_uniform:
        inds = np.random.randint(0, P, size=P)
    else:
        weights = np.exp(log_weights - lse)
        w_cs = np.cumsum(weights)
        U = np.random.rand(P)
        inds = np.searchsorted(w_cs, U, side="left")
    particles[:] = particles[inds]
    particles[0] = p0  # Restore the first particle
    log_weights[:] = -np.log(P)  # Reset log weights to uniform


@njit(parallel=True)
def xlogy(x, y):
    return np.where((x == 0.0) & (y == 0.0), 0.0, x * np.log(y))


@njit(parallel=True)
def xlog1py(x, y):
    return np.where((x == 0.0) & (y == -1.0), 0.0, x * np.log1p(y))


@njit(parallel=True, cache=True)
def forward_filter(
    obs, thetas, s, t, particles, log_weights, ref_path, alpha, gamma, ll, N_E, seed
):
    """
    Forward algorithm for the particle filter.
    obs: observations [T], 0/1 array
    thetas: [T, D] admixture loadings over D populations (rows sum to 1)
    s: selection matrix [T, D]
    t: [T] time indices
    pi: initial state distribution: particles, log_weights
    """
    np.random.seed(seed)
    N = len(obs)
    P, D = particles.shape
    ll[0] = 0.0
    # forward filtering recursion
    ell = 0
    weights_are_uniform = False
    inds = np.empty(P, dtype=np.int32)
    for i in range(N):
        is_transition = (i > 0) and (t[i] != t[i - 1])

        if is_transition:
            # resample if ess is too low
            log_weights -= logsumexp(log_weights)
            ess = 1.0 / np.sum(np.exp(2 * log_weights))
            if ess < P / 2:
                # print("ess", ess, "resampling")
                resample(particles, log_weights, ll, weights_are_uniform)
                weights_are_uniform = True

            # Save state
            alpha[ell] = particles
            gamma[ell] = log_weights
            ell += 1

            # Mutation step
            p = particles / 2 / N_E
            s_t = s[t[i]]
            p_prime = (1 + s_t / 2) * p / (1 + s_t / 2 * p)
            for j in prange(P):
                for k in range(D):
                    if particles[j, k] > 0 and particles[j, k] < 2 * N_E:
                        # sampling not necessary if fixed. also small numerical errors
                        # can cause p_prime \notin [0, 1]
                        particles[j, k] = random_binomial_large_N(
                            2 * N_E, p_prime[j, k]
                        )

            particles[0] = ref_path[t[i]]

        else:
            # Observation step
            x = particles / 2 / N_E
            log_p_obs = np.log(
                np.sum(
                    np.exp(
                        np.log(thetas[i]) + xlogy(obs[i], x) + xlog1py(1 - obs[i], -x)
                    ),
                    axis=1,
                )
            )
            log_weights += log_p_obs
            weights_are_uniform = False

    # Final resample
    alpha[ell] = particles
    gamma[ell] = log_weights


@jax.jit
def gsm(log_w, key):
    """
    Gumbel softmax sampling.
    log_w: [P] array of log weights
    Returns: [P] array of sampled indices
    """
    P = log_w.shape[0]
    G = jax.random.gumbel(key, shape=(P, P))
    return jnp.argmax(G + log_w, axis=0)


def backward_sample_batched(
    alpha, gamma, s: np.ndarray, N_E: int, seed: int = 0
) -> np.ndarray:
    # alpha: (N, P, D)
    # gamma: (N, P)
    if (~np.isfinite(alpha).any()) or (~np.isfinite(gamma).any()):
        raise ValueError("NaN detected in alpha or gamma")

    key = jax.random.key(seed)
    rng_np = np.random.default_rng(seed)

    N, P, D = alpha.shape
    ret = np.empty((N, P, D), dtype=np.int32)

    # Step 0
    key, subkey = jax.random.split(key)
    j = gsm(gamma[N - 1], subkey)  # Sample indices using Gumbel softmax
    ret[0] = alpha[N - 1, j]
    n1 = np.asarray(ret[0], dtype=jnp.int32)  # [P, D]

    # steps 1, ..., N
    for i in range(1, N):
        s_t = s[N - i - 1]
        p0 = alpha[N - 1 - i] / 2 / N_E  # [P, D]
        p_prime = (1 + s_t / 2) * p0 / (1 + s_t / 2 * p0)  # [P, D]
        log_w = gamma[N - 1 - i] + scipy.stats.binom.logpmf(n1, 2 * N_E, p_prime).sum(
            axis=1
        )
        key, subkey = jax.random.split(key)
        j = gsm(log_w, subkey)  # Sample indices using Gumbel softmax
        ret[i] = n1 = alpha[N - 1 - i, j]  # [P, D]

    return ret.transpose(
        (1, 0, 2)
    )  # Return shape (P, N, D) for consistency with the original code


MAX_N = 400
MAX_D = 4


def binom_logpmf_cupy(x, n, p):
    """
    Compute the log PMF of a binomial distribution.
    x: number of successes
    n: number of trials
    p: probability of success
    """
    lgamma = cupyx.scipy.special.gammaln
    log_p = cupy.log(p)
    log_q = cupy.log1p(-p)
    return (
        lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * log_p + (n - x) * log_q
    )


def _binom_logpmf(x, n, p):
    """
    Compute the log PMF of a binomial distribution.
    x: number of successes
    n: number of trials
    p: probability of success
    """
    if x < 0 or x > n:
        return float("-inf")
    if p == 0.0:
        return 0.0 if x == 0 else float("-inf")
    if p == 1.0:
        return 0.0 if x == n else float("-inf")
    if p < 0 or p > 1:
        return float("-inf")
    log_p = math.log(p)
    log_q = math.log1p(-p)
    return (
        math.lgamma(n + 1)
        - math.lgamma(x + 1)
        - math.lgamma(n - x + 1)
        + x * log_p
        + (n - x) * log_q
    )


binom_logpmf = njit(_binom_logpmf, cache=True)
# binom_logpmf_gpu = cuda.jit(device=True)(_binom_logpmf)


# @cuda.jit(cache=True)
def backward_sample_kernel(alpha, gamma, s, N_E, ret, rng_states):
    b = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    B = ret.shape[0]
    if b >= B:
        return

    N, P, D = alpha.shape
    shared_ret = cuda.local.array(shape=(MAX_N, MAX_D), dtype=numba.int32)

    # Step 0
    max_score = float("-inf")
    argmax_j = -1
    for j in range(P):
        g = -math.log(-math.log(xoroshiro128p_uniform_float32(rng_states, b)))
        score = gamma[N - 1, j] + g
        if score > max_score:
            max_score = score
            argmax_j = j
    for k in range(D):
        shared_ret[0, k] = alpha[N - 1, argmax_j, k]

    # steps 1, ..., N
    for i in range(1, N):
        max_score = float("-inf")
        argmax_j = -1
        for j in range(P):
            log_w = gamma[N - 1 - i, j]
            for k in range(D):
                s_t = s[N - i - 1, k]
                n1 = shared_ret[i - 1, k]
                p0 = alpha[N - 1 - i, j, k] / 2 / N_E
                p_prime = (1 + s_t / 2) * p0 / (1 + s_t / 2 * p0)
                log_w += binom_logpmf_gpu(n1, 2 * N_E, p_prime)
            g = -math.log(-math.log(xoroshiro128p_uniform_float32(rng_states, b)))
            score = log_w + g
            if score > max_score:
                max_score = score
                argmax_j = j
        for k in range(D):
            shared_ret[i, k] = alpha[N - 1 - i, argmax_j, k]

    for i in range(N):
        for k in range(D):
            ret[b, i, k] = shared_ret[i, k]


def backward_sample_batched0(
    alpha, gamma, s: np.ndarray, N_E: int, seed: int = 0
) -> np.ndarray:
    """
    Sample B backward trajectories using the CUDA kernel.

    Args:
        alpha: (n, p, d) float32 array.
        s: (n - 1,) float32 array.
        N_E: Binomial parameter (int).
        B: Number of samples to draw.
        seed: RNG seed.

    Returns:
        ret: (B, n, d) float32 array of samples.
    """
    N, P, D = alpha.shape
    assert N <= MAX_N
    assert D <= MAX_D
    threads_per_block = 64
    B = P
    blocks = (B + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    alpha_d = cuda.to_device(alpha.astype(np.int32))
    gamma_d = cuda.to_device(gamma.astype(np.float32))
    s_d = cuda.to_device(s.astype(np.float32))
    ret_d = cuda.device_array((B, N, D), dtype=np.int32)

    # RNG
    rng_states = create_xoroshiro128p_states(B, seed=seed)

    # Launch
    backward_sample_kernel[blocks, threads_per_block](
        alpha_d, gamma_d, s_d, N_E, ret_d, rng_states
    )

    return ret_d.copy_to_host()


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
