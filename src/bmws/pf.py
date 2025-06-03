"particle filter"

import math

import numba
import numpy as np
from numba import config, cuda, njit, objmode, prange, vectorize
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

config.CUDA_ENABLE_PYNVJITLINK = 1


@njit
def random_binomial_large_N(n, p):
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


@njit
def logsumexp(a):
    """
    Numerically stable logsumexp function.
    """
    a_max = np.max(a)
    if np.isneginf(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


@cuda.jit
def gumbel_max_resample_kernel(log_weights, rng_states, inds):
    """Each thread samples one index via Gumbel-max over log_weights."""
    tid = cuda.grid(1)
    P = log_weights.shape[0]

    if tid < P:
        max_val = float("-inf")
        max_idx = -1
        for j in range(P):
            u = xoroshiro128p_uniform_float32(rng_states, tid)
            g = -math.log(-math.log(u))  # Gumbel(0,1)
            val = log_weights[j] + g
            if val > max_val:
                max_val = val
                max_idx = j
        inds[tid] = max_idx


def gumbel_max_resample(log_weights: np.ndarray, seed: int) -> np.ndarray:
    P = log_weights.shape[0]
    threads_per_block = 64
    blocks = (P + threads_per_block - 1) // threads_per_block

    # Allocate on device
    d_log_weights = cuda.to_device(log_weights.astype(np.float32))
    d_inds = cuda.to_device(np.arange(P, dtype=np.int32))
    rng_states = create_xoroshiro128p_states(P, seed=seed)

    # Launch
    gumbel_max_resample_kernel[blocks, threads_per_block](
        d_log_weights, rng_states, d_inds
    )

    return d_inds.copy_to_host()


@njit(parallel=True)
def resample(particles, log_weights, ll):
    # resample particles according to weights
    (P,) = log_weights.shape
    lse = logsumexp(log_weights)
    ll[0] += lse - np.log(P)
    weights = np.exp(log_weights - lse)
    w_cs = np.cumsum(weights)
    U = np.random.rand(P)
    inds = np.searchsorted(w_cs, inds, side="left")
    particles[:] = particles[inds]  # Resample particles based on indices
    log_weights[:] = -np.log(P)


@njit
def log_obs_likelihood(
    x: np.ndarray, obs: np.ndarray, theta: np.ndarray, N_E: int
) -> float:
    """Compute log p(obs | x, theta)"""
    p = x / (2 * N_E)
    p_safe = np.where((x == 0) | (x == 2 * N_E), 0.5, p)

    log_p = obs * np.log(p_safe) + (1 - obs) * np.log1p(-p_safe)
    log_p = np.where((obs == 0) & (x == 0), 0.0, log_p)
    log_p = np.where((obs == 1) & (x == 0), -np.inf, log_p)
    log_p = np.where((obs == 0) & (x == 2 * N_E), -np.inf, log_p)
    log_p = np.where((obs == 1) & (x == 2 * N_E), 0.0, log_p)

    return logsumexp(log_p + np.log(theta))


@njit(parallel=True)
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
    for i in range(N):
        is_transition = (i > 0) and (t[i] != t[i - 1])

        if is_transition:
            if weights_are_uniform:
                inds = np.random.randint(0, P, size=P)
            else:
                log_weights -= logsumexp(log_weights)
                cum_weights = np.exp(log_weights).cumsum()
                u = np.random.rand(P)
                inds = np.searchsorted(cum_weights, u, side="left")
            p0 = np.copy(particles[0])  # Save the first particle
            particles[:] = particles[inds]
            particles[0] = p0
            log_weights[:] = -np.log(P)  # Reset log_weights to uniform
            weights_are_uniform = True

            # Save state
            alpha[ell] = particles
            gamma[ell] = log_weights
            ell += 1

            # Mutation step
            p = particles / 2 / N_E
            s_t = s[t[i]]
            p_prime = (1 + s_t / 2) * p / (1 + s_t / 2 * p)
            for j in prange(P):  # mutate only particles ≠ 0
                for k in range(D):
                    particles[j, k] = random_binomial_large_N(2 * N_E, p_prime[j, k])

            # Set fixed particle to ref path
            for k in range(D):
                particles[0, k] = ref_path[t[i], k]

        # Observation step
        for j in prange(P):
            n = particles[j]
            log_p_x = log_obs_likelihood(n, obs[i], thetas[i], N_E)
            log_weights[j] += log_p_x

        weights_are_uniform = False

    # Final resample
    log_weights -= logsumexp(log_weights)
    cum_weights = np.exp(log_weights).cumsum()
    u = np.random.rand(P - 1)
    inds = np.searchsorted(cum_weights, u, side="left")
    particles[1:] = particles[inds]
    log_weights[:] = -np.log(P)  # Reset log_weights to uniform
    alpha[ell] = particles
    gamma[ell] = log_weights


@njit(nogil=True)
def backward_sample(alpha, s, seed, N_E):
    n, p, d = alpha.shape
    ret = np.full((n, d), np.nan)
    log_weights = np.empty(p)
    g = np.random.gumbel(0.0, 1.0, p)
    ind = np.argmax(g + log_weights)
    ret[0] = alpha[-1, ind]
    for i in range(1, n):
        x = 2 * N_E * expit(ret[i - 1])
        for j in range(p):
            logit_p_prime = alpha[-(i + 1), j] + np.log1p(s[-i] / 2)
            log_weights[j] = np.sum(binom_logpmf(x, 2 * N_E, logit_p_prime))
        g = np.random.gumbel(0.0, 1.0, p)
        ind = np.argmax(g + log_weights)
        ret[i] = alpha[-(i + 1), ind]
    return ret


MAX_N = 400
MAX_D = 4


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
binom_logpmf_gpu = cuda.jit(device=True)(_binom_logpmf)


@cuda.jit(cache=True)
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


def backward_sample_batched(
    alpha, gamma, s: np.ndarray, N_E: int, B: int, seed: int = 0
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
    s = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
    t = np.array([0, 1, 2, 3, 4])
    pi = np.random.beta(1, 100, (10_000, 2))
    seed = 42
    ll_out = np.zeros(1)
    alpha = forward_filter(obs, thetas, s, t, pi, seed, ll_out, N_E=1e4)
    ret = backward_sample_batched(alpha, s, 1e4, 10_000, 1)
    print(ret)
