"particle filter"

import math

import numpy as np
from numba import config, cuda, float32, njit, prange, vectorize
from numba.cuda.random import create_xoroshiro128p_states

config.CUDA_ENABLE_PYNVJITLINK = 1


@njit
def logit(x):
    """
    Numerically stable logit function.
    """
    x_safe = np.where(np.isclose(x, 0.0) | np.isclose(x, 1.0), 0.5, x)
    return np.where(
        np.isclose(x, 0.0),
        -np.inf,
        np.where(
            np.isclose(x, 1.0),
            np.inf,
            np.log(x_safe / (1 - x_safe)),
        ),
    )


@njit
def expit(x):
    """
    Numerically stable sigmoid function.
    """
    return np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))


@njit
def binom_logpmf(y: int, n: int, logit_p: float) -> float:
    """Log-likelihood of Binomial(y | n, sigmoid(logit_p)) without computing sigmoid."""
    # omit the expensive to compute binomial coefficient
    # C = -betaln(n - y + 1, y + 1) - np.log1p(n)
    return np.where(
        np.isneginf(logit_p),
        np.where(y == 0, 0.0, -np.inf),
        np.where(
            np.isinf(logit_p),
            np.where(y == n, 0.0, -np.inf),
            y * logit_p - n * np.logaddexp(0.0, logit_p),
        ),
    )


@cuda.jit(device=True)
def binom_logpmf_gpu(y: int, n: int, logit_p: float) -> float:
    """CUDA device function: log P(Y=y | Binomial(n, sigmoid(logit_p)))."""
    if logit_p == float("-inf"):
        return 0.0 if y == 0 else float("-inf")
    elif logit_p == float("inf"):
        return 0.0 if y == n else float("-inf")
    else:
        # logaddexp(0, x) = log(1 + exp(x)), numerically stable
        if logit_p > 0:
            return y * logit_p - n * (logit_p + math.log1p(math.exp(-logit_p)))
        else:
            return y * logit_p - n * math.log1p(math.exp(logit_p))


@njit
def random_binomial_large_N(n, p):
    if p * n < 10.0:
        return np.random.poisson(n * p)
    elif p * n > n - 10.0:
        return n - np.random.poisson(n * (1 - p))
    z = np.random.normal(n * p, np.sqrt(n * p * (1 - p)))
    k = np.rint(z)
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


@njit(parallel=True)
def forward_filter(obs, thetas, s, t, pi, seed, ll_out, N_E):
    """
    Forward algorithm for the particle filter.
    obs: observations [T], 0/1 array
    thetas: [T, D] admixture loadings over D populations (rows sum to 1)
    s: selection matrix [T, D]
    t: [T] time indices
    pi: initial state distribution
    """
    np.random.seed(seed)
    n = len(obs)
    p, d = pi.shape
    alpha = np.full((n, p, d), np.nan)
    particles = pi
    log_weights = np.empty(p)
    inds = np.empty(p, dtype=np.int32)
    ll_out[0] = 0.0

    # Recursion
    for i in range(n):
        tr = False
        ll = np.nan
        if (i > 0) and (t[i] != t[i - 1]):
            # Resample particles under transition
            s_t = s[t[i]]
            lsp = np.log1p(s_t / 2)
            for j in prange(p):
                logit_prime = particles[j] + lsp
                prob = expit(logit_prime)
                for k in range(d):
                    n_prime = random_binomial_large_N(2 * N_E, prob[k])
                    particles[j, k] = logit(n_prime / 2 / N_E)
            tr = True
        else:
            # observation
            log_theta = np.log(thetas[i])
            for j in prange(p):
                log_p_b = binom_logpmf(obs[i], 1, particles[j])
                log_weights[j] = logsumexp(log_p_b + log_theta)
            # resample particles according to weights
            lse = logsumexp(log_weights)
            ll = lse - np.log(p)
            ll_out[0] += ll
            if ~np.isfinite(ll_out[0]):
                print(ll)
                print(particles)
                print(lse)
                print(log_weights)
                print(log_theta)
                print(obs[i])
                assert False
            for j in prange(p):
                g = np.random.gumbel(0.0, 1.0, p)
                inds[j] = np.argmax(g + log_weights)
            particles = particles[inds]
            # if i >= 199:
            #     breakpoint()
            # diversity = len(np.unique(inds))
            # other_inds = np.random.choice(p, p=np.exp(log_weights - lse), replace=True, size=p)
            # other_diversity = len(np.unique(other_inds))
        # print("i:{} trans:{} theta:{} t:{} obs:{} mean_p:{} ll:{}".format(
        #     i, tr,
        #     thetas[i],
        #     t[i], obs[i], np.mean(expit(particles), 0),
        #     ll,
        # ))
        alpha[i] = particles
        # number of unique particles
    return alpha


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


@cuda.jit
def backward_sample_kernel(alpha, s, N_E, ret, rng_states):
    b = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    B = ret.shape[0]
    if b >= B:
        return

    n, p, d = alpha.shape
    shared_ret = cuda.local.array(shape=(MAX_N, MAX_D), dtype=float32)

    # Step 0
    j = int(cuda.random.xoroshiro128p_uniform_float32(rng_states, b) * p)
    for k in range(d):
        shared_ret[0, k] = alpha[n - 1, j, k]

    for i in range(1, n):
        max_score = float("-inf")
        argmax_j = -1
        for j in range(p):
            log_w = 0.0
            for k in range(d):
                p0 = (1 + math.tanh(shared_ret[i - 1, k] / 2)) / 2  # sigmoid
                logit_p1 = alpha[n - 1 - i, j, k] + math.log1p(s[n - i - 1, k] / 2.0)
                log_w += 2 * N_E * binom_logpmf_gpu(p0, 1, logit_p1)
            g = -math.log(
                -math.log(cuda.random.xoroshiro128p_uniform_float32(rng_states, b))
            )
            score = log_w + g
            if score > max_score:
                max_score = score
                argmax_j = j
        for k in range(d):
            shared_ret[i, k] = alpha[n - 1 - i, argmax_j, k]

    for i in range(n):
        for k in range(d):
            ret[b, i, k] = shared_ret[i, k]


def backward_sample_batched(
    alpha: np.ndarray, s: np.ndarray, N_E: int, B: int, seed: int = 0
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
    n, p, d = alpha.shape
    assert n <= MAX_N
    assert d <= MAX_D
    threads_per_block = 64
    blocks = (B + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    alpha_d = cuda.to_device(alpha.astype(np.float32))
    s_d = cuda.to_device(s.astype(np.float32))
    ret_d = cuda.device_array((B, n, d), dtype=np.float32)

    # RNG
    rng_states = create_xoroshiro128p_states(B, seed=seed)

    # Launch
    backward_sample_kernel[blocks, threads_per_block](
        alpha_d, s_d, N_E, ret_d, rng_states
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
