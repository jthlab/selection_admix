import logging

from bmws.betamix import Dataset

logger = logging.getLogger(__name__)
from typing import Dict, Union

import numpy as np

from bmws.common import f_sh
from bmws.estimate import empirical_bayes, estimate, jittable_estimate


def sim_wf(
    N: np.ndarray,
    s: np.ndarray,
    h: np.ndarray,
    f0: int,
    rng: Union[int, np.random.Generator],
):
    """Simulate T generations under wright-fisher model with population size 2N, where
    allele has initial frequency f0 and the per-generation selection coefficient is
    s[t], t=1,...,T.

    Returns:
        Vector f = [f[0], f[1], ..., f[T]] of allele frequencies at each generation.
    """
    assert 0 <= f0 <= 1
    T = len(s)
    assert T == len(h)
    assert T == len(N)

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    f = np.zeros(T + 1)
    f[0] = f0
    for t in range(1, T + 1):
        p = f_sh(f[t - 1], s[t - 1], h[t - 1])
        f[t] = rng.binomial(2 * N[t - 1], p) / (2 * N[t - 1])
    return f

def sim_full(
    mdl: Dict,
    seed: int,
    D: int = 100,
    Ne: int = 1000,
    n: int = 100,  # sample size
    d: int = 10,  # sampling interval
):
    T = len(mdl["s"]) + 1  # number of time points
    rng = np.random.default_rng(seed)
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    obs = np.zeros(T, dtype=int)
    size = np.zeros(T, dtype=int)
    obs[::d] = rng.binomial(n, af[::d])  # sample n haploids every d generations
    size[::d] = n
    return obs, size

def sim_and_fit(
    mdl: Dict,
    seed: int,
    lam: float,
    Ne=1e4,  # effective population size
    n=100,  # sample size - see below
    k=10,  # sampling interval - either an integer, or an iterable of sampling times.
    Ne_fit=None,  # Ne to use for estimation (if different to that used for simulation)
    em_iterations=3,  # use empirical bayes to infer prior hyperparameters
    M=100,  # number of mixture components
    **kwargs
):
    # this is now just a frontend for sim_admix
    T = len(mdl['s']) + 1
    mdl = {k: np.array(v)[..., None] for k, v in mdl.items()}
    N = n
    K = 1
    thetas = np.ones([T, N, K])
    samples = np.zeros([T, N])
    samples[::k] = n
    Ne = np.full_like(mdl["s"], Ne)
    res = sim_admix(mdl, seed, lam, thetas, samples, Ne, Ne_fit, em_iterations, M, **kwargs)
    return res

def sim_admix(
    mdl: Dict,
    seed: int,
    lam: float,
    thetas, samples,
    Ne=1e4,  # effective population size
    Ne_fit=None,  # Ne to use for estimation (if different to that used for simulation)
    em_iterations=3,  # use empirical bayes to infer prior hyperparameters
    M=100,  # number of mixture components
    **kwargs):
    '''
    Simulate from Wright-Fisher model for several populations, sample admixed individulas,
    and perform inference on resulting dataset.

    Params:
        n: If an integer, sample this many individuals every k-th time point.
    '''
    # Parameters
    assert thetas.ndim == 3
    T, N, K = thetas.shape
    assert samples.ndim == 2
    assert samples.shape == (T, N)
    assert mdl["s"].ndim == 2
    assert mdl["s"].shape == (T - 1, K)
    if isinstance(Ne, float):
        Ne = np.full_like(mdl["s"], Ne)
    assert Ne.ndim == 2
    assert Ne.shape == (T - 1, K)

    # Set up population size.
    if not Ne_fit:
        Ne_fit = Ne
    else:
        Ne_fit = np.array(Ne_fit)
        assert Ne_fit.shape == Ne.shape

    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    # simulate the wright-fisher model. all parameter vectors are reversed so that times runs from past to present.
    s = mdl["s"]
    h = mdl["h"]
    f0 = mdl["f0"]
    afs = [sim_wf(Ne_i[::-1], s_i[::-1], h_i[::-1], f0_i, rng)[::-1]
          for Ne_i, s_i, h_i, f0_i in zip(Ne.T, s.T, h.T, f0.T)]
    afs = np.transpose(afs)
    obs = np.zeros([T, N, 2], dtype=int)
    for t in range(T):
        for n in range(N):
            k = rng.choice(K, p=thetas[t, n])
            p = afs[t, k]
            a = samples[t, n]
            d = rng.binomial(a, p)
            obs[t, n] = [a, d]
    data = Dataset(thetas=thetas, obs=obs)

    # setup prior
    s = np.zeros([T - 1, data.K])
    for i in range(em_iterations):
        logger.info("EM iteration %d", i)
        prior = empirical_bayes(s, data, Ne, M)
        s = estimate(data, Ne_fit, lam=lam, prior=prior, **kwargs)

    return {"s_hat": s, "obs": obs, "Ne": Ne, "true_af": afs, "prior": prior}