import logging

import jax.numpy as jnp

from bmws.betamix import Dataset, forward

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
    **kwargs,
):
    # this is now just a frontend for sim_admix
    T = len(mdl["s"]) + 1
    mdl = {k: np.array(v)[..., None] for k, v in mdl.items()}
    N = n
    K = 1
    thetas = np.ones([T, N, K])
    samples = np.zeros([T, N])
    samples[::k] = n
    Ne = np.full_like(mdl["s"], Ne)
    res = sim_admix(
        mdl, seed, lam, thetas, samples, Ne, Ne_fit, em_iterations, M, **kwargs
    )
    return res


def sim_admix(
    mdl: dict,
    data: Dataset,
    seed: int,
    em_iterations=3,
    M=100,
    estimate_kwargs={"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
):
    """
    Simulate from Wright-Fisher model for several populations, sample admixed individulas,
    and perform inference on resulting dataset.

    Params:
        mdl: A dictionary containing the entries "s" and "f0", detailing the selection coefficients
             initial frequencies. The leading dimensions of each of theses should be K, the number of
             populations, and the trailing dimension of "s" should be T - 1, the number of time points minus one.
        data: A Dataset which conveys the sampling model and admixture proportions. data.obs is ignored.
        seed: random seed
        Ne_fit: Ne parameter used for fitting, if different from that used for simulation. Same semantics as Ne.
        estimate_kwargs: arguments passed to :estimate: when model fitting.
    """
    # Parameters
    T = data.T
    K = mdl["s"].shape[1]
    assert mdl["s"].ndim == 2
    assert mdl["s"].shape == (T, K)
    Ne = mdl["Ne"]
    if isinstance(Ne, float):
        Ne = np.full_like(mdl["s"], Ne)
    h = np.full_like(mdl["s"], 0.5)
    assert Ne.ndim == 2
    assert Ne.shape == (T, K)

    Ne_fit = Ne

    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    # simulate the wright-fisher model. all parameter vectors are reversed so that times runs from past to present.
    s = mdl["s"]
    f0 = mdl["f0"]
    afs = [
        sim_wf(Ne_i[::-1], s_i[::-1], h_i[::-1], f0_i, rng)[::-1]
        for Ne_i, s_i, h_i, f0_i in zip(Ne.T, s.T, h.T, f0.T)
    ]
    afs = np.transpose(afs)
    assert afs.shape == (T + 1, K)
    obs = []
    for t, theta, (n, _) in zip(*data):
        k = rng.choice(K, p=theta)
        p = afs[t, k]
        d = rng.binomial(n, p)
        obs.append((int(n), d))
    sim_data = data._replace(obs=jnp.array(obs))

    # setup prior
    s = np.zeros([T, data.K])
    ab = np.ones([2, data.K]) + 1e-4
    for i in range(em_iterations):
        logger.info("EM iteration %d", i)
        ab, prior = empirical_bayes(ab0=ab, s=s, data=sim_data, Ne=Ne, M=M)
        logger.info("ab: %s", ab)
        s = estimate(data=sim_data, Ne=Ne_fit, prior=prior, **estimate_kwargs)
        logger.info("s: %s", s)

    return {
        "s_hat": s,
        "sim": sim_data,
        "obs": obs,
        "Ne": Ne,
        "true_af": afs,
        "prior": prior,
    }
