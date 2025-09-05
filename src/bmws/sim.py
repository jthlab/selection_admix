import logging

import jax.numpy as jnp

from bmws.data import Dataset

from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def p_prime(s, p):
    return (1 + s / 2) * p / (1 + s / 2 * p)


def sim_wf(
    N: np.ndarray,
    s: np.ndarray,
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
    assert T == len(N)

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    f = np.zeros(T + 1)
    f[0] = f0
    for t in range(1, T + 1):
        p = p_prime(p=f[t - 1], s=s[t - 1])
        f[t] = rng.binomial(2 * N[t - 1], p) / (2 * N[t - 1])
    return f


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
    assert Ne.ndim == 2
    assert Ne.shape == (T, K)

    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    # simulate the wright-fisher model. all parameter vectors are reversed so that times runs from past to present.
    s = mdl["s"]
    f0 = mdl["f0"]
    afs = [
        sim_wf(Ne_i[::-1], s_i[::-1], f0_i, rng)[::-1]
        for Ne_i, s_i, f0_i in zip(Ne.T, s.T, f0.T)
    ]
    afs = np.transpose(afs)
    assert afs.shape == (T + 1, K)
    obs = []
    for t, theta, (n, _) in zip(*data):
        theta = np.array(theta, dtype=np.float64)
        theta /= np.sum(theta, dtype=np.float64)
        k = rng.choice(K, p=theta)
        p = afs[t, k]
        d = rng.binomial(n, p)
        obs.append((int(n), d))
    sim_data = data._replace(obs=jnp.array(obs))

    return sim_data, afs
