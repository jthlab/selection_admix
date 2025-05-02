from dataclasses import dataclass
from functools import partial
from typing import Union

import equinox as eqx
import jax
import jaxopt
import numpy as np
import scipy.optimize
import scipy.stats
from jax import jacfwd, jit, lax
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.example_libraries.optimizers import adagrad
from jax.scipy.special import betaln, logsumexp, xlog1py, xlogy
from loguru import logger
from scipy.optimize import minimize

from bmws.betamix import (
    BetaMixture,
    SpikedBeta,
    _construct_prior,
    _transition,
    forward,
    loglik,
    safe_lae,
)
from bmws.data import Dataset


def _prox_nuclear_norm(X, alpha=1.0, scaling=1.0):
    r"""
    argmin (1/2)||X - Y||_F^2 + alpha * scaling * ||X||_*
    """
    # aka lasso on spectrum
    u, s, vt = jnp.linalg.svd(X, full_matrices=False, compute_uv=True)
    s_hat = jaxopt.prox.prox_lasso(s, alpha, scaling)
    return u @ jnp.diag(s_hat) @ vt


def _obj(s, Ne, data: Dataset, prior: BetaMixture, alpha, beta):
    # s: ((), [T, K])
    s_bar, ds = s
    assert ds.shape == Ne.shape
    ll = loglik(s_bar + ds, Ne, data, prior)
    temporal_diff = jnp.sum(jnp.diff(ds, axis=0) ** 2)
    pairwise_diff = jnp.sum(ds**2)
    ret = -ll + alpha * temporal_diff + beta * pairwise_diff
    # _, ret = id_print((s.mean(axis=0), ret), what="s/ret")
    jax.debug.print("ll:{} td:{} pd:{}", ll, temporal_diff, pairwise_diff)
    return ret


obj = jit(value_and_grad(_obj))


@jnp.vectorize
def _beta_logpdf(x, a, b):
    x0 = jnp.isclose(x, 0.0)
    x1 = jnp.isclose(x, 1.0)
    z = ((a > 1) & x0) | ((b > 1) & x1)
    x_safe = jnp.where(z, 0.5, x)
    r = xlogy(a - 1, x_safe) + xlog1py(b - 1, -x_safe) - betaln(a, b)
    return jnp.where(z, -jnp.inf, r)


@partial(vmap, in_axes=(0, 0, None))
def _interp(a, b, M) -> BetaMixture:
    return BetaMixture.interpolate(
        lambda x: _beta_logpdf(x, a, b), M, norm=True, log_f=True
    )


@dataclass
class _Optimizer:
    # cache optimizer objects to prevent recompiles
    _instance = None
    M: int

    def __post_init__(self):
        def _eb_loss(ab, s, Ne, data, alpha, beta):
            # ab: [2, K]
            a, b = ab
            prior = _interp(a, b, self.M)
            # prior = jax.vmap(lambda _: prior)(jnp.arange(data.K))
            ret = _obj(s, Ne, data, prior, alpha=alpha, beta=beta)
            jax.debug.print("eb_loss: ab:{} ret:{}", ab, ret)
            return ret

        opt = jaxopt.LBFGSB(fun=_eb_loss, maxiter=50, maxls=10)
        self._eb_opt = jit(opt.run)

        opt = jaxopt.LBFGSB(fun=_obj, maxiter=50, maxls=10)
        self._ll_opt = jit(opt.run)

    def run_eb(self, ab0, s, Ne, data, alpha, beta):
        # prevent weak_type so that compiles only happen once
        print(ab0, s, Ne, alpha, beta)
        ab0, s, Ne, alpha, beta = jax.tree.map(
            lambda a: jnp.asarray(a, dtype=jnp.float64), (ab0, s, Ne, alpha, beta)
        )
        data = data._replace(
            t=jnp.asarray(data.t, dtype=jnp.int64),
            theta=jnp.asarray(data.theta, dtype=jnp.float64),
            obs=jnp.asarray(data.obs, dtype=jnp.int64),
        )
        lb = jnp.full_like(ab0, 1.0 + 1e-4)
        ub = jnp.full_like(ab0, 100.0)
        bounds = (lb, ub)
        res = self._eb_opt(
            # ab0, bounds=bounds, s=s, Ne=Ne, data=data, alpha=alpha, beta=beta
            ab0,
            bounds=bounds,
            s=s,
            Ne=Ne,
            data=data,
            alpha=alpha,
            beta=beta,
        )
        logger.debug("eb result: {}", res)
        ab = a_star, b_star = res.params
        # a_star = vmap(lambda _: a_star)(jnp.arange(data.K))
        # b_star = vmap(lambda _: b_star)(jnp.arange(data.K))
        # a_star, b_star = res
        return ab, _interp(a_star, b_star, self.M)

    def run_ll(self, s0, alpha, beta, gamma, Ne, data, prior):
        s0, Ne, alpha, beta, gamma, prior = jax.tree.map(
            lambda a: jnp.asarray(a, dtype=jnp.float64),
            (s0, Ne, alpha, beta, gamma, prior),
        )
        data = data._replace(
            t=jnp.asarray(data.t, dtype=jnp.int64),
            theta=jnp.asarray(data.theta, dtype=jnp.float64),
            obs=jnp.asarray(data.obs, dtype=jnp.int64),
        )
        bounds = [jax.tree.map(lambda a: jnp.full_like(a, x), s0) for x in (-0.1, 0.1)]
        res = self._ll_opt(
            s0,
            bounds=bounds,
            alpha=alpha,
            beta=beta,
            Ne=Ne,
            data=data,
            prior=prior,
        )
        logger.debug("MLE result: {}", res)
        return res.params

    @classmethod
    def factory(cls, M: int) -> "_Optimizer":
        if cls._instance is None:
            logger.debug("creating new optimizer")
            cls._instance = _Optimizer(M)
        return cls._instance


def empirical_bayes(
    ab0, s, data: Dataset, Ne, M, num_steps=100, learning_rate=1.0, alpha=1.0, beta=1.0
) -> BetaMixture:
    "maximize marginal likelihood w/r/t prior hyperparameters"
    opt = _Optimizer.factory(M)
    return opt.run_eb(ab0, s, Ne, data, alpha=alpha, beta=beta)


@partial(jit, static_argnums=5)
def jittable_estimate(obs, Ne, lam, prior, learning_rate=0.1, num_steps=100):
    opt_init, opt_update, get_params = adagrad(learning_rate)
    params = jnp.zeros(len(Ne))
    opt_state = opt_init(params)

    @value_and_grad
    def loss_fn(s):
        return _obj(s, Ne, obs, prior, lam)

    def step(i, opt_state):
        value, grads = loss_fn(get_params(opt_state))
        opt_state = opt_update(i, grads, opt_state)
        return opt_state

    opt_state = lax.fori_loop(0, num_steps, step, opt_state)

    return get_params(opt_state)


def estimate(
    data: Dataset,
    s0,
    Ne: np.ndarray,
    alpha,
    beta,
    gamma,
    prior: BetaMixture,
):
    assert prior.a.ndim == 2  # [K, M]
    assert prior.a.shape[0] == data.K
    M = prior.a.shape[1]
    opt = _Optimizer.factory(M)
    return opt.run_ll(
        s0=s0, Ne=Ne, alpha=alpha, beta=beta, gamma=gamma, data=data, prior=prior
    )


def _prep_data(data):
    times, Ne, obs = zip(
        *sorted([(ob.t, ob.Ne, (ob.sample_size, ob.num_derived)) for ob in data])[::-1]
    )
    if len(times) != len(set(times)):
        raise ValueError("times should be distinct")
    obs = np.array(obs)
    assert np.all(obs[:, 1] <= obs[:, 0])
    assert np.all(obs[:, 1] >= 0)
    return np.array(Ne), obs, times


def sample_paths(
    s: np.ndarray,
    Ne: np.ndarray,
    data: Dataset,
    prior: BetaMixture,
    k: int = 1,
    seed: int = 1,
):
    """
    Sample allele frequency paths from posterior distribution.

    Args:
        s: selection coefficient at each time point (T - 1, K)
        Ne:  diploid effective population size at each time point (T - 1, K)
        data: Dataset object
        prior: prior on initial allele frequency
        k: number of paths to sample
        seed: seed for random number generator

    Returns:
        Array of shape (k, T), containing k samples from the allele frequency posterior.

    Notes:
        - s and Ne control the Wright-Fisher transitions that occur *between* each time points.
          Therefore, there they have one less entry than the number of observations.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), k)
    betas, _ = forward(s, Ne, data, prior)

    def f(key):
        return _sample_path(betas, s, Ne[0, 0].astype(int), data, prior, key)

    return jax.vmap(f)(keys)


def _sample_path(
    betas,
    s: np.ndarray,
    N: int,
    data: Dataset,
    prior: BetaMixture,
    key,
):

    # joint distribution of k_[t-1] and k_t
    K = jnp.arange(N + 1)

    def f(accum, seq):
        key, k_i1, beta_i1 = accum
        beta_i, s_i, i, t = seq

        # sample from joint distribution:
        # p(k_i, k_i1) = betabinom(beta1, k_i) * log

        def log_p(k):
            # probability of k[i+1] given k[i]
            x = k / N
            p = (1 + s_i / 2) * x / (1 + s_i / 2 * x)
            r1 = jax.scipy.stats.binom.logpmf(k_i1, N, p)
            # probability of k[i]
            r2 = (
                beta_i.log_r
                + beta_i.f_x.log_c
                + jax.scipy.stats.betabinom.logpmf(k, N, beta_i.f_x.a, beta_i.f_x.b)
            )
            r2 = r2.at[0].set(safe_lae(r2[0], beta_i.log_p[0]))
            r2 = r2.at[-1].set(safe_lae(r2[-1], beta_i.log_p[1]))
            r2 = logsumexp(r2)
            # overall probability
            return r1 + r2

        lp = vmap(log_p)(K)
        key, subkey = jax.random.split(key)
        k = jax.random.categorical(subkey, lp)
        return (key, k, beta_i), (t, k / N)

    mask = jnp.append(data.t[1:] != data.t[:-1], True)
    t = data.t[mask]
    betas = jax.tree.map(lambda a: a[mask], betas)
    # the betas will be updated after each observation, but we only care about the
    # posterior at each distinct time point. thin the betas to only give the posterior
    # at each transition.
    beta_last, betas = [
        jax.tree.map(lambda a: a[sl], betas) for sl in (-1, slice(None, -1))
    ]

    def sample_beta(beta0, betas, s, key):
        keys = jax.random.split(key, 2)
        p0 = beta0.sample(keys[0])
        k0 = jax.random.binomial(keys[0], N, p0).astype(int)
        init = (keys[1], k0, beta0)
        seq = (betas, s[t[:-1]], jnp.arange(len(t) - 1), t[:-1])
        _, (tt, samples) = lax.scan(f, init, seq, reverse=True)
        return jnp.append(samples, k0 / N)

    keys = jax.random.split(key, s.shape[1])
    samples = jax.vmap(sample_beta, in_axes=(0, 1, 1, 0))(beta_last, betas, s, keys)
    return samples[..., ::-1]
