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
from jax.scipy.special import betaln, xlog1py, xlogy
from loguru import logger
from scipy.optimize import minimize

from bmws.betamix import BetaMixture, SpikedBeta, _construct_prior, forward, loglik
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
    # s: [T, K]
    ll = loglik(s, Ne, data, prior)
    temporal_diff = jnp.sum(jnp.diff(s, axis=0) ** 2)
    pairwise_diff = 0.5 * jnp.sum((s[:, None, :] - s[:, :, None]) ** 2)
    ret = -ll + alpha * temporal_diff + beta * pairwise_diff
    # _, ret = id_print((s.mean(axis=0), ret), what="s/ret")
    # jax.debug.print('ll:{} td:{} pd:{}', ll, temporal_diff, pairwise_diff)
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
            # jax.debug.print("eb_loss: ab:{} ret:{}", ab, ret)
            return ret

        opt = jaxopt.ProjectedGradient(
            fun=_eb_loss,
            projection=jaxopt.projection.projection_box,
            tol=0.1,
            implicit_diff=False,
            # unroll=True,
            # jit=False,
        )
        self._eb_opt = jit(opt.run)
        # self._eb_opt = opt.run

        opt = jaxopt.ProjectedGradient(
            fun=_obj,
            projection=jaxopt.projection.projection_box,
            implicit_diff=False,
            tol=0.1,
            # unroll=True,
            # jit=False,
        )
        # opt = jaxopt.ScipyBoundedMinimize(
        #     fun=_obj,
        # )
        self._ll_opt = jit(opt.run)
        # self._ll_opt = opt.run

    def run_eb(self, ab0, s, Ne, data, alpha, beta):
        lb = jnp.full_like(ab0, 1.0 + 1e-4)
        ub = jnp.full_like(ab0, 100.0)
        bounds = (lb, ub)
        res = self._eb_opt(
            # ab0, bounds=bounds, s=s, Ne=Ne, data=data, alpha=alpha, beta=beta
            ab0,
            hyperparams_proj=bounds,
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
        bounds = (jnp.full_like(s0, -0.2), jnp.full_like(s0, 0.2))
        res = self._ll_opt(
            s0,
            # bounds=bounds,
            hyperparams_proj=bounds,
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


def estimate_em(
    obs: np.ndarray,
    Ne: np.ndarray,
    alpha,
    beta,
    gamma,
    em_iterations: int = 3,
    solver_options: dict = {},
):
    M = 100
    s = np.zeros(len(obs) - 1)
    for i in range(em_iterations):
        prior = empirical_bayes(s, obs, Ne, M)
        s = estimate(obs, Ne, lam=lam, prior=prior, solver_options=solver_options)

    return s, prior


def estimate(
    data: Dataset,
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
    s0 = np.zeros([data.T, data.K])
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
        return _sample_path(betas, s, Ne, data, prior, key)

    return jax.vmap(f)(keys)


def _sample_path(
    betas,
    s: np.ndarray,
    Ne: np.ndarray,
    data: Dataset,
    prior: BetaMixture,
    key,
):
    def f(accum, seq):
        key, x_i1, Ne_i1, beta_i1 = accum
        beta_i, Ne_i, s_i, i, t = seq
        # sampling
        k = x_i1 * Ne_i1
        N = Ne_i1
        # sample from pdf which is proportional to beta_i(x) * p(x_i1 | x_i = x)
        # - use metropolis hastings since I don't know how to sample from this distribution
        # - we have p(x_i1 | x_i = x) = binom(N_i1 * x_i1; N_i1; f(x)) \propto f(x)^k (1-f(x))^(Ni1-k)
        #   where f(x) is the selectino operator.
        # - if s=0 then f(x)=x so the target density is just proportional to a beta mixture
        # - so use this as the base measure
        f_x_star = beta_i.f_x._replace(a=beta_i.f_x.a + k, b=beta_i.f_x.b + Ne_i1 - k)
        beta_star = beta_i._replace(f_x=f_x_star)

        # proposal distribution
        log_q = partial(beta_star, log=True)

        def log_pi(x):
            # target distribution
            y = (1 + s_i / 2) * x / (1 + s_i / 2 * x)
            return log_q(x) + jax.scipy.stats.binom.logpmf(k, N, y)

        def cond(tup):
            x_t, key, a, i = tup
            return (a < 100) & (i < 10_000)

        spikes = [k == 0, k == N]
        sample_q = partial(beta_star.sample, spikes=spikes)

        def mh(tup):
            x_t, key, a, i = tup
            keys = jax.random.split(key, 3)
            x_prime = sample_q(keys[0])
            log_alpha = log_pi(x_prime) - log_pi(x_t) + log_q(x_t) - log_q(x_prime)
            # U < alpha => log(u) < log_alpha => exp(1) > -log(alpha)
            accept = jax.random.exponential(keys[1]) > -log_alpha
            x_t1 = jnp.where(accept, x_prime, x_t)
            return (x_t1, keys[2], a + accept, i + 1)

        keys = jax.random.split(key, 3)
        init = (x_i1, keys[1], 0, 0)
        x, _, a, _ = lax.while_loop(cond, mh, init)
        # jax.debug.print("init:{} x1:{} x:{} Ne_i:{} s_i:{} t:{} a:{} i:{}", init[0], x_i1, x, Ne_i, s_i, t, a, i)
        return (keys[2], x, Ne_i, beta_i), x

    mask = jnp.append(data.t[1:] != data.t[:-1], True)
    t = data.t[mask]
    betas = jax.tree.map(lambda a: a[mask], betas)
    # the betas will be updated after each observation, but we only care about the
    # posterior at each distinct time point. thin the betas to only give the posterior
    # at each transition.
    beta_last, betas = [
        jax.tree.map(lambda a: a[sl], betas) for sl in (-1, slice(None, -1))
    ]

    @jit
    def sample_beta(beta0, betas, Ne, s):
        keys = jax.random.split(key, 2)
        x0 = beta0.sample(keys[0])
        init = (keys[1], x0, Ne[t[-1]], beta0)
        seq = (betas, Ne[t[:-1]], s[t[:-1]], jnp.arange(len(t) - 1), t[:-1])
        _, samples = lax.scan(f, init, seq, reverse=True)
        return samples

    samples = jax.vmap(sample_beta, in_axes=(0, 1, 1, 1))(beta_last, betas, Ne, s)
    return samples
