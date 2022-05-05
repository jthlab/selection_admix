import logging
from dataclasses import dataclass
from functools import partial
from typing import Union

import jax
import jaxopt
import numpy as np
import scipy.optimize
import scipy.stats
from jax import jit, lax, value_and_grad, numpy as jnp, tree_map, vmap
from jax.example_libraries.optimizers import adagrad

from bmws.betamix import BetaMixture, Dataset, _construct_prior, SpikedBeta, forward, loglik

logger = logging.getLogger(__name__)

logging.getLogger('absl').setLevel(logging.DEBUG)


def _prox_nuclear_norm(X, alpha=1.0, scaling=1.0):
    r'''
    argmin (1/2)||X - Y||_F^2 + alpha * scaling * ||X||_*
    '''
    # aka lasso on spectrum
    u, s, vt = jnp.linalg.svd(X, full_matrices=False, compute_uv=True)
    s_hat = jaxopt.prox.prox_lasso(s, alpha, scaling)
    return u @ jnp.diag(s_hat) @ vt


def _obj(s, Ne, data: Dataset, prior: BetaMixture, lam, C):
    ll = loglik(s, Ne, data, prior)
    ret = -ll + lam * (jnp.diff(s, axis=1) ** 2).sum()
    # from jax.experimental.host_callback import id_print
    # _, ret = id_print((s, ret), what="s/ret")
    return ret / C

obj = jit(value_and_grad(_obj))


def beta_pdf(x, a, b):
    from jax.scipy.special import betaln, xlog1py, xlogy

    # assumes a, b >= 1.
    x01 = jnp.isclose(x, 0.0) | jnp.isclose(x, 1.0)
    safe_x = jnp.where(x01, 0.5, x)
    return jnp.where(
        x01,
        0.0,
        jnp.exp(xlogy(a - 1, safe_x) + xlog1py(b - 1, -safe_x) - betaln(a, b)),
    )


@partial(vmap, in_axes=(0, 0, None))
def _interp(a, b, M) -> BetaMixture:
    return BetaMixture.interpolate(lambda x: beta_pdf(x, a, b), M, norm=True)


@dataclass
class _Optimizer:
    # cache optimizer objects to prevent recompiles
    _instance = None
    M: int

    def __post_init__(self):
        def _eb_loss(log_ab, s, Ne, data):
            a, b = 1.0 + jnp.exp(log_ab)
            prior = _interp(a, b, self.M)
            return _obj(s, Ne, data, prior, 0.0, 1.0)

        self._eb_opt = jit(jaxopt.GradientDescent(_eb_loss, implicit_diff=False).run)
        self._ll_opt = jit(jaxopt.ProximalGradient(
            fun=_obj, prox=_prox_nuclear_norm, implicit_diff=False).run)

    def run_eb(self, log_ab0, s, Ne, data):
        res = self._eb_opt(log_ab0, s=s, Ne=Ne, data=data)
        a_star, b_star = 1. + jnp.exp(res.params)
        return _interp(a_star, b_star, self.M)

    def run_ll(self, s0, lam, gamma, Ne, data, prior):
        f, df = obj(s0, Ne, data, prior, lam, 1.)
        C = abs(df).max() / 0.2  # scale so that a stepsize of 1 results in a change of at most |0.2| in s
        res = self._ll_opt(s0, hyperparams_prox=gamma, C=C, lam=lam, Ne=Ne, data=data, prior=prior)
        return res.params


    @classmethod
    def factory(cls, M: int) -> "_Optimizer":
        if _Optimizer._instance is None:
            _Optimizer._instance = _Optimizer(M)
        return _Optimizer._instance





def empirical_bayes(s, data: Dataset, Ne, M, num_steps=100, learning_rate=1.0) -> BetaMixture:
    "maximize marginal likelihood w/r/t prior hyperparameters"
    opt = _Optimizer.factory(M)
    params0 = jnp.zeros([2, data.K])
    return opt.run_eb(params0, s, Ne, data)


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
    em_iterations: int = 3,
    lam: float = 1.0,
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
    prior: BetaMixture,
    lam: float = 1.0,
    gamma: float = 1.0,
):
    assert prior.a.ndim == 2  # [K, M]
    assert prior.a.shape[0] == data.K
    M = prior.a.shape[1]
    opt = _Optimizer.factory(M)
    s0 = np.zeros([data.T - 1, data.K])
    return opt.run_ll(s0, lam, gamma, Ne, data, prior)


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


@partial(jit, static_argnums=(3, 4))
def sample_paths(
    s: np.ndarray,
    Ne: np.ndarray,
    obs: np.ndarray,
    k: int,
    seed: int=1,
    prior: Union[int, BetaMixture] = BetaMixture.uniform(100),
):
    """
    Sample allele frequency paths from posterior distribution.

    Args:
        s: selection coefficient at each time point (T - 1,)
        Ne:  diploid effective population size at each time point (T - 1,)
        obs: (sample size, # derived alleles) observed at each time point (T, 2)
        k: number of paths to sample
        seed: seed for random number generator
        prior: prior on initial allele frequency

    Returns:
        Array of shape (k, T), containing k samples from the allele frequency posterior.

    Notes:
        - obs[0] denotes the most recent observation; obs[-1] is the most ancient.
        - s and Ne control the Wright-Fisher transitions that occur *between* each time points.
          Therefore, there they have one less entry than the number of observations.
    """
    rng = jax.random.PRNGKey(seed)

    prior = _construct_prior(prior)

    def _sample_spikebeta(beta: SpikedBeta, rng, cond):
        # -1, M represent the special fixed states 0/1
        log_p0p = beta.log_p0 + jnp.log(cond != -1)
        log_p1p = beta.log_p1 + jnp.log(cond != -2)
        log_p = jnp.concatenate(
            [log_p0p[None], log_p1p[None], beta.log_r + beta.f_x.log_c]
        )
        sub1, sub2 = jax.random.split(rng)
        s = jax.random.categorical(sub1, log_p) - 2
        x = jnp.where(
            s < 0,
            jnp.take(jnp.array([0.0, 1.0]), 2 + s),
            jax.random.beta(sub2, beta.f_x.a[s], beta.f_x.b[s]),
        )
        return (s, x)

    (betas, beta_n), _ = forward(s, Ne, obs, prior)

    beta0 = tree_map(lambda a: a[0], betas)
    beta1n = tree_map(
        lambda a, b: jnp.concatenate([a[1:], b[None]]), betas, beta_n
    )
    betas = tree_map(lambda a, b: jnp.concatenate([a, b[None]]), betas, beta_n)

    def _f(tup, beta):
        rng, s1 = tup
        rng, sub1, sub2 = jax.random.split(rng, 3)
        s, x = _sample_spikebeta(beta, sub1, s1)
        return (rng, s), x

    def _g(rng, _):
        rng, sub = jax.random.split(rng)
        s0, x0 = _sample_spikebeta(beta0, sub, 0)
        _, xs = jax.lax.scan(_f, (sub, s0), beta1n)
        return rng, jnp.concatenate([x0[None], xs])

    _, ret = jax.lax.scan(_g, rng, None, length=k)
    return ret, betas