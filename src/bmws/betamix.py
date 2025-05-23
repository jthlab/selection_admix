"beta mixture with spikes model"

import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import NamedTuple, Union, Callable, Any
import operator

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
import jaxopt
from jax.nn import sigmoid
from jax.scipy.special import betaln, digamma, gammaln, logsumexp, xlog1py, xlogy, logit
from jax.scipy.stats import binom
from jax.experimental.sparse import BCOO
from loguru import logger
import optax
import optimistix as optx
import wrapt

import flowjax.flows
import flowjax.bijections
from flowjax.distributions import AbstractDistribution, Normal, Transformed
import gnuplotlib as gp

from bmws.data import Dataset
from .flsa import flsa
from .util import tree_stack, tree_unstack
import wadler_lindig as wl


Ne = 10_000
NUM_SAMPLES = 20_000

def _debug_print(fmtstr, *args):
    def pp(*args):
        print(fmtstr.format(*[wl.pformat(a, short_arrays=False) for a in args]))

    jax.debug.callback(pp, *args)


def cond(pred, f1, f2, *args, **kwargs):
    """
    A conditional function that takes a predicate and two functions.
    If the predicate is true, it calls f1 with the given arguments,
    otherwise it calls f2 with the same arguments.
    """
    if pred:
        return f1(*args, **kwargs)
    else:
        return f2(*args, **kwargs)


def scan(f, init, seq, reverse=False):
    """
    A scan function that applies a function f to an initial value and a sequence.
    If reverse is True, it applies the function in reverse order.
    """
    if reverse:
        seq = jax.tree_util.tree_map(lambda x: x[::-1], seq)
    accum = init
    ret = []
    batch_size = jax.tree.reduce(lambda x, y: x, jax.tree.map(lambda x: x.shape[0], seq))
    for i in range(batch_size):
        xi = jax.tree.map(lambda x: x[i], seq)
        accum, res = f(accum, xi)
        ret.append(res)
    ret = tree_stack(ret)
    if reverse:
        ret = jax.tree_util.tree_map(lambda x: x[::-1], ret)
    return accum, ret

if __debug__:
    print("Debug mode enabled. Using Python loops for scan and cond.")
    OPT_KW = dict(verbose=frozenset({"step_size", "loss"}))
else:
    scan = lax.scan
    cond = lax.cond
OPT_KW = dict(verbose=frozenset())

def sample_one_categorical(key, *, logits):
    'generate one sample from a categorical distribution with streaming'
    def body(accum, i):
        a, b, key = accum
        key, subkey = jax.random.split(key)
        g = jax.random.gumbel(subkey)
        k = g + logits[i]
        b = jnp.where(k > a, i, b)
        a = jnp.maximum(k, a)
        return (a, b, key), None

    (_, i_star, _) = lax.scan(body, (-jnp.inf, -1, key), jnp.arange(logits.shape[0]))[0]
    return i_star


def binom_logpmf(y: int, n: int, *, logit_p: float) -> float:
    """Log-likelihood of Binomial(y | n, sigmoid(logit_p)) without computing sigmoid."""
    log_binom_coeff = -betaln(n - y + 1, y + 1) - jnp.log(n + 1)
    return log_binom_coeff + y * logit_p - n * jnp.logaddexp(0.0, logit_p)


def random_binomial_large_N(key, N, p, no_boundary=True):
    assert N >= 1e4
    key0, key1, key2 = jax.random.split(key, 3)
    mu = N * p
    sigma2 = N * p * (1 - p)
    x = mu + jnp.sqrt(sigma2) * jax.random.normal(key0, shape=p.shape)
    y_norm = x.round(0).astype(int)
    y_pois0 = jax.random.poisson(key1, N * p, shape=p.shape)
    y_pois1 = jax.random.poisson(key2, N * (1 - p), shape=p.shape)
    ret = jnp.select(
        [N * p < 10., N * (1 - p) < 10.],
        [y_pois0, N - y_pois1],
        y_norm
    )
    ret = jnp.where(no_boundary, ret.clip(1, N - 1), ret)
    return ret


@jax.tree_util.register_dataclass
@dataclass
class Selection:
    T: float = field(metadata=dict(static=True))
    s: jnp.ndarray

    @property
    def t(self):
        assert self.s.ndim == 2
        M = len(self.s)
        return jnp.linspace(0, self.T, M)

    def __call__(self, xq, derivative=0):
        assert xq.ndim == 1
        t = self.t

        def f(si):
            return interpax.interp1d(xq, t, si, extrap=True)

        return vmap(f, in_axes=1, out_axes=1)(self.s)
        # return vmap(interpax.interp1d, in_axes=(None, 0, 1), out_axes=1)(t, xq, self.s)


    def roughness(self):
        x = jnp.linspace(0, self.T, self.s.shape[0])
        ds2 = self(x, derivative=1)
        return jnp.trapezoid(ds2 ** 2, x, axis=0).sum()

    def __call__(self, xq, derivative=0):
        assert self.s.ndim == 2
        assert xq.ndim == 1
        assert derivative == 0
        return self.s[xq]

    @classmethod
    def default(cls, T, K):
        s = np.zeros((T, K))
        return cls(T=T, s=s)

class EarlyStop(optx.AbstractMinimiser):
    solver: optx.AbstractMinimiser
    validation_func: Callable[[Any, Any], float]
    patience: int = 100

    def init(self, *args, **kwargs):
        state = self.solver.init(*args, **kwargs)
        # cast to jnp.array because eqx.is_array is used inside the stack to partition later on
        return dict(base_state=state, best_loss=jnp.array(jnp.inf), patience=jnp.array(self.patience))

    def step(self, fn, y, args, options, state, tags):
        y1, state['base_state'], aux = self.solver.step(fn, y, args, options, state['base_state'], tags)
        val_loss = self.validation_func(y, args)
        better = val_loss < state['best_loss']
        state['best_loss'] = jnp.where(better, val_loss, state['best_loss'])
        state['patience'] = jnp.where(better, self.patience, state['patience'] - 1)
        return y1, state, aux

    def terminate(self, fn, y, args, options, state, tags):
        stop, result = self.solver.terminate(fn, y, args, options, state['base_state'], tags)
        early_stop = state['patience'] <= 0
        stop = jnp.where(early_stop, True, stop)
        result = optx.RESULTS.where(early_stop, optx.RESULTS.successful, result)
        return stop, result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        return self.solver.postprocess(fn, y, aux, args, options, state['base_state'], tags, result)

    @property
    def atol(self):
        return self.solver.atol

    @property
    def rtol(self):
        return self.solver.rtol

    @property
    def norm(self):
        return self.solver.norm


class KDEDistribution(NamedTuple):
    data: Any
    weights: Any

    @staticmethod
    def _bw(kde):
        samples = kde.dataset.T
        weights = kde.weights

        if weights is None:
            weights = jnp.ones(len(samples)) / len(samples)

        # break the samples, weights into train, test
        T = int(len(samples) * 0.8)
        s_train, s_test = jnp.array_split(samples, [T])
        w_train, w_test = jnp.array_split(weights, [T])
        w_train /= jnp.sum(w_train)
        w_test /= jnp.sum(w_test)

        def f(h):
            kde_train = jax.scipy.stats.gaussian_kde(s_train.T, weights=w_train, bw_method=h)
            return -jnp.average(kde_train.logpdf(s_test.T), weights=w_test)
        
        # find the optimal bandwidth
        hs = jnp.geomspace(0.01, 0.5, 20)
        lls = lax.map(f, hs)
        h_star = hs[lls.argmin()]

        if __debug__:
            jax.debug.print("h_star:{}", h_star, ordered=True)

        return h_star

    @property
    def D(self):
        return self.data.shape[1]

    @property
    def _kde(self):
        return jax.scipy.stats.gaussian_kde(self.data.T, weights=self.weights, bw_method=self._bw)

    def refit(self, samples, weights=None):
        assert samples.ndim == 2

        if weights is None:
            weights = jnp.ones(len(samples)) / len(samples)

        return self.__class__(samples, weights)

    def log_pdf(self, x):
        return self._kde.logpdf(x)

    def sample(self, key, n_samples=1):
        return self._kde.resample(key, shape=(n_samples,)).T


# def logit(x):

# @jax.tree_util.register_dataclass
# @dataclass
# class Selection:
#     T: float = field(metadata=dict(static=True))
#     s: jnp.ndarray
# 
#     def __call__(self, xq, derivative=0):
#         assert self.s.ndim == 2
#         assert xq.ndim == 1
#         return self.s[xq]
# 
# 
#     def smoothness(self):
#         return jnp.sum(jnp.diff(self.s, 0) ** 2)
# 
#     @classmethod
#     def default(cls, T, K):
#         s = np.zeros((T, K))
#         return cls(T=T, s=s)


def p_prime(s, p):
    return (1 + s / 2) * p / (1 + s / 2 * p)

def logit_p_prime(s, *, logit_p):
    return logit_p + jnp.log1p(s / 2)

def filter_inexact(pytree):
    return eqx.filter(pytree, eqx.is_inexact_array)


class FlowHMM:
    def __init__(self, D, key=None):
        if key is None:
            key = jax.random.key(0)
        self._key = key
        self._D = D
        data = jax.random.beta(key, a=1., b=100., shape=(NUM_SAMPLES, D))
        self.prior = KDEDistribution(data, weights=jnp.ones(NUM_SAMPLES) / NUM_SAMPLES)
        # self.prior = flowjax.flows.masked_autoregressive_flow(
        #     key,
        #     base_dist=flowjax.distributions.Normal(loc=jnp.zeros(D)),
        #     flow_layers=3,
        #     nn_width=128,
        #     nn_activation=jax.nn.silu,
        # )
        # sig = flowjax.bijections.Sigmoid(shape=(D,))
        # self.model = Transformed(fl, sig)
        self.model_nc = eqx.filter(self.prior, eqx.is_inexact_array, inverse=True)

    def get_key(self):
        self._key, ret = jax.random.split(self._key)
        return ret

    def _fit_c(self, model_c, data, weights=None):
        if weights is None:
            weights = jnp.ones(len(data)) / len(data)

        return self.prior.refit(data, weights=weights)

        # split data into train and test
        s = int(len(data) * 0.8)
        train, test = jnp.array_split(data, [s])
        w_train, w_test = jnp.array_split(weights, [s])
        w_train = w_train / jnp.sum(w_train)
        w_test = w_test / jnp.sum(w_test)

        # objective
        def loss(model_c, args):
            data, weights, lam = args
            model = self.combine(model_c)
            with jax.debug_infs(True):
                lp = model.log_prob(data)
            l1 = -jnp.average(lp, weights=weights)
            l2 = jax.tree.reduce(operator.add, jax.tree.map(lambda x: jnp.sum(x ** 2), model_c))
            ret = l1 + lam * l2
            return jnp.where(~jnp.isfinite(ret), jnp.inf, ret)

        def val_loss(model_c, args):
            return loss(model_c, (test, w_test, 0.0))

        opt = optx.OptaxMinimiser(optax.adam(1e-3), rtol=1e-4, atol=1e-4, **OPT_KW)
        opt = EarlyStop(opt, val_loss, patience=100)

        with jax.debug_nans(True):
            soln = optx.minimise(loss, opt, model_c, args=(train, w_train, 1e-4), max_steps=None)

        if __debug__:
            jax.debug.print("soln stats: {}", soln.stats)
        return soln.value

    def _fit(self, model, data, weights=None):
        return self.combine(self._fit_c(self.filter(model), data, weights=weights))

    def fit_prior(self, ab):
        assert ab.shape == (2, self._D)
        key0, key1, key2 = jax.random.split(self.get_key(), 3)
        samples = logit(jax.random.beta(key0, *ab, shape=(NUM_SAMPLES, self._D)))
        self.prior = self.prior.refit(samples)

    def combine(self, model_c):
        return eqx.combine(model_c, self.model_nc)

    def filter(self, model):
        return eqx.filter(model, eqx.is_inexact_array)

    def transition(self, fc: AbstractDistribution, s: jnp.ndarray, key) -> AbstractDistribution:
        """Given a prior distribution on population allele frequency, compute posterior after
        one round of WF mating."""
        f = self.combine(fc)
        key, subkey = jax.random.split(key)
        logit_p = f.sample(subkey, NUM_SAMPLES)
        p = sigmoid(logit_p)
        pp = p_prime(s[None, :], p)
        n = random_binomial_large_N(key, 2 * Ne, pp)
        x = n / (2 * Ne)
        y = logit(x).clip(-1e2, 1e2)
        fr = self._fit(f, y)
        ret = self.filter(fr)
        return ret


    def binom_sampling(self, fc, n: int, d: int, theta: jnp.ndarray, key) -> AbstractDistribution:
        # prob(data | p) = f(p). so prob(p | data) ~ f(p) * pi(p)
        f = self.combine(fc)
        logit_p = f.sample(key, NUM_SAMPLES)

        @vmap
        def ll(lp):
            return logsumexp(binom_logpmf(d, n, logit_p=lp) + jnp.log(theta))


        loglik = ll(logit_p)
        # LL = log \int p(y|x) p(x) dx ~= log (1/n) \sum_i p(y|x_i) where x_i ~ p(x)
        lse = logsumexp(loglik)
        ll = lse - jnp.log(NUM_SAMPLES)
        weights = jnp.exp(loglik - lse)
        fp = self._fit(f, logit_p, weights)
        return self.filter(fp), ll


    def _forward_helper(self, accum, tup):
        model0_c, ll0, last_t, key = accum
        datum, s_t, i = tup
        key, subkey = jax.random.split(key)
        model1_c = cond(
            datum.t != last_t, 
            self.transition,
            lambda *args: args[0],
            model0_c, s_t, subkey
        )
        n, d = datum.obs
        model2_c, ll1 = cond(
            (datum.t == last_t) & (n > 0),
            self.binom_sampling,
            lambda *args: (args[0], 0.),
            model1_c, n, d, datum.theta, key
        )
        ll = ll0 + ll1
        accum = (model2_c, ll, datum.t, key)
        if __debug__:
            models = map(self.combine, [model0_c, model1_c, model2_c])
            samples = jnp.array([m.sample(key, 10_000) for m in models])
            samples = sigmoid(samples)
            means = [jnp.mean(s, axis=0) for s in samples]
            sems = [jnp.std(s, axis=0) / jnp.sqrt(len(s)) for s in samples]
            covs = [jnp.cov(s, rowvar=False) for s in samples]
            _debug_print("datum:{}\ns:{}\nmeans:\n{}\nsems:\n{}\ncovs:\n{}\n", datum, s_t, jnp.array(means), jnp.array(sems), jnp.array(covs))
        return accum, (model2_c, ll1)


    def forward(self, sln: Selection, prior_c, data: Dataset, key):
        """
        Run the forward algorithm for the BMwS model.

        Args:
            s: selection coefficient at each time point for each of the K populations (T - 1, K)
            Ne:  diploid effective population size at each time point for each of the K populations (T - 1, K)
            data: data to compute likelihood
            beta: prior distribution on allele frequencies

        Returns:
            Tuple (betas, lls). betas [T, K, M] are the filtering distributions, and lls are the conditional likelihoods.
        """
        s = sln(data.t)
        init = (prior_c, 0.0, data.t[0], key)
        seq = (data, s, jnp.arange(len(s)))

        (_, ll, _, _), (models, lls) = scan(
            self._forward_helper,
            init,
            seq
        )

        return models, dict(ll=ll)


    def sample_paths(self, sln: Selection, prior_c, data: Dataset, key, k: int=1):
        """
        Sample paths from the model.

        Args:
            models: list of models
            sln: selection coefficients
            data: data to sample from
            k: number of samples to draw

        Returns:
            Tuple (t, y). t is the time points, and y is the sampled paths.
        """
        key, subkey = jax.random.split(key)
        models, aux =  self.forward(sln, prior_c, data, subkey)
        keys = jax.random.split(key, k)
        ts, paths = vmap(lambda k: self._sample_path(models, sln, data, k))(keys)
        return ts[0], paths, aux


    def _sample_path(self, models, sln: Selection, data: Dataset, key):
        models = self.filter(models)
        model0, models = [
            jax.tree.map(lambda a: a[sl], models)
            for sl in [-1, slice(None, -1)]
        ]
        mask = data.t[:-1] != data.t[1:]
        models_mask, t_mask = jax.tree.map(lambda a: a[mask], (models, data.t[:-1]))
        key0, key1, key2 = jax.random.split(key, 3)
        l0 = self.combine(model0).sample(key0)[0]
        p0 = sigmoid(l0)
        # y0 = random_binomial_large_N(key1, 2 * Ne, p0).astype(int)

        init = (p0, key2)
        s = sln(t_mask)
        seq = (models_mask, s, t_mask)

        def f(accum, carry):
            pi, key = accum
            model_c, si, ti = carry
            model = self.combine(model_c)
            key, subkey = jax.random.split(key)
            # yi | y ~ binom(Ne, p(s, y / Ne))
            logit_p = model.sample(subkey, NUM_SAMPLES)
            logit_pp = logit_p_prime(si, logit_p=logit_p)
            yi = (pi * (2 * Ne)).astype(int)
            log_ps = binom_logpmf(yi, 2 * Ne, logit_p=logit_pp).sum(1)
            a = sample_one_categorical(subkey, logits=log_ps)
            logit_p = logit_p[a]
            p = sigmoid(logit_p)
            return (p, key), (ti, logit_p)

        t, samples = lax.scan(f, init, seq, reverse=True)[1]
        return jnp.append(t, data.t[-1]), jnp.concatenate([samples, l0[None]])
    

    def em(self, sln0: Selection, data: Dataset, alpha=1.0, em_iterations=5):
        def trans(logit_p1, p0, s):
            logit_pp = logit_p_prime(s=s, logit_p=logit_p1)
            # binom(2 * Ne, 2 * Ne * p0, pp) => 
            # lls = 2 * Ne * (xlogy(p0, pp) + xlog1py(1 - p0, 1. - pp)) + const
            # = 2 * Ne * binary_cross_entropy(label=p0, pp)

            @vmap
            def bce(label, logit):
                # binary cross entropy in logit domain
                return jnp.maximum(logit, 0) - logit * label + jnp.logaddexp(0.0, -jnp.abs(logit))

            # note: bce = loss = -ll
            return 2 * Ne * -bce(label=p0, logit=logit_pp).sum()


        def obj(sln, args):
            logit_paths, t = args
            s = sln(t[1:])  # t

            @vmap
            def f(logit_p1i, p0i):
                lls = vmap(trans)(logit_p1i, p0i, s)
                return lls.sum()

            lls = f(logit_paths[:, :-1], sigmoid(logit_paths[:, 1:]))
            BOUND = 0.1
            bound_pen = jax.nn.relu(jnp.abs(s) - BOUND).sum()
            ret = -lls.mean() + 100. * bound_pen
            return jnp.where(~jnp.isfinite(ret), jnp.inf, ret)

        def prox(sln, l1reg, scaling):
            prox_s = vmap(flsa, (1, None), 1)(sln.s, l1reg * scaling)
            return replace(sln, s=prox_s)

        opt = jaxopt.ProximalGradient(obj, prox)

        def step(sln, logit_paths, t):
            with jax.debug_nans(False):
                # return optx.minimise(obj, bfgs, sln, args=(logit_paths, t), max_steps=None).value
                res = opt.run(sln, hyperparams_prox=alpha, args=(logit_paths, t))
                return res.params

        def sample_paths(sln, prior_c, key):
            return self.sample_paths(sln, prior_c, data, key=key, k=NUM_SAMPLES)

        fit = self._fit_c

        if not __debug__:
            step, sample_paths, fit = map(jit, (step, sample_paths, fit))

        sln = sln0
        prior_c = self.filter(self.prior)

        lls = []
        last_lls = None

        for i in range(em_iterations):
            t, logit_paths, aux = sample_paths(sln, prior_c, self.get_key())
            lls.append(float(aux['ll']))
            sln = step(sln, logit_paths=logit_paths, t=t)
            logit_p0 = logit_paths[:, 0]
            paths = sigmoid(logit_paths).mean(0)
            s = sln(t)
            gp.plot(np.array(lls), _with="lines", title="lls", terminal="dumb 120,20", unset="grid")
            gp.plot(*[(t[::-1], y[::-1]) for y in paths.T], _with="lines", title="afs", terminal="dumb 120,30", unset="grid")
            gp.plot(*[(t[::-1], y[::-1]) for y in s.T], _with="lines", title="selection", terminal="dumb 120,30", unset="grid")
            self.prior = self.prior.refit(logit_p0)
            prior_c = self.filter(self.prior)

        return sln, self.combine(prior_c)


## TESTS
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(0)

def test_logit_p_prime(rng):
    p = rng.uniform(0, 1, (100,))
    s = rng.uniform(-1, 1, (100,))
    logit_p = logit(p)
    logit_p1 = logit_p_prime(s, logit_p=logit_p)
    p1 = sigmoid(logit_p1)
    p2 = p_prime(s, p)
    np.testing.assert_allclose(p1, p2)
