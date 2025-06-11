import timeit
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import blackjax
import gnuplotlib as gp
import interpax
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optimistix as optx
import tqdm
from jax import jit, lax, vmap

import bmws.data

from .data import Dataset
from .flsa import flsa
from .pf import backward_sample_batched, forward_filter


@contextmanager
def timed(msg="Elapsed"):
    start = timeit.default_timer()
    yield
    end = timeit.default_timer()
    print(f"{msg}: {end - start:.6f}s")


@jax.tree_util.register_dataclass
@dataclass
class SplineSelection:
    T: float = field(metadata=dict(static=True))
    s: jnp.ndarray

    @property
    def M(self):
        assert self.s.ndim == 2
        return self.s.shape[0]

    @property
    def t(self):
        return jnp.linspace(0, self.T, self.M)

    def __call__(self, xq, derivative=0):
        assert xq.ndim == 1

        def f(si):
            return interpax.interp1d(xq, self.t, si, extrap=True)

        return vmap(f, in_axes=1, out_axes=1)(self.s)

    def roughness(self):
        ds = jnp.diff(self.s, n=2, axis=0)
        return jnp.sum(ds**2)

    @classmethod
    def default(cls, T, K):
        s = np.zeros((5, K))
        return cls(T=T, s=s)


@jax.tree_util.register_dataclass
@dataclass
class PiecewiseSelection:
    T: float = field(metadata=dict(static=True))
    s: jnp.ndarray

    @property
    def t(self):
        assert self.s.ndim == 2
        M = len(self.s)
        return jnp.linspace(0, self.T, M)

    def roughness(self):
        return 0.0  # computed via prox

    def __call__(self, xq, derivative=0):
        assert self.s.ndim == 2
        assert xq.ndim == 1
        assert derivative == 0
        return self.s[xq]

    @classmethod
    def default(cls, T, K):
        s = np.zeros((T, K))
        return cls(T=T, s=s)


Selection = SplineSelection


def sample_paths(sln, prior, data, num_paths, mean_paths, N_E, key):
    def get_seed():
        nonlocal key
        key, subkey = jax.random.split(key)
        return int(jax.random.randint(subkey, (), 0, 2**31 - 1))

    td = data.t[1:] != data.t[:-1]
    t_diff = np.r_[data.t[0], data.t[1:][td]]
    particles, log_weights = prior
    P, K = particles.shape
    T = len(t_diff)
    alpha = np.zeros((T, P, K), dtype=np.int32)
    gamma = np.zeros((T, P), dtype=np.float32)
    # have to convert to np.array because of buffer protocol stuff
    ll = np.zeros(1)
    theta = data.theta.clip(1e-5, 1 - 1e-5)
    theta /= theta.sum(1, keepdims=True)
    forward_filter(
        *map(
            np.array,
            (
                data.obs[:, 1],
                theta,
                sln(data.t),
                data.t,
                particles,
                log_weights,
                mean_paths,
            ),
        ),
        alpha,
        gamma,
        ll,
        N_E,
        get_seed(),
    )
    paths = backward_sample_batched(
        alpha, gamma, sln(t_diff), N_E, num_paths, get_seed()
    )
    mean_paths[:] = paths[0]
    # paths[:, 0] corresponds to alpha[-1], i.e. t=0
    # reverse the paths so that the time corresponds to the time array, i.e. in reverse order (t=T, T-1, ..., 0)
    paths = jnp.array(paths)[:, ::-1]
    return paths, t_diff, ll


def gibbs(
    sln0: Selection,
    data: Dataset,
    alpha,
    niter,
    M,
    mean_paths=None,
    seed=42,
    N_E=1e4,
):
    def binom_logpmf(n, N, p):
        p0 = jnp.isclose(p, 0.0)
        p1 = jnp.isclose(p, 1.0)
        p_safe = jnp.where(p0 | p1, 0.5, p)
        return jnp.select(
            [p0 & (n == 0), p0 & (n > 0), p1 & (n == N), p1 & (n < N)],
            [0.0, -jnp.inf, 0.0, -jnp.inf],
            jax.scipy.stats.binom.logpmf(n, N, p_safe),
        )

    def trans(n0, n1, s):
        p0 = n0 / 2 / N_E
        p_prime = (1 + s / 2) * p0 / (1 + s / 2 * p0)
        ret = binom_logpmf(n1, 2 * N_E, p_prime)
        # in rare cases, the sampler can produce paths that are impossible under the prior.
        # this can occur if the particle approximation is too coarse.
        # this makes the whole loglik equal -inf. so filter out these cases.
        ret = jnp.where((n0 == 0) & (n1 > 0), 0.0, ret)
        ret = jnp.where((n0 == 2 * N_E) & (n1 < 2 * N_E), 0.0, ret)
        return ret.sum()

    def obj(sln, args):
        alpha, beta, paths, t = args
        s = sln(t[1:])  # t

        @vmap
        def f(n0, n1):
            return vmap(trans)(n0, n1, s).sum()

        lls = f(paths[:, :-1], paths[:, 1:])
        BOUND = 0.1
        bound_pen = jax.nn.relu(jnp.abs(s) - BOUND).sum()
        ret = (
            -lls.mean()
            + 1e3 * bound_pen
            + alpha * sln.roughness()
            + beta * jnp.abs(sln.s).sum()
        )
        return jnp.where(~jnp.isfinite(ret), jnp.inf, ret)

    # assumption: data always starts and ends with an observation (not a transition)
    # illustration: t = [5, 5, 4, 3, 2, 1, 0, 0, 0]

    NUM_PARTICLES = M
    NUM_PATHS = NUM_PARTICLES

    # if not __debug__:
    #     step = jit(step)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    sln = sln0

    if mean_paths is None:
        mean_paths = bmws.data.mean_paths(data, NUM_PARTICLES)

    particles = (2 * N_E * mean_paths[:, -1]).astype(np.int32)  # scale to 2 * N_E
    log_weights = np.full(NUM_PARTICLES, -np.log(NUM_PARTICLES))  # uniform prior
    mean_paths = (2 * N_E * mean_paths.mean(0)).astype(np.int32)
    assert particles.shape == (NUM_PARTICLES, data.K)
    prior = (particles, log_weights)

    paths, t, ll = sample_paths(
        sln, prior, data, NUM_PARTICLES, mean_paths, N_E, subkey
    )

    beta = alpha

    # run warmup once
    def logdensity(x):
        return -obj(x, (alpha, beta, paths, t))

    # HMC
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    key, warmup_key, sample_key = jax.random.split(key, 3)
    (state, warmup_parameters), _ = warmup.run(warmup_key, sln, num_steps=1000)

    # a, b such that gamma with mean=alpha and variance=sqrt alpha
    # a*b = alpha
    # a*b**2 = sqrt(alpha)
    # alpha*b = sqrt(alpha) => b=1/sqrt(alpha)
    prior_a = prior_b = jnp.sqrt(alpha)

    @jit
    def step(sln, alpha, beta, paths, t, key):
        def logdensity(x):
            return -obj(x, (alpha, beta, paths, t))

        nuts = blackjax.nuts(logdensity, **warmup_parameters)
        state = nuts.init(sln)

        def body(state, key):
            state, _ = nuts.step(key, state)
            return state, None

        key0, key1, key2 = jax.random.split(key, 3)
        last_state, _ = lax.scan(body, state, jax.random.split(key0, 100))
        # sample alpha conditionally
        a = prior_a + sln.M / 2
        b_alpha = prior_b + sln.roughness() / 2
        b_beta = prior_b + jnp.abs(sln.s).sum() / 2
        alpha = jax.random.gamma(key1, a=a, shape=()) * b_alpha
        beta = jax.random.gamma(key2, a=a, shape=()) * b_beta
        return last_state.position, alpha, beta

    # em loop
    lls = []
    last_lls = None
    ret = []
    for i in tqdm.trange(niter):
        assert prior[0].shape == (NUM_PARTICLES, data.K)
        key, subkey = jax.random.split(key)
        paths, t, ll = sample_paths(
            sln, prior, data, NUM_PARTICLES, mean_paths, N_E, subkey
        )
        lls = np.append(lls, ll)
        # since time runs backwards (see above) paths[:, 0]
        # is the distribution at the most ancient time point
        prior = (
            paths[:, 0],
            np.full(NUM_PARTICLES, -np.log(NUM_PARTICLES), dtype=np.float32),
        )
        p = paths.mean(0) / 2 / N_E
        # gp.plot(
        #     *[(t[::-1], y[::-1]) for y in p.T],
        #     _with="lines",
        #     title="afs",
        #     terminal="dumb 120,30",
        #     unset="grid",
        # )
        key, subkey = jax.random.split(key)
        sln, alpha, beta = step(
            sln, alpha=alpha, beta=beta, paths=paths, t=t, key=subkey
        )
        s = sln(t[:-1])
        # gp.plot(
        #     *[(t[::-1][:-1], y[::-1]) for y in s.T],
        #     _with="lines",
        #     title="selection",
        #     terminal="dumb 120,30",
        #     unset="grid",
        # )
        # print(lls)
        # print(f"{sln.roughness()=} {alpha=} {beta=}")
        ret.append((sln, paths[0]))

    slns, paths = zip(*ret)
    return (slns, paths)
