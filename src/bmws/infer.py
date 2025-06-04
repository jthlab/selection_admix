import timeit
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import gnuplotlib as gp
import interpax
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optimistix as optx
from jax import jit, vmap

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
    def t(self):
        assert self.s.ndim == 2
        M = len(self.s)
        return jnp.linspace(0, self.T, M)

    def __call__(self, xq, derivative=0):
        assert xq.ndim == 1

        def f(si):
            return interpax.interp1d(xq, self.t, si, extrap=True)

        return vmap(f, in_axes=1, out_axes=1)(self.s)

    def roughness(self, order=2):
        t = self.t
        s = self(t)
        ds = jnp.diff(s, axis=0)
        return jnp.sum(jnp.abs(s)) + jnp.sum(ds**2)

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


def em(
    sln0: Selection,
    data: Dataset,
    alpha,
    em_iterations,
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
        paths, t = args
        s = sln(t[1:])  # t

        @vmap
        def f(n0, n1):
            return vmap(trans)(n0, n1, s).sum()

        lls = f(paths[:, :-1], paths[:, 1:])
        BOUND = 0.1
        bound_pen = jax.nn.relu(jnp.abs(s) - BOUND).sum()
        ret = -lls.mean() + 1e3 * bound_pen + alpha * sln.roughness()
        return jnp.where(~jnp.isfinite(ret), jnp.inf, ret)

    def prox(sln, l1reg, scaling):
        prox_s = vmap(flsa, (1, None), 1)(sln.s, l1reg * scaling)
        return replace(sln, s=prox_s)

    bfgs = optx.BFGS(1e-4, 1e-4)
    opt = jaxopt.ProximalGradient(obj, prox)

    def step(sln, paths, t):
        with jax.debug_nans(True):
            return optx.minimise(obj, bfgs, sln, args=(paths, t), max_steps=None).value
        # res = opt.run(sln, hyperparams_prox=alpha, args=(paths, t))
        # return res.params

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

    # em loop
    lls = []
    last_lls = None
    ret = []
    for i in range(em_iterations):
        assert prior[0].shape == (NUM_PARTICLES, data.K)
        key, subkey = jax.random.split(key)
        paths, t_diff, ll = sample_paths(
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
        gp.plot(
            *[(t_diff[::-1], y[::-1]) for y in p.T],
            _with="lines",
            title="afs",
            terminal="dumb 120,30",
            unset="grid",
        )
        if True:
            sln = step(sln, paths=paths, t=t_diff)
            s = sln(t_diff[:-1])
            gp.plot(
                *[(t_diff[::-1][:-1], y[::-1]) for y in s.T],
                _with="lines",
                title="selection",
                terminal="dumb 120,30",
                unset="grid",
            )
        print(lls)
        print(f"roughness: {sln.roughness(0)} {sln.roughness(1)} {sln.roughness(2)}")
        ret.append((sln, prior))

    return (sln, prior), {"paths": paths, "ret": ret}
