import os
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial

import blackjax
import gnuplotlib as gp
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import Progress
from jax import grad, jit, lax, vmap
from jax.scipy.special import xlog1py, xlogy

import bmws.data

from .data import Dataset
from .pf import backward_trace, forward_filter
from .timer import timer


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
        s = np.zeros((10, K))
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
    def default(cls, T, K,s0=0.05):
        s = np.zeros((T, K))+s0
        return cls(T=T, s=s)


Selection = SplineSelection


def sample_paths(sln, prior, z, obs, t, num_paths, ref_path, mean_path, N_E, key):
    subkey1, subkey2 = jax.random.split(key)
    particles, log_weights = prior
    P, D = particles.shape
    alpha, gamma, ancestors, loglik = map(
        np.asarray,
        forward_filter(
            z,
            obs,
            sln(t),
            particles,
            log_weights,
            ref_path,
            mean_path,
            N_E,
            subkey1,
        ),
    )
    seed = jax.random.randint(subkey2, (), 0, 2**31 - 1)
    path = backward_trace(alpha, gamma, ancestors, int(seed))
    # paths[:, 0] corresponds to alpha[-1], i.e. t=0
    # reverse the paths so that the time corresponds to the time array, i.e. in reverse order (t=T, T-1, ..., 0)
    path = path[::-1]
    ref_path[:] = path
    return path, dict(alpha=alpha, gamma=gamma, ancestors=ancestors, loglik=loglik)


def gibbs(
    sln0: Selection,
    data: Dataset,
    alpha,
    niter,
    M,
    seed=42,
    N_E=1e4,
):
    # organize observations into lists of ragged arrays
    br = np.where(data.t[1:] != data.t[:-1])[0]

    def f(x):
        lst = np.array_split(x, br + 1)
        lst[1:] = [x[1:] for x in lst[1:]]
        return lst

    obs, thetas = map(f, (data.obs, data.theta))
    t = np.arange(data.T)[::-1]
    assert t.shape[0] == len(obs) == len(thetas)
    padded_obs = []
    padded_thetas = []
    N_max = max(len(ob) for ob in obs)
    for ob, th in zip(obs, thetas):
        obp, thp = [
            np.pad(x, [(0, N_max - len(x)), (0, 0)], constant_values=0)
            for x in (ob, th)
        ]
        padded_obs.append(obp)
        padded_thetas.append(thp)

    obs = np.array(padded_obs, dtype=np.int32)
    thetas = np.array(padded_thetas, dtype=np.float32)

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
        alpha, beta, paths = args
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

    # if not __debug__:
    #     step = jit(step)

    key = jax.random.PRNGKey(seed)
    sln = sln0

    if True:
        key, subkey = jax.random.split(key)
        bootstrap_paths = bmws.data.bootstrap_paths(
            data, NUM_PARTICLES, subkey.tolist()
        )
    # make time run backwards
    bootstrap_paths = bootstrap_paths[:, ::-1]
    bootstrap_paths = (
        (2 * N_E * bootstrap_paths).clip(1, 2 * N_E - 1).astype(np.int32)
    )  # scale to 2 * N_E
    particles = bootstrap_paths[:, 0]
    log_weights = np.full(NUM_PARTICLES, -np.log(NUM_PARTICLES))  # uniform prior
    ref_path = bootstrap_paths.mean(0).astype(np.int32)
    mean_path = ref_path.copy()
    assert particles.shape == (NUM_PARTICLES, data.K)
    prior = (particles, log_weights)

    # prepare thetas
    thetas = [th.clip(1e-5, 1 - 1e-5) for th in thetas]
    thetas = [th / th.sum(1, keepdims=True) for th in thetas]

    # initialize z to prior
    z = []
    D = data.K
    I_D = np.eye(data.K, dtype=np.int32)
    for th in thetas:
        keys = jax.random.split(key, len(th) + 1)
        key = keys[0]
        zti = jax.vmap(lambda sk, p: jax.random.choice(sk, D, p=p))(keys[1:], th)
        z.append(I_D[zti])

    z = jnp.array(z, dtype=np.int32)  # [N, D]

    key, subkey = jax.random.split(key)
    path, _ = sample_paths(
        sln, prior, z, obs, t, NUM_PARTICLES, ref_path, mean_path, N_E, subkey
    )
    beta = alpha / 100.0

    # a, b such that gamma with mean=alpha and variance=sqrt alpha
    # a*b = alpha
    # a*b**2 = sqrt(alpha)
    # alpha*b = sqrt(alpha) => b=1/sqrt(alpha)
    prior_alpha_a = prior_alpha_b = jnp.sqrt(alpha)
    prior_beta_a = prior_beta_b = jnp.sqrt(beta)

    @timer
    @jit
    def step(sln, alpha, beta, path, key):
        def logdensity(x):
            return -obj(x, (alpha, beta, path[None]))

        # HMC
        # warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        # key, warmup_key, sample_key = jax.random.split(key, 3)
        # (state, warmup_parameters), _ = warmup.run(warmup_key, sln, num_steps=1000)

        df = grad(logdensity)(sln)
        infnorm = partial(jnp.linalg.norm, ord=jnp.inf)
        C = jax.tree.reduce(max, jax.tree.map(infnorm, df))

        # mcmc = blackjax.nuts(logdensity, **warmup_parameters)
        mcmc = blackjax.mala(logdensity, 1e-3 / C)
        state = mcmc.init(sln)

        def body(state, key):
            return mcmc.step(key, state)

        key0, key1, key2 = jax.random.split(key, 3)
        last_state, info = lax.scan(body, state, jax.random.split(key0, 2))
        jax.debug.print("{}", info.acceptance_rate)
        # sample alpha conditionally
        a_alpha = prior_alpha_a + sln.M / 2
        a_beta = prior_beta_a + sln.M / 2
        b_alpha = prior_alpha_b + sln.roughness() / 2
        b_beta = prior_beta_b + jnp.abs(sln.s).sum() / 2
        alpha = jax.random.gamma(key1, a=a_alpha, shape=()) * b_alpha
        beta = jax.random.gamma(key2, a=a_beta, shape=()) * b_beta
        return last_state.position, alpha, beta

    # em loop
    ret = []
    logliks = []

    with Progress() as progress:
        task = progress.add_task("MCMC...", total=niter)
        for i in range(niter):
            assert prior[0].shape == (NUM_PARTICLES, data.K)
            key, subkey = jax.random.split(key)
            path, aux = sample_paths(
                sln, prior, z, obs, t, NUM_PARTICLES, ref_path, mean_path, N_E, subkey
            )
            logliks.append(aux["loglik"])

            new_z = []

            for zt, pt, th, ob in zip(z, path, thetas, obs):
                n = ob[:, 0]  # [N]
                d = ob[:, 1]
                u = np.sum(n * d, axis=0)  # [N, D]
                v = np.sum(n * (1 - d), axis=0)
                x = pt / 2 / N_E
                log_p = jnp.log(th) + xlogy(u, x) + xlog1py(v, -x)
                key, subkey = jax.random.split(key)
                zti = jax.random.categorical(subkey, logits=log_p, axis=1)  # N
                new_z.append(I_D[zti])

            z = np.array(new_z, dtype=np.int32)  # [N, D]
            key, subkey = jax.random.split(key)
            sln, alpha, beta = step(sln, alpha=alpha, beta=beta, path=path, key=subkey)

            # convert these to numpy to spill cpu instead of gpu memory
            # should help with gpu oom errors
            ret.append(jax.tree.map(np.array, (sln, path, z)))

            # check if /tmp/break exists, break if so, and delete the file
            if os.path.exists("/tmp/break"):
                with suppress(FileNotFoundError):
                    # in case the file is deleted in the meantime
                    os.remove("/tmp/break")
                breakpoint()
            if os.path.exists("/tmp/stop"):
                with suppress(FileNotFoundError):
                    os.remove("/tmp/stop")
                sys.exit(1)
            progress.update(task, advance=1)
            if i % 10 == 0:
                p = path / 2 / N_E
                gp.plot(
                    *[(t[::-1], y[::-1]) for y in p.T],
                    _with="lines",
                    title="afs",
                    yrange=(0.0, 1.0),
                    terminal="dumb 120,30",
                    unset="grid",
                )
                s = sln(t[:-1])
                gp.plot(
                    *[(t[::-1][:-1], y[::-1]) for y in s.T],
                    _with="lines",
                    title="selection",
                    yrange=(-0.1, 0.1),
                    terminal="dumb 120,30",
                    unset="grid",
                )
            # print(lls)
            # print(f"{sln.roughness()=} {alpha=} {beta=}")

    slns, paths, zs = zip(*ret)
    return (slns, paths, zs), np.array(logliks)
