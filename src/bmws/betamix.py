"beta mixture with spikes model"
from functools import partial
from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.scipy.special import betaln, gammaln, logsumexp, xlog1py, xlogy


@partial(vmap, in_axes=(0, 0))
def _safe_lae(x, y):
    both = jnp.isneginf(x) & jnp.isneginf(y)
    x_safe = jnp.where(both, 1.0, x)
    y_safe = jnp.where(both, 1.0, y)
    return jnp.where(both, -jnp.inf, jnp.logaddexp(x_safe, y_safe))


def _scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in zip(*xs):
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)


class Dataset(NamedTuple):
    r"""
    A dataset of alleles sampled over time.

    Params:
        thetas: array of shape [T, N, K] giving the admixture loadings for each sample at each time point.
        obs: array of shape [T, N, 2]; obs[t, i, 0] is the number of alleles sampled from indiv. i at time t, while
            obs[t, i, 1] is the number of derived alleles that were observed.
    """

    thetas: jnp.ndarray
    obs: jnp.ndarray

    @property
    def K(self) -> int:
        assert self.thetas.ndim == 3
        return self.thetas.shape[2]

    @property
    def T(self) -> int:
        return self.thetas.shape[0]

    @property
    def N(self) -> int:
        N = self.obs.shape[1]
        assert N == self.thetas.shape[1]
        return N

    def resort(self) -> Tuple["Dataset", np.ndarray]:
        """
        Reorder the dataset such that the nonzero observations come first in epoch.

        Returns:
            Tuple (ds, nonzero_i) containing the resorted dataset, and a vector nonzero_i of shape [T]
            containing the index of the highest nonzero entry.
        """
        thetar = np.empty_like(self.thetas)
        obsr = np.empty_like(self.obs)
        nzi = np.zeros(self.T, dtype=int)
        for t in range(self.T):
            nz = self.obs[t, :, 0] > 0
            obsr[t] = np.concatenate([self.obs[t, nz], self.obs[t, ~nz]])
            thetar[t] = np.concatenate([self.thetas[t, nz], self.thetas[t, ~nz]])
            nzi[t] = nz.sum()
        return Dataset(obs=obsr, thetas=thetar), nzi

    @classmethod
    def from_single_pop(cls, obs):
        """Initialize a dataset from observations.

        Params:
            obs: Array of shape (T, 2). obs[t, 0] is the number of samples at time t,
                 obs[t, 1] is the number of derived alleles.
        """
        T = len(obs)
        thetas = jnp.ones([T, 1, 1])
        return cls(thetas=thetas, obs=obs[:, None])


def _logbinom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _wf_trans(s, N, a, b):
    # X` = Y / N where:
    #
    #     Y ~ Binomial(N, p') | 0 < Y < N ,
    #     p' = p + p(1-p)(s/2),
    #     p ~ Beta(a,b)
    #
    # E(Y | 0 < Y < N) = N(p' - p'^n)
    # EX' = Ep'(1-p'^{N-1})
    EX = (a * (2 + 2 * a + b * (2 + s))) / (2.0 * (a + b) * (1 + a + b))
    # EX = 0.5 * (
    #     (a * (2 + 2 * a + b * (2 + s))) / (a + b) / (1 + a + b)
    #     - (
    #         (2 * (a + b + N) + b * N * s)
    #         * jnp.exp(
    #             gammaln(a + b) + gammaln(a + N) - gammaln(a) - gammaln(1 + a + b + N)
    #         )
    #     )
    # )
    # var(X') = E var(X'|X) + var E(X' | X) = E p'(1-p')/N + var(x + x(1-x)*(s/2))

    # E(X'|X) = p'(1-p'^(N-1))
    # E p'(1-p') / N
    Evar = (
        (a * b * (4 - a * (-2 + s) + b * (2 + s)))
        / (2.0 * (a + b) * (1 + a + b) * (2 + a + b))
        / N
    )
    varE = (
        a
        * b
        * (
            4 * (1 + a + b) * (2 + a + b) * (3 + a + b)
            - 4 * (a - b) * (1 + a + b) * (3 + a + b) * s
            + (a + a**3 - a**2 * (-2 + b) + b * (1 + b) ** 2 - a * b * (2 + b)) * s**2
        )
    ) / (4.0 * (a + b) ** 2 * (1 + a + b) ** 2 * (2 + a + b) * (3 + a + b))
    var = Evar + varE
    # EX = E(p')
    # var = E(p'(1-p')/N) + var(p) + (s/2)^2 var(p(1-p)) + s * cov(p, p(1-p))
    u = EX * (1 - EX) / var - 1.0
    a1 = u * EX
    b1 = u * (1 - EX)
    return a1, b1


class BetaMixture(NamedTuple):
    """Mixture of beta pdfs:

    M = len(c) - 1
    p(x) = sum_{i=0}^{M} c[i] x^(a[i] - 1) (1-x)^(b[i] - 1) / beta(a[i], b[i])
    """

    a: np.ndarray
    b: np.ndarray
    log_c: np.ndarray

    @property
    def c(self):
        return jnp.exp(self.log_c)

    @classmethod
    def uniform(cls, M) -> "BetaMixture":
        return cls.interpolate(lambda x: 1.0, M)

    @classmethod
    def interpolate(cls, f, M, norm=False, z01=True) -> "BetaMixture":
        # bernstein polynomial basis:
        # sum_{i=0}^(M) f(i/M) binom(M,i) x^i (1-x)^(M-i)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) x^(i-1) (1-x)^(M-i+2-1)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) * beta(i,M-i+2) x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   = sum_{i=1}^(M+1) f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        a = 1 + z01 - 1
        b = M + 2 - z01 - 1
        i = jnp.arange(1, M + 2, dtype=float)
        c = vmap(f)((i[a:b] - 1) / M) / (1 + M)
        if z01:
            z = jnp.zeros(1)
            c = jnp.concatenate([z, c, z])
        c0 = jnp.isclose(c, 0.0)  # work around nans in gradients
        c_safe = jnp.where(c0, 1.0, c)
        log_c = jnp.where(c0, -jnp.inf, jnp.log(c_safe))
        if norm:
            log_c -= logsumexp(log_c)
        return cls(a=i, b=M - i + 2, log_c=log_c)

    @property
    def M(self):
        return self.log_c.shape[-1]

    @property
    def moments(self):
        "return a matrix A such that A @ self.c = [EX, EX^2] where X ~ self"
        a = self.a
        b = self.b
        EX = a / (a + b)  # EX
        EX2 = a * (a + 1) / (a + b) / (1 + a + b)  # EX2
        return jnp.array([EX, EX2 - EX**2])

    def __call__(self, x):
        return np.exp(
            self.log_c
            + xlogy(self.a - 1, x)
            + xlog1py(self.b - 1, -x)
            - betaln(self.a, self.b)
        ).sum()

    def plot(self, K=100, ax=None) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        x = np.linspace(0.0, 1.0, K)
        y = np.vectorize(self)(x)
        ax.plot(x, y)


class SpikedBeta(NamedTuple):
    log_p: jnp.ndarray  # spikes at 0 and 1
    f_x: BetaMixture  # abs. continuous component

    def sample_component(self, rng):
        sub1, sub2, sub3 = jax.random.split(rng, 4)
        i = jax.random.categorical(sub1, self.f_x.log_c)
        p = jax.random.beta(sub2, self.f_x.a[i], self.f_x.b[i])
        q = jnp.concatenate([self.log_p, self.log_r[None]])
        j = jax.random.categorical(sub3, q)
        return jnp.array([0.0, 1.0, p])[j]

    @property
    def log_r(self):
        # the probability of the absolutely continuous component
        return jnp.log1p(-jnp.exp(self.log_p).sum())

    @property
    def M(self):
        "The number of mixture components"
        return self.f_x.M

    def plot(self):
        import matplotlib.pyplot as plt

        self.f_x.plot()
        plt.bar(0.0, self.p0, 0.05, alpha=0.2, color="tab:red")
        plt.bar(1.0, self.p1, 0.05, alpha=0.2, color="tab:red")


def transition(
    f: SpikedBeta, s: jnp.ndarray, Ne: jnp.ndarray, data: Dataset, nzi: int
) -> SpikedBeta:
    """Given a prior distribution on population allele frequency, compute posterior after
    observing data at dt generations in the future"""
    # var(X') = E var(X'|X) + var E(X' | X)
    #         ~= var(X'|X = EX) + var E(X' | X)

    def lp(fi, si, Nei):
        # compute update spike probabilities and wf transition
        log_p, (a, b, log_c) = fi
        x = jnp.array([0, Nei])[:, None]  # [2, 1]
        log_p_c = logsumexp(
            log_c + betaln(x + a, Nei - x + b) - betaln(a, b), axis=1
        )  # [2] probability of loss or fixation in continuous component
        log_p = vmap(jnp.logaddexp, (0, 0))(log_p, log_p_c)
        a1, b1 = _wf_trans(si, Nei, a, b)
        return SpikedBeta(log_p, BetaMixture(a1, b1, log_c))

    fs = vmap(lp, (0, 0, 0))(f, s, Ne)
    # now process the observation
    ret = _binom_sampling_admix(fs, data, nzi)
    return ret


def _binom_sampling(n, d, f: SpikedBeta):
    log_p, (a, b, log_c) = f
    log_r = f.log_r
    a1 = a + d
    b1 = b + n - d
    # probability of the data given each mixing component
    log_p_mix = betaln(a1, b1) - betaln(a, b) + _logbinom(n, d)
    log_p01 = log_c + log_p_mix
    lp = log_p + jnp.log(d == jnp.array([0, n]))
    log_c1 = log_r + log_p01
    ll = logsumexp(jnp.concatenate([lp, log_c1]))
    # p(p=1|d) = p(d|p=1)*p01 / p(d)
    log_p = lp - ll
    # update mixing coeffs
    # p(c | data) = p(data | c) * p(c) / p(data)
    log_c1 -= logsumexp(log_c1)
    # posterior after binomial sampling --
    beta = SpikedBeta(log_p, BetaMixture(a1, b1, log_c1))
    return beta, ll


def _binom_sampling_admix(
    fs: SpikedBeta, data: Dataset, nzi: int
) -> Tuple[SpikedBeta, float]:
    """Compute filtering distributions after observing alleles.

    Params:
        fs: initial filtering distributions (arrays of shape [K] or [K, M])
        data: Dataset containing the observed data [N, 2]
        nzi: Index of highest nonzero entry in data array. (used to speed up computation.)
    """
    M = fs.M

    def apply(accum, tup):
        fs0, ll, i = accum

        def _f1(n, d, fs0, ll, theta) -> Tuple[SpikedBeta, float]:
            return (fs0, ll)

        def _f2(n, d, fs0, ll, theta) -> Tuple[SpikedBeta, float]:
            # compute action of binomial sampling across all populations
            fs1, ll1 = vmap(_binom_sampling, in_axes=(None, None, 0))(n, d, fs0)
            ll += logsumexp(jnp.log(theta) + ll1)

            # in each population k the posterior is
            # (1-theta[k]) * [original mixture] + theta[k] * [new mixture]
            # combine these into one and retain the M mixture components with the highest weights
            def _combine(t: float, beta0: SpikedBeta, beta1: SpikedBeta):
                lt = jnp.array([jnp.log1p(-t), jnp.log(t)])
                log_c1 = jnp.concatenate(
                    [lt[0] + beta0.f_x.log_c, lt[1] + beta1.f_x.log_c]
                )  # [2 * M]
                a1 = jnp.concatenate([beta0.f_x.a, beta1.f_x.a])
                b1 = jnp.concatenate([beta0.f_x.b, beta1.f_x.b])
                r = (
                    log_c1.argsort()
                )  # take the M largest mixture components (out of 2M)
                top = r[M:]
                a1 = a1[top]
                b1 = b1[top]
                log_c1 = log_c1[top]
                log_c1 -= logsumexp(log_c1)  # renormalize
                # now update the spike probabilities
                log_p1 = _safe_lae(lt[0] + beta0.log_p, lt[1] + beta1.log_p)
                return SpikedBeta(log_p1, BetaMixture(a1, b1, log_c1))

            fs2 = vmap(_combine, (0, 0, 0))(theta, fs0, fs1)
            return (fs2, ll)

        theta, ob = tup
        n, d = ob
        # short-circuit to no-op (_f1) when there are no more nonzero observations.
        # fs2, ll = _f2(n, d, fs0, ll, theta)
        fs2, ll = lax.cond(i >= nzi, _f1, _f2, n, d, fs0, ll, theta)
        return (fs2, ll, i + 1), None

    # sequentially apply the above function to each observed individual.
    # expand a/b/log_c to have new mixture components
    (fs1, ll, _), _ = lax.scan(apply, (fs, 0.0, 0), data)
    # (fs1, ll, _), _ = _scan(apply, (fs, 0.0, 0), data)
    return (fs1, ll)


# @partial(jit, static_argnums=3)
def forward(s, Ne, data: Dataset, nzi: jnp.ndarray, beta: BetaMixture):
    """
    Run the forward algorithm for the BMwS model.

    Args:
        s: selection coefficient at each time point for each of the K populations (T - 1, K)
        Ne:  diploid effective population size at each time point for each of the K populations (T - 1, K)
        data: data to compute likelihood
        nzi: number of actual observations (upper bounded by N) at each epoch.

    Returns:
        Tuple (betas, lls). betas [T, K, M] are the filtering distributions, and lls are the conditional likelihoods.
    """
    T, N, K = data.thetas.shape
    assert s.shape == (T - 1, data.K)
    assert Ne.shape == (T - 1, data.K)
    assert data.obs.shape == (T, N, 2)

    def _f(accum, d):
        beta0, ll, i = accum
        beta, ll_i = transition(beta0, **d)
        return (beta, ll_i + ll, i + 1), beta

    ninf = jnp.full([data.K, 2], -100.0)
    pr = SpikedBeta(ninf, beta)
    beta0, ll0 = _binom_sampling_admix(pr, jax.tree.map(lambda x: x[-1], data), nzi[-1])

    if False:
        # compile-free loop for debugging
        betas = []
        f_x = beta0
        for i in range(1, 1 + len(Ne)):
            f_x, ll = _f(
                f_x,
                {
                    "Ne": Ne[-i],
                    "data": jax.tree.map(lambda x: x[-i - 1], data),
                    "s": s[-i],
                },
            )
            betas.append(f_x)
    else:
        data1 = jax.tree.map(lambda x: x[:-1], data)
        (_, ll, _), betas = lax.scan(
            _f,
            (beta0, ll0, 0),
            {"Ne": Ne, "data": data1, "s": s, "nzi": nzi[:-1]},
            reverse=True,
        )

    return (betas, beta0), ll


def loglik(s, Ne, data: Dataset, nzi: jnp.ndarray, beta0: BetaMixture):
    return forward(s, Ne, data, nzi, beta0)[1]


def _construct_prior(prior: Union[int, BetaMixture]) -> BetaMixture:
    if isinstance(prior, int):
        M = prior
        prior = BetaMixture.uniform(M)
    return prior
