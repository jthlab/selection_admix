"beta mixture with spikes model"
from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.experimental.host_callback import id_print
from jax.tree_util import tree_map

id_print = lambda x, **kwargs: x
from jax.scipy.special import betaln, gammaln, logsumexp, xlog1py, xlogy


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
    def K(self):
        assert self.thetas.ndim == 3
        return self.thetas.shape[2]

    @property
    def T(self):
        return self.thetas.shape[0]

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
    a, b = id_print((a, b), what="ab")
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
            + (a + a**3 - a**2 * (-2 + b) + b * (1 + b) ** 2 - a * b * (2 + b))
            * s**2
        )
    ) / (4.0 * (a + b) ** 2 * (1 + a + b) ** 2 * (2 + a + b) * (3 + a + b))
    Evar, varE = id_print((Evar, varE), what="Evar/VarE")
    var = Evar + varE
    # EX = E(p')
    # var = E(p'(1-p')/N) + var(p) + (s/2)^2 var(p(1-p)) + s * cov(p, p(1-p))
    u = EX * (1 - EX) / var - 1.0
    u = id_print(u, what="u")
    a1 = u * EX
    b1 = u * (1 - EX)
    a1, b1 = id_print((a1, b1), what="a1b1")
    return a1, b1


class BetaMixture(NamedTuple):
    """Mixture of beta pdfs:

    M = len(c) - 1
    p(x) = sum_{i=0}^{M} c[i] x^(a[i] - 1) (1-x)^(b[i] - 1) / beta(a[i], b[i])
    """

    a: np.ndarray
    b: np.ndarray
    log_c: np.ndarray

    @classmethod
    def uniform(cls, M) -> "BetaMixture":
        return cls.interpolate(lambda x: 1.0, M)

    @classmethod
    def interpolate(cls, f, M, norm=False) -> "BetaMixture":
        # bernstein polynomial basis:
        # sum_{i=0}^(M) f(i/M) binom(M,i) x^i (1-x)^(M-i)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) x^(i-1) (1-x)^(M-i+2-1)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) * beta(i,M-i+2) x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   = sum_{i=1}^(M+1) f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        i = jnp.arange(1, M + 2, dtype=float)
        c = vmap(f)((i - 1) / M) / (1 + M)
        c0 = jnp.isclose(c, 0.0)  # work around nans in gradients
        c_safe = jnp.where(c0, 1.0, c)
        log_c = jnp.where(c0, -jnp.inf, jnp.log(c_safe))
        if norm:
            log_c -= logsumexp(log_c)
        return cls(a=i, b=M - i + 2, log_c=log_c)

    @property
    def M(self):
        return len(self.log_c) - 1

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
    log_p0: float
    log_p1: float
    f_x: BetaMixture

    def sample_component(self, rng):
        sub1, sub2, sub3 = jax.random.split(rng, 4)
        i = jax.random.categorical(sub1, self.f_x.log_c)
        p = jax.random.beta(sub2, self.f_x.a[i], self.f_x.b[i])
        j = jax.random.categorical(
            sub3, jnp.array([self.log_p0, self.log_p1, self.log_r])
        )
        return jnp.array([0.0, 1.0, p])[j]

    @property
    def log_r(self):
        return jnp.log1p(-jnp.exp(self.log_p0 + self.log_p1))

    @property
    def M(self):
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
    a, b, log_c = f.f_x
    # s = id_print(s, what="s")
    # probability mass for fixation. p(beta=0) = p0 + \int_0^1 f(x) (1-x)^n, and similarly for p1
    def lp(log_p0, log_p1, a, b, log_c, s, Ne):
        log_p0 += logsumexp(
            log_c + gammaln(a + b) + gammaln(b + Ne) - gammaln(b) - gammaln(a + b + Ne),
        )
        log_p1 += logsumexp(
            log_c + gammaln(a + b) + gammaln(a + Ne) - gammaln(a) - gammaln(a + b + Ne),
        )
        a1, b1 = _wf_trans(s, Ne, a, b)
        return log_p0, log_p1, a1, b1

    log_p0, log_p1, a1, b1 = vmap(lp, (0,) * 7)(f.log_p0, f.log_p1, a, b, log_c, s, Ne)
    fs = SpikedBeta(log_p0, log_p1, BetaMixture(a1, b1, log_c))
    # now model binomial sampling
    # p(d_k|d_{k-1},...,d_1) = \int p(d_k|x_k) p(x_k | d_k-1,..,d1)
    # probability of data arising from each mixing component
    # a1, b1 = id_print((a1, b1), what="wf_trans")
    from jax.experimental.host_callback import id_print

    # fs = id_print(fs, what="binom call")
    ret = _binom_sampling_admix(fs, data, nzi)
    # ret = id_print(ret, what="ret/trans")
    return ret


def _binom_sampling(n, d, f: SpikedBeta):
    log_p0, log_p1, (a, b, log_c) = f
    log_r = jnp.log1p(-jnp.exp(log_p0 + log_p1))
    log_r, _, _ = id_print((log_r, log_p0, log_p1), what="r/p0/p1")
    a1 = a + d
    b1 = b + n - d
    # probability of the data given each mixing component
    log_p_mix = betaln(a1, b1) - betaln(a, b) + _logbinom(n, d)
    log_p_mix = id_print(log_p_mix, what="p_mix")
    log_p01 = log_c + log_p_mix
    lp0 = log_p0 + jnp.log(d == 0)
    lp1 = log_p1 + jnp.log(d == n)
    ll = logsumexp(jnp.concatenate([lp0[None], log_r + log_p01, lp1[None]]))
    # p(p=1|d) = p(d|p=1)*p01 / pd
    log_p0 = lp0 - ll
    log_p1 = lp1 - ll
    # update mixing coeffs
    # p(c | data) = p(data | c) * p(c) / p(data)
    log_c1 = log_r + log_p01
    log_c1 -= logsumexp(log_c1)
    # posterior after binomial sampling --
    beta = SpikedBeta(log_p0, log_p1, BetaMixture(a1, b1, log_c1))
    # beta = id_print(beta, what="beta")
    return beta, ll


def _binom_sampling_admix(
    fs: SpikedBeta, data: Dataset, nzi: int
) -> Tuple[SpikedBeta, float]:
    """Compute filtering distributions after sampling a bunch of alleles.

    Params:
        fs: initial filtering distributions (arrays of shape [K] or [K, M])
        data: Dataset containing the observed data [N, 2]
    """
    a, b, _ = fs.f_x

    def apply(accum, tup):
        # accum, tup = id_print((accum, tup), what="accum/tup")
        fs0, ll, i = accum

        def _f1(n, d, fs0, ll, theta):
            return (fs0, ll)

        def _f2(n, d, fs0, ll, theta):
            fs1, ll1 = vmap(_binom_sampling, in_axes=(None, None, 0))(n, d, fs0)
            from jax.experimental.host_callback import id_print

            fs1, _, _, _ = id_print((fs1, (n, d), i, nzi), what="f2: fs1/n/d/i/nnz")
            ll += logsumexp(jnp.log(theta) + ll1)

            def log_comb(x, y):
                u = jnp.log1p(-theta) + x
                v = jnp.log(theta) + y
                # calling logsumexp with both entries negative leads to gradient nans
                simulinf = jnp.isneginf(u) & jnp.isneginf(v)
                u_safe = jnp.where(simulinf, 1.0, u)
                v_safe = jnp.where(simulinf, 1.0, v)
                ret = logsumexp(jnp.array([u_safe, v_safe]), axis=0)
                return jnp.where(simulinf, -jnp.inf, ret)

            # fs0 = id_print(fs0, what="fs0/lp0")
            lp0 = log_comb(fs0.log_p0, fs1.log_p0)
            # fs1 = id_print(fs0, what="fs1/lp1")
            lp1 = log_comb(fs0.log_p1, fs1.log_p1)
            # fs1 = id_print(fs0, what="fs2/beta")
            lc = log_comb(fs0.f_x.log_c.T, fs1.f_x.log_c.T).T
            b = fs0.f_x._replace(log_c=lc)
            fs2 = SpikedBeta(log_p0=lp0, log_p1=lp1, f_x=b)
            # ll = id_print((fs2,ll), what="expensive")
            return (fs2, ll)

        theta, ob = tup
        n, d = ob
        # try to optimize over when n is zero, even though we still do O(N) operations
        fs2, ll = lax.cond(i > nzi, _f1, _f2, n, d, fs0, ll, theta)
        return (fs2, ll, i + 1), None

    (fs2, ll, _), _ = lax.scan(apply, (fs, 0.0, 0), data)
    return (fs2, ll)


# @partial(jit, static_argnums=3)
def forward(s, Ne, data: Dataset, nzi: jnp.ndarray, prior: BetaMixture):
    """
    Run the forward algorithm for the BMwS model.

    Args:
        s: selection coefficient at each time point for each of the K populations (T - 1, K)
        Ne:  diploid effective population size at each time point for each of the K populations (T - 1, K)
        data: data to compute likelihood

    Returns:
        Tuple (betas, lls). betas [T, K, M] are the filtering distributions, and lls are the conditional likelihoods.
    """
    T, N, K = data.thetas.shape
    assert s.shape == (T - 1, data.K)
    assert Ne.shape == (T - 1, data.K)
    assert data.obs.shape == (T, N, 2)

    # from jax.experimental.host_callback import id_print
    # data = id_print(data, what="data")

    def _f(accum, d):
        beta0, ll = accum
        beta, ll_i = transition(beta0, **d)
        # beta, ll_i = id_print((beta, ll_i), what="_f")
        return (beta, ll_i + ll), beta

    ninf = jnp.full(data.K, -jnp.inf)
    pr = SpikedBeta(ninf, ninf, prior)
    beta0, ll0 = _binom_sampling_admix(pr, tree_map(lambda x: x[-1], data), nzi[-1])
    # beta0, ll0 = id_print((beta0, ll0), what="ret/init")

    if False:
        # compile-free loop for debugging
        lls = []
        betas = []
        f_x = beta0
        for i in range(1, 1 + len(Ne)):
            f_x, (_, ll) = _f(
                f_x,
                {"Ne": Ne[-i], "data": tree_map(lambda x: x[-i - 1], data), "s": s[-i]},
            )
            betas.append(f_x)
            lls.append(ll)
    else:
        data1 = tree_map(lambda x: x[:-1], data)
        (_, ll), betas = lax.scan(
            _f,
            (beta0, ll0),
            {"Ne": Ne, "data": data1, "s": s, "nzi": nzi[:-1]},
            reverse=True,
        )
    from jax.experimental.host_callback import id_print

    return (betas, beta0), ll


def loglik(s, Ne, data: Dataset, nzi: jnp.ndarray, prior: BetaMixture):
    return forward(s, Ne, data, nzi, prior)[1]


def _construct_prior(prior: Union[int, BetaMixture]) -> BetaMixture:
    if isinstance(prior, int):
        M = prior
        prior = BetaMixture.uniform(M)
    return prior
