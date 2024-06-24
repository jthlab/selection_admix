"beta mixture with spikes model"
from functools import partial
from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.special import betaln, gammaln, logsumexp, xlog1py, xlogy
from loguru import logger

from bmws.data import Dataset


def _breakpoint_if_nonfinite(x):
    all_finite = lambda x: jnp.isfinite(x).all()
    is_finite = jax.tree.reduce(lambda a, b: all_finite(a) & all_finite(b), x)
    def true_fn(x):
        return x
    def false_fn(x):
        jax.debug.breakpoint()
        return x
    return lax.cond(is_finite, true_fn, false_fn, x)


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
        return cls.interpolate(lambda x: 1.0, M, norm=True)

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
        return jnp.exp(
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
        # log(1 - exp(log_p).sum()) = log(1 - sum(exp(log_p)))
        # = logsumexp(0, log_p) = log1p(-exp(log_p).sum())
        a = jnp.concatenate([jnp.zeros(1), self.log_p])
        b = jnp.array([1, -1.0, -1.0])
        return logsumexp(a=a, b=b)

    @property
    def M(self):
        "The number of mixture components"
        return self.f_x.M

    def plot(self):
        import matplotlib.pyplot as plt

        self.f_x.plot()
        plt.bar(0.0, self.p0, 0.05, alpha=0.2, color="tab:red")
        plt.bar(1.0, self.p1, 0.05, alpha=0.2, color="tab:red")


def _transition(f: SpikedBeta, s: jnp.ndarray, Ne: jnp.ndarray) -> SpikedBeta:
    """Given a prior distribution on population allele frequency, compute posterior after
    observing data at dt generations in the future"""
    # var(X') = E var(X'|X) + var E(X' | X)
    #         ~= var(X'|X = EX) + var E(X' | X)

    @partial(vmap, in_axes=(0, 0, 0))
    def lp(fi, si, Nei):
        # compute update spike probabilities and wf transition
        log_p, (a, b, log_c) = fi
        log_r = fi.log_r
        x = jnp.array([0, Nei])[:, None]  # [2, 1]
        # probability mass for fixation. p(beta=0) = p0 + (1-p0) * \int_0^1 f(x) (1-x)^n, and similarly for p1
        log_p_c = logsumexp(
            log_r + log_c + betaln(x + a, Nei - x + b) - betaln(a, b), axis=1
        )  # [2] probability of loss or fixation in continuous component
        log_p1 = vmap(jnp.logaddexp, (0, 0))(log_p, log_p_c)
        a1, b1 = _wf_trans(si, Nei, a, b)
        # jax.debug.print("log_p:{} log_p1:{}", log_p, log_p1)
        return SpikedBeta(log_p1, BetaMixture(a1, b1, log_c))

    return lp(f, s, Ne)


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
    # clip to avoid numerical issues -- if p0/p1 ~= 1 then the model becomes degenerate
    log_p = jnp.clip(log_p, -100, -1e-5)
    beta = SpikedBeta(log_p, BetaMixture(a1, b1, log_c1))
    return beta, ll


def _binom_sampling_admix(fs: SpikedBeta, datum: Dataset) -> Tuple[SpikedBeta, float]:
    """Compute filtering distributions after observing alleles.

    Params:
        fs: initial filtering distribution
        data: Dataset containing the observed data point
    """
    fs0 = fs
    M = fs.M
    n, d = datum.obs
    fs1, llk = vmap(_binom_sampling, in_axes=(None, None, 0))(n, d, fs)
    ll = logsumexp(jnp.log(datum.theta) + llk)

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
        # approximate the mixture on 2*M components by one with M components
        if True:
            r = log_c1.argsort()  # take the M largest mixture components (out of 2M)
            top = r[M:]
            a1 = a1[top]
            b1 = b1[top]
            log_c1 = log_c1[top]
            log_c1 -= logsumexp(log_c1)  # renormalize
            i_star = a1.argsort()
            bm = jax.tree.map(lambda a: a[i_star], BetaMixture(a1, b1, log_c1))
        else:
            bm0 = BetaMixture(a1, b1, log_c1)
            bm = BetaMixture.interpolate(bm0, M - 1, norm=True)
        # now update the spike probabilities
        log_p1 = _safe_lae(lt[0] + beta0.log_p, lt[1] + beta1.log_p)
        return SpikedBeta(log_p1, bm)

    fs2 = vmap(_combine, (0, 0, 0))(datum.theta, fs0, fs1)
    return fs2, ll


# @partial(jit, static_argnums=3)
def forward(s: jnp.ndarray, Ne: jnp.ndarray, data: Dataset, beta: BetaMixture):
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
    s = jnp.array(s)
    Ne = jnp.array(Ne)
    assert Ne.ndim == 2
    T, K = Ne.shape
    assert s.shape == Ne.shape == (T, K)

    @jit
    def _f(accum, datum):
        beta0, ll0, last_t = accum
        t = datum.t
        trans = last_t != datum.t
        beta1 = _transition(beta0, s[t], Ne[t])
        beta2 = jax.tree.map(partial(jnp.where, trans), beta0, beta1)
        # now process the observation
        beta, ll1 = _binom_sampling_admix(beta2, datum)
        return (beta, ll0 + ll1, datum.t), beta

    ninf = jnp.full([data.K, 2], -100.0)
    pr = SpikedBeta(ninf, beta)
    data0 = jax.tree.map(lambda x: x[0], data)
    beta0, ll0 = _binom_sampling_admix(pr, data0)

    if False:
        logger.warning("Compiling unrolled forward loop!")
        # compile-free loop for debugging
        betas = []
        f_x = beta0
        ll = ll0
        t = data.t[0]
        for i in range(1, len(data.t)):
            (f_x, ll, t), _ = _f(
                (f_x, ll, t),
                jax.tree.map(lambda x: x[i], data),
            )
            betas.append(f_x)
    else:
        data1 = jax.tree.map(lambda x: x[1:], data)
        (_, ll, _), betas = lax.scan(
            _f,
            (beta0, ll0, data.t[0]),
            data1,
        )

    return (betas, beta0), ll


def loglik(s, Ne, data: Dataset, beta0: BetaMixture):
    return forward(s=s, Ne=Ne, data=data, beta=beta0)[1]


def _construct_prior(prior: Union[int, BetaMixture]) -> BetaMixture:
    if isinstance(prior, int):
        M = prior
        prior = BetaMixture.uniform(M)
    return prior
