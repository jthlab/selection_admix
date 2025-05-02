"beta mixture with spikes model"
import os
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Union

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.special import betaln, digamma, gammaln, logsumexp, xlog1py, xlogy
from loguru import logger

from bmws.data import Dataset


@jax.tree_util.register_dataclass
@dataclass
class Selection:
    T: float = field(metadata=dict(static=True))
    s: jnp.ndarray

    def __call__(self, xq, derivative=0):
        M = len(self.s)
        t = jnp.linspace(0, self.T, M)
        return interpax.interp1d(xq, t, self.s, extrap=True)


def log1mexp(x):
    # log(1 - exp(x)), x < 0
    # x = eqx.error_if(x, x >= 0, msg="x >= 0")
    return jnp.where(x < -0.693, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def safe_lae(x, y):
    x_safe = jnp.where(jnp.isneginf(x), 1.0, x)
    y_safe = jnp.where(jnp.isneginf(y), 1.0, y)
    return jnp.select(
        [
            jnp.isneginf(x) & jnp.isneginf(y),
            jnp.isneginf(x) & ~jnp.isneginf(y),
            ~jnp.isneginf(x) & jnp.isneginf(y),
        ],
        [-jnp.inf, y, x],
        jnp.logaddexp(x_safe, y_safe),
    )


def safe_logsumexp(v):
    all_neginf = jnp.isneginf(v).all()
    v_safe = jnp.where(all_neginf, 0.0, v)
    return jnp.where(all_neginf, -jnp.inf, logsumexp(v_safe))


def _scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in zip(*xs):
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)


def _logbinom(n: int, k: int) -> float:
    """Compute the natural logarithm of the binomial coefficient.

    Args:
        n (int): The number of trials.
        k (int): The number of successes.

    Returns:
        float: The natural logarithm of the binomial coefficient.
    """
    return jnp.where(
        n > 0,
        -jnp.log(n + 1) - betaln(k + 1, n - k + 1),
        jnp.where(k == 0, 0.0, -jnp.inf),
    )


def _wf_trans(s, N, a, b):
    # X` = Y / N where:
    #
    #     Y ~ Binomial(N, p') | 0 < Y < N ,
    #     p' = p + p(1-p)(s/2),
    #     p ~ Beta(a,b)
    #
    # E(Y | 0 < Y < N) = N(p' - p'^n)
    # EX' = Ep'(1-p'^{N-1})
    mean = (a * (2 + 2 * a + b * (2 + s))) / (2.0 * (a + b) * (1 + a + b))
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
    return _inv_beta(mean, var)


def _inv_beta(mean, var):
    u = mean * (1 - mean) / var - 1
    a = mean * u
    b = (1 - mean) * u
    return a, b


class BetaMixture(NamedTuple):
    """Mixture of beta pdfs:

    M = len(c) - 1
    p(x) = sum_{i=0}^{M} c[i] x^(a[i] - 1) (1-x)^(b[i] - 1) / beta(a[i], b[i])
    """

    a: np.ndarray
    b: np.ndarray
    log_c: np.ndarray

    def top_k(self, k):
        """Return the top k components of the mixture"""
        log_c, i = lax.top_k(self.log_c, k)
        log_c -= logsumexp(log_c)
        return BetaMixture(self.a[i], self.b[i], log_c)

    def sample(self, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        i = jax.random.categorical(keys[0], self.log_c)
        return jax.random.beta(keys[1], self.a[i], self.b[i])

    @property
    def c(self):
        return jnp.exp(self.log_c)

    @classmethod
    def uniform(cls, M) -> "BetaMixture":
        return cls.interpolate(lambda x: 0.0, M, norm=True, log_f=True)

    @classmethod
    def interpolate(cls, f, M, norm=False, log_f=False) -> "BetaMixture":
        # bernstein polynomial basis:
        # sum_{i=0}^(M) f(i/M) binom(M,i) x^i (1-x)^(M-i)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) x^(i-1) (1-x)^(M-i+2-1)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) * beta(i,M-i+2) x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   = sum_{i=1}^(M+1) f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   we assume that f(0) = f(1) = 0 hence
        #   = sum_{i=2}^M f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        N = M + 1
        i = jnp.arange(2, N + 1, dtype=float)  # N - 1 => M + 1
        if log_f:
            log_c = vmap(f)((i - 1) / N) - jnp.log1p(N)
        else:
            c = vmap(f)((i - 1) / N) / (1 + N)
            c0 = jnp.isclose(c, 0.0)  # work around nans in gradients
            c_safe = jnp.where(c0, 1.0, c)
            log_c = jnp.where(c0, -jnp.inf, jnp.log(c_safe))
        if norm:
            log_c -= logsumexp(log_c)
        return cls(a=i, b=N - i + 2, log_c=log_c)

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
        return jnp.array([EX, EX2])

    @property
    def mean(self):
        return jnp.sum(self.moments[0] * self.c, -1)

    @property
    def var(self):
        return jnp.sum(self.moments[1] * self.c, -1) - self.mean**2

    def __call__(self, x, log=False):
        ret = logsumexp(
            self.log_c
            + xlogy(self.a - 1, x)
            + xlog1py(self.b - 1, -x)
            - betaln(self.a, self.b)
        )
        if not log:
            ret = jnp.exp(ret)
        return ret

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

    @classmethod
    def safe_init(cls, log_p, f_x):
        "ensure that the spikes sum up to strictly less than 1 and the mixture sums to one"
        log_p = jnp.clip(log_p, -jnp.inf, -1e-8)
        # if logaddexp(log_p) >= 0 then we want to multiply p by .99999 / p.sum().
        # equivalently, we want to add log(.99999 / p.sum()) = log(1-1e-8) - log_ps
        log_ps = safe_lae(log_p[0], log_p[1])
        safe_log_p = jnp.where(jnp.isneginf(log_ps), 1.0, log_p)
        safe_log_ps = jnp.where(jnp.isneginf(log_ps), 1.0, log_ps)
        log_p = jnp.where(
            log_ps >= 0,
            safe_log_p + jnp.log1p(-1e-8) - safe_log_ps,
            log_p,
        )
        # adjust the continuous weights to be 1 - lae(log_p)
        log_ps = safe_lae(log_p[0], log_p[1])
        # log_ps = eqx.error_if(log_ps, log_ps >= jnp.log1p(-1e-8), msg="log_ps >= 0")
        # logsumexp(f_x.log_c) = log1p(-exp(log_ps))
        log_c = f_x.log_c - logsumexp(f_x.log_c) + log1mexp(log_ps)
        return cls(log_p, f_x._replace(log_c=log_c))

    def sample(self, key: jax.random.PRNGKey, spikes: bool = None):
        if spikes is None:
            spikes = [True, True]
        keys = jax.random.split(key, 2)
        f = self.f_x.sample(keys[0])
        a = jnp.array([0.0, 1.0, f])
        p_spike = jnp.exp(self.log_p)
        p_spike *= jnp.array(spikes)
        p = jnp.append(p_spike, 1 - p_spike.sum())
        p /= p.sum()
        return jax.random.choice(keys[1], a, p=p)

    def __call__(self, x, log: bool = False):
        ret = jnp.select(
            [jnp.isclose(x, 0.0), jnp.isclose(x, 1.0)],
            [self.log_p[0], self.log_p[1]],
            self.log_r + self.f_x(x, log=True),
        )
        if not log:
            ret = jnp.exp(ret)
        return ret

    # def mean(self):
    #     p = self.p
    #     return p @ jnp.array([0.0, 1.0]) + (1 - p.sum()) * self.f_x.moments[0]

    @property
    def mean(self):
        p = jnp.exp(self.log_p)
        return p @ jnp.array([0.0, 1.0]) + (1 - p.sum(-1)) * self.f_x.mean

    @property
    def EX2(self):
        p = jnp.exp(self.log_p)
        return p @ jnp.array([0.0, 1.0]) + (1 - p.sum()) * self.f_x.moments[1]

    @property
    def var(self):
        return self.EX2 - self.mean**2

    @property
    def log_r(self):
        lp = safe_lae(self.log_p[..., 0], self.log_p[..., 1]).clip(-jnp.inf, -1e-8)
        return log1mexp(lp)

    @property
    def r(self):
        return jnp.exp(self.log_r)

    @property
    def M(self):
        "The number of mixture components"
        return self.f_x.M

    def plot(self):
        import matplotlib.pyplot as plt

        self.f_x.plot()
        plt.bar(0.0, self.p0, 0.05, alpha=0.2, color="tab:red")
        plt.bar(1.0, self.p1, 0.05, alpha=0.2, color="tab:red")


@jit
def _transition(f: SpikedBeta, s: jnp.ndarray, Ne: jnp.ndarray) -> SpikedBeta:
    """Given a prior distribution on population allele frequency, compute posterior after
    one round of WF mating."""

    @vmap
    def lp(fi, si):
        # compute update spike probabilities and wf transition
        # mu = fi.mean
        # sigma2 = fi.var
        # a0, b0 = _inv_beta(mu, sigma2)
        # a1, b1 = _wf_trans(si, Nei, a0, b0)
        # D = jnp.sqrt(
        #     jnp.sum((fi.f_x.a - a1) ** 2 + (fi.f_x.b - b1) ** 2))
        #
        # log_p = -D
        # log_p -= logsumexp(log_p)
        # log_c1 = fi.f_x.log_c + log_p
        # log_c1 -= logsumexp(log_p)
        # log_c1 = eqx.error_if(log_c1, jnp.isnan(log_c1), "log_c1 nan")

        # jax.debug.print("mu:{} sigma2:{} a0:{} a1:{} D:{} log_c1:{}", mu, sigma2, a0, a1, D, log_c1, ordered=True)
        # return fi._replace(
        #     f_x=fi.f_x._replace(log_c=log_c1)
        # )
        log_p, (a, b, log_c) = fi
        assert log_p.shape == (2,)
        log_r = fi.log_r
        x = jnp.array([0, Ne])[:, None]  # [2, 1]
        # probability mass for fixation. p(af=0) = p0 + (1-p0-p1) * \int_0^1 f(x) (1-x)^n, and similarly for p1
        log_p_c = log_r + logsumexp(
            log_c + betaln(x + a, Ne - x + b) - betaln(a, b), axis=1
        )  # [2] probability of loss or fixation in continuous component
        log_p1 = safe_lae(log_p, log_p_c)
        a1, b1 = _wf_trans(si, Ne, a, b)
        EX0 = jnp.isclose(b1, -1.0)
        EX1 = jnp.isclose(a1, -1.0)
        log_p2_0 = safe_lae(log_p1[0], safe_logsumexp(jnp.where(EX0, log_c, -jnp.inf)))
        log_p2_1 = safe_lae(log_p1[1], safe_logsumexp(jnp.where(EX1, log_c, -jnp.inf)))
        log_p2 = jnp.array([log_p2_0, log_p2_1])
        log_c = jnp.where(EX0 | EX1, -jnp.inf, log_c)
        log_c1 = fi.f_x.log_c
        return SpikedBeta.safe_init(log_p2, BetaMixture(a1, b1, log_c))

    assert f.log_p.ndim == 2
    assert f.log_p.shape[1] == 2
    ret = lp(f, s)
    return ret


def _binom_sampling(n, d, f: SpikedBeta):
    log_p, (a, b, log_c) = f
    assert log_p.shape == (2,)
    log_r = f.log_r
    a1 = a + d
    b1 = b + n - d
    # probability of the data given the absolutely continuous component
    log_p_data_comp_cont = jnp.where(
        jnp.isneginf(log_c),
        -jnp.inf,
        # beta binomial pmf
        log_c + betaln(a1, b1) - betaln(a, b) + _logbinom(n, d),
    )
    log_p_data_cont = safe_logsumexp(log_p_data_comp_cont)
    # probability of the data given the spike components
    log_p_data_spike = jnp.where((n > 0) & (d == jnp.array([0, n])), 0.0, -jnp.inf)
    # overall probability of the data
    ll = safe_lae(log_r + log_p_data_cont, safe_logsumexp(log_p + log_p_data_spike))
    # probability of spikes given data: p(spike | data) =
    # p(data | spike) p(spike) / p(data)
    log_p1 = log_p_data_spike + log_p - ll
    # probability of mixing components given data:
    # p(comp | data, cont) = p(data | comp, cont) * p(comp | cont) / p(data | cont)
    log_c1 = log_p_data_comp_cont - log_p_data_cont
    # posterior after binomial sampling --
    # some_c1 might be -inf for some components that have extremely low likelihood
    beta = SpikedBeta.safe_init(log_p1, BetaMixture(a1, b1, log_c1))
    return beta, ll


def _binom_sampling_admix(
    fs: SpikedBeta, datum: Dataset, i, key
) -> tuple[SpikedBeta, float]:
    """Compute filtering distributions after observing alleles.

    Params:
        fs: initial filtering distribution
        data: Dataset containing the observed data point
    """
    fs0 = fs
    M = fs.M
    K = fs.log_p.shape[0]
    n, d = datum.obs
    assert fs.log_p.ndim == 2
    fs1, llk = vmap(_binom_sampling, (None, None, 0))(n, d, fs)
    log_lam = llk + jnp.log(datum.theta)
    ll = safe_logsumexp(log_lam)
    # now compute posteriors

    def combine(f0, f1, log_theta, log_v):
        # posterior, f(p) \propto f_0(p) [theta*binom(n,d)*p^d(1-p)^n-d + v]
        # = theta*f1(p) + v*f0(p),  (v = \sum_{-i} theta_j ll_j)
        a1 = jnp.concatenate([f1.f_x.a, f0.f_x.a])
        b1 = jnp.concatenate([f1.f_x.b, f0.f_x.b])
        log_c1 = jnp.concatenate([log_theta + f1.f_x.log_c, log_v + f0.f_x.log_c])
        bm0 = BetaMixture(a1, b1, log_c1)

        # this is memory hungry
        @jax.remat
        def f(bm):
            return BetaMixture.interpolate(
                lambda x: bm0(x, log=True), 10 * M, norm=True, log_f=True
            ).top_k(M)

        bm1 = f(bm0)

        # now compute updated spike probabilities
        # now compute updated spike probabilities
        # log p(p=0|data) = log p(data|p=0) + log p(p=0) - log p(data)
        #                 = log[theta * 1{n=d=0} + v] + log p(p=0) - log p(data)
        zn = jnp.array([0, n])
        log_p1 = (
            vmap(safe_lae, (0, None))(
                jnp.where((n > 0) & (d == zn), log_theta, -jnp.inf), log_v
            )
            + f0.log_p
            - ll
        )
        return SpikedBeta.safe_init(log_p1, bm1)

    # log_V[i] = \sum_{-i} theta_j ll_j
    dp = partial(jnp.delete, log_lam, assume_unique_indices=True)
    log_Vj = vmap(dp)(jnp.arange(K))
    log_V = vmap(safe_logsumexp)(log_Vj)
    fs2 = vmap(combine)(fs0, fs1, jnp.log(datum.theta), log_V)
    return fs2, ll


def _tree_where(cond, a, b):
    return jax.tree.map(partial(jnp.where, cond), a, b)


def _forward_helper(accum, tup, Ne):
    beta0, ll0, last_t, key = accum
    datum, s_t, i = tup
    assert beta0.log_p.ndim == 2
    assert beta0.log_p.shape[1] == 2
    beta1 = _tree_where(datum.t != last_t, _transition(beta0, s_t, Ne), beta0)
    # beta1 = eqx.error_if(beta1, cs.any(), msg="spikes >= 1")
    # now process the observation
    n, d = datum.obs
    key, subkey = jax.random.split(key)
    beta2, ll1 = _tree_where(
        (datum.t == last_t) & (n > 0),
        _binom_sampling_admix(beta1, datum, i, subkey),
        (beta1, 0.0),
    )
    ll = ll0 + ll1
    accum = (beta2, ll, datum.t, key)
    return accum, beta2


# @partial(jit, static_argnums=3)
def forward(s: Selection, Ne: jnp.ndarray, data: Dataset, beta: BetaMixture):
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
    ninf = jnp.full([data.K, 2], -jnp.inf)
    beta0 = SpikedBeta(ninf, beta)
    s_t = s(data.t)
    init = (beta0, 0.0, data.t[0], jax.random.PRNGKey(1))

    if os.environ.get("BMWS_UNROLL_FORWARD"):
        logger.warning("Compiling unrolled forward loop!")
        # compile-free loop for debugging
        betas = []
        state = init
        for i in range(1, len(data.t)):
            state, beta = _forward_helper(
                state,
                jax.tree.map(lambda x: x[i], (data, s_t, Ne_t)) + (i,),
            )
            betas.append(beta)
            ll = state[1]
    else:
        (_, ll, _, _), betas = lax.scan(
            partial(_forward_helper, Ne=Ne),
            init,
            (data, s_t, jnp.arange(len(s_t))),
        )

    return betas, ll


def loglik(s, Ne, data: Dataset, beta0: BetaMixture):
    return forward(s=s, Ne=Ne, data=data, beta=beta0)[1]


def _construct_prior(prior: Union[int, BetaMixture]) -> BetaMixture:
    if isinstance(prior, int):
        M = prior
        prior = BetaMixture.uniform(M)
    return prior
