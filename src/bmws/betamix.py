"beta mixture with spikes model"
from functools import partial
from typing import NamedTuple, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.special import betaln, gammaln, logsumexp, xlog1py, xlogy
from loguru import logger

from bmws.data import Dataset


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

    def top_k(self, k):
        """Return the top k components of the mixture"""
        i = self.log_c.argsort()[-k:]
        log_c = self.log_c[i]
        log_c -= logsumexp(log_c)
        return BetaMixture(self.a[i], self.b[i], log_c)

    @property
    def c(self):
        return jnp.exp(self.log_c)

    @classmethod
    def uniform(cls, M) -> "BetaMixture":
        return cls.interpolate(lambda x: 0.0, M, norm=True, log_f=True)

    @classmethod
    def interpolate(
        cls, f, M, interval=(0.0, 1.0), norm=False, log_f=False
    ) -> "BetaMixture":
        # bernstein polynomial basis:
        # sum_{i=0}^(M) f(i/M) binom(M,i) x^i (1-x)^(M-i)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) x^(i-1) (1-x)^(M-i+2-1)
        #   = sum_{i=1}^(M+1) f((i-1)/M) binom(M,i-1) * beta(i,M-i+2) x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   = sum_{i=1}^(M+1) f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   we assume that f(0) = f(1) = 0 hence
        #   = sum_{i=2}^M f((i-1)/M) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        # slice up interval into M chunks
        u, v = interval
        # slice up the interval into M slices
        delta = (v - u) / M
        Q = jnp.ceil(1 / delta)
        # M + 2 because we assume that f(0)=f(1)=0, so those first and last slices don't count
        Q = jnp.maximum(Q, M + 2)
        # assume f is zero outside of (u, v), so find the index i0 such that (u, v) \in [i0 / Q, (i0 + 1) / Q, ..., (i0 + M - 1) / Q]
        i0 = jnp.floor(u * Q).clip(2)
        i = i0 + jnp.arange(M)
        if log_f:
            log_c = vmap(f)((i - 1) / Q) - jnp.log1p(Q)
        else:
            c = vmap(f)((i - 1) / Q) / (1 + Q)
            c0 = jnp.isclose(c, 0.0)  # work around nans in gradients
            c_safe = jnp.where(c0, 1.0, c)
            log_c = jnp.where(c0, -jnp.inf, jnp.log(c_safe))
        if norm:
            log_c -= logsumexp(log_c)
        return cls(a=i, b=Q - i + 2, log_c=log_c)

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
        return self.moments[0].dot(self.c)

    @property
    def var(self):
        return self.moments[1].dot(self.c) - self.mean**2

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
        keys = jax.random.split(key, 3)
        i = jax.random.categorical(keys[0], self.f_x.log_c)
        f = jax.random.beta(keys[1], self.f_x.a[i], self.f_x.b[i])
        a = jnp.array([0.0, 1.0, f])
        p_spike = jnp.exp(self.log_p)
        p_spike *= jnp.array(spikes)
        p = jnp.append(p_spike, 1 - p_spike.sum())
        p /= p.sum()
        return jax.random.choice(keys[2], a, p=p)

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

    # def EX2(self):
    #     p = self.p
    #     return p @ jnp.array([0.0, 1.0]) + (1 - p.sum()) * self.f_x.moments[1]

    # def var(self):
    #     return self.EX2() - self.mean()**2

    @property
    def log_r(self):
        lp = safe_lae(self.log_p[0], self.log_p[1]).clip(-jnp.inf, -1e-8)
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

    @partial(vmap, in_axes=(0, 0, 0))
    def lp(fi, si, Nei):
        # compute update spike probabilities and wf transition
        log_p, (a, b, log_c) = fi
        assert log_p.shape == (2,)
        log_r = fi.log_r
        x = jnp.array([0, Nei])[:, None]  # [2, 1]
        # probability mass for fixation. p(af=0) = p0 + (1-p0-p1) * \int_0^1 f(x) (1-x)^n, and similarly for p1
        log_p_c = log_r + logsumexp(
            log_c + betaln(x + a, Nei - x + b) - betaln(a, b), axis=1
        )  # [2] probability of loss or fixation in continuous component
        log_p1 = safe_lae(log_p, log_p_c)
        a1, b1 = _wf_trans(si, Nei, a, b)
        EX0 = jnp.isclose(b1, -1.0)
        EX1 = jnp.isclose(a1, -1.0)
        log_p2_0 = safe_lae(log_p1[0], safe_logsumexp(jnp.where(EX0, log_c, -jnp.inf)))
        log_p2_1 = safe_lae(log_p1[1], safe_logsumexp(jnp.where(EX1, log_c, -jnp.inf)))
        log_p2 = jnp.array([log_p2_0, log_p2_1])
        log_c = jnp.where(EX0 | EX1, -jnp.inf, log_c)
        # jax.debug.print("si:{} wfa:{} wfb:{} wfa1:{} wfb1:{}", si, a, b, a1, b1, ordered=True)
        return SpikedBeta.safe_init(log_p2, BetaMixture(a1, b1, log_c))

    assert f.log_p.ndim == 2
    assert f.log_p.shape[1] == 2
    ret = lp(f, s, Ne)
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
    log_c1 = eqx.error_if(log_c1, jnp.isnan(log_c1).any(), msg="nan in log_c1")
    beta = SpikedBeta.safe_init(log_p1, BetaMixture(a1, b1, log_c1))
    return beta, ll


@jit
def _binom_sampling_admix(fs: SpikedBeta, datum: Dataset) -> Tuple[SpikedBeta, float]:
    """Compute filtering distributions after observing alleles.

    Params:
        fs: initial filtering distribution
        data: Dataset containing the observed data point
    """
    fs0 = fs
    M = fs.M
    n, d = datum.obs
    assert fs.log_p.ndim == 2
    fs1, llk = vmap(_binom_sampling, in_axes=(None, None, 0))(n, d, fs)
    ll = logsumexp(jnp.log(datum.theta) + llk)

    # in each population k the posterior is
    # (1-theta[k]) * [original mixture] + theta[k] * [new mixture]
    # combine these into one and retain the M mixture components with the highest weights
    def _combine(theta: float, beta0: SpikedBeta, beta1: SpikedBeta):
        lt = jnp.array([jnp.log1p(-theta), jnp.log(theta)])
        log_c1 = jnp.concatenate(
            [lt[0] + beta0.f_x.log_c, lt[1] + beta1.f_x.log_c]
        )  # [2 * M]
        # since n=1, d is either 0 or 1
        # if d = 0 then a1 = [a, a] and b1 = [b, b+1] with
        # log_c1 = [log_c0, log_c1] + log_p_mix
        a1 = jnp.concatenate([beta0.f_x.a, beta1.f_x.a])
        b1 = jnp.concatenate([beta0.f_x.b, beta1.f_x.b])
        # approximate the mixture on 2*M components
        bm0 = BetaMixture(a1, b1, log_c1)
        if True:
            bm = bm0.top_k(M)
        else:
            mean = bm0.mean
            var = bm0.var
            sd = jnp.sqrt(var)
            u = jnp.maximum(0.0, mean - 3 * sd)
            v = jnp.minimum(1.0, mean + 3 * sd)
            bm = BetaMixture.interpolate(bm0, M, norm=True, interval=(u, v))
        # now update the spike probabilities
        log_p1 = safe_lae(lt[0] + beta0.log_p, lt[1] + beta1.log_p)
        ret = SpikedBeta.safe_init(log_p1, bm)
        return ret

    fs2 = vmap(_combine)(datum.theta, fs0, fs1)
    # jax.debug.print("fs0:{} fs1:{} fs2:{}", fs0, fs1, fs2, ordered=True)
    return fs2, ll


def _tree_where(cond, a, b):
    return jax.tree.map(partial(jnp.where, cond), a, b)


def check_spikes(beta: SpikedBeta):
    return safe_lae(beta.log_p[0], beta.log_p[1]) >= 0


def _fix_spikes(beta: SpikedBeta):
    # ensure that the larger of the two spikes is strictly less that 1 - the smaller of the two
    m = jnp.minimum(log1mexp(beta.log_p.min()), -1e-8)
    log_p = jnp.clip(beta.log_p, -jnp.inf, m)
    return beta._replace(log_p=log_p)


def _forward_helper(accum, tup):
    beta0, ll0, last_t = accum
    datum, s_t, Ne_t, i = tup
    # jax.debug.print("beta0:{} datum:{}", beta0, datum, ordered=True)
    # jax.debug.print('log_p:{} b.mean:{} datum:{}', beta0.log_p, vmap(lambda p: p.f_x.mean)(beta0), datum, ordered=True)
    assert beta0.log_p.ndim == 2
    assert beta0.log_p.shape[1] == 2
    beta1 = _tree_where(datum.t != last_t, _transition(beta0, s_t, Ne_t), beta0)
    # jax.debug.print("beta1:{}", beta1, ordered=True)
    # beta1 = eqx.error_if(beta1, cs.any(), msg="spikes >= 1")
    # now process the observation
    n, d = datum.obs
    beta2, ll1 = _tree_where(n > 0, _binom_sampling_admix(beta1, datum), (beta1, 0.0))
    ll = ll0 + ll1
    # jax.debug.print("beta2:{}", beta2, ordered=True)
    accum = (beta2, ll, datum.t)
    nan = jax.tree.reduce(
        lambda x, y: x | y, jax.tree_map(lambda a: jnp.isnan(a).any(), accum)
    )
    accum = eqx.error_if(accum, nan, msg="nan in accum")
    return accum, beta2


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
    assert Ne.ndim == 2
    T, K = Ne.shape
    assert s.shape == Ne.shape == (T, K)

    ninf = jnp.full([data.K, 2], -jnp.inf)
    beta0 = SpikedBeta(ninf, beta)
    s_t = s[data.t]
    Ne_t = Ne[data.t]

    if False:
        logger.warning("Compiling unrolled forward loop!")
        # compile-free loop for debugging
        betas = []
        f_x = beta0
        ll = 0.0
        t = data.t[0]
        for i in range(1, len(data.t)):
            (f_x, ll, t), _ = _forward_helper(
                (f_x, ll, t),
                jax.tree.map(lambda x: x[i], (data, s_t, Ne_t)),
            )
            betas.append(f_x)
    else:
        (_, ll, _), betas = lax.scan(
            _forward_helper,
            (beta0, 0.0, data.t[0]),
            (data, s_t, Ne_t, jnp.arange(len(Ne_t))),
        )

    return betas, ll


def loglik(s, Ne, data: Dataset, beta0: BetaMixture):
    return forward(s=s, Ne=Ne, data=data, beta=beta0)[1]


def _construct_prior(prior: Union[int, BetaMixture]) -> BetaMixture:
    if isinstance(prior, int):
        M = prior
        prior = BetaMixture.uniform(M)
    return prior
