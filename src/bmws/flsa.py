# JAX implementation of:
# // Dynamic programming algorithm for the 1d fused lasso problem
# // (Ryan's implementation of Nick Johnson's algorithm)
from typing import Union

from jax import jit, lax
from jax import numpy as jnp


# from jax.experimental.host_callback import id_print
def id_print(x, *args, **kwargs):
    return x


@jit
def flsa(y: jnp.ndarray, lam: Union[float, jnp.ndarray]):
    y1 = jnp.atleast_1d(y)
    lam = jnp.atleast_1d(lam)
    n = len(y1)
    # if lam is scalar, broadcast to correct dims
    if lam.size == 1:
        lam = jnp.full_like(y1[:-1], lam)
    assert lam.ndim == y1.ndim == 1
    assert lam.size == y1.size - 1
    if n == 1:
        return y1
    return jnp.where(jnp.isclose(lam, 0.0).all(), y1, _prox_dp(y1, lam).squeeze())


# FIXME I adapted Johnson's algorithm to the case of unequal lambda's in the obvious way. however it has problems if
# some of the lambdas are zero. For our applications, we only ever have _all_ lambda=0 or they are all nonzero, so this
# is easy to catch. To fix this, need to be more careful about maintaining the piecewise linear representation---when
# some lambdas are zero, you get degenerate intervals.
def _prox_dp(y: jnp.ndarray, lam: jnp.ndarray):
    # // These are used to store the derivative of the
    # // piecewise quadratic function of interest
    # double afirst, alast, bfirst, blast;
    # double *x = (double*)malloc(2*n*sizeof(double));
    # double *a = (double*)malloc(2*n*sizeof(double));
    # double *b = (double*)malloc(2*n*sizeof(double));
    # int l,r;
    # // These are the knots of the back-pointers
    # double *tm = (double*)malloc((n-1)*sizeof(double));
    # double *tp = (double*)malloc((n-1)*sizeof(double));
    n = len(y)
    x, a, b = jnp.zeros([3, 2 * n])

    # // We step through the first iteration manually
    ell = n - 1
    r = n
    tm0 = -lam[0] + y[0]
    tp0 = lam[0] + y[0]
    x = x.at[ell].set(tm0)
    x = x.at[r].set(tp0)
    a = a.at[ell].set(1)
    b = b.at[ell].set(-y[0] + lam[0])
    a = a.at[r].set(-1)
    b = b.at[r].set(y[0] + lam[0])
    afirst = 1  # these never seem to change
    bfirst = -lam[0] - y[1]
    alast = -1
    blast = -lam[0] + y[1]

    # // Now iterations 2 through n-1
    # int lo, hi;
    # double alo, blo, ahi, bhi;
    # for (int k=1; k<n-1; k++) {
    def body1(carry, ylam):
        # // Compute lo: step up from l until the
        # // derivative is greater than -lam
        # alo = afirst;
        # blo = bfirst;
        y, lam = ylam
        afirst, bfirst, alast, blast, ell, r, a, b, x = carry
        afirst, bfirst, alast, blast = id_print(
            (afirst, bfirst, alast, blast), what="af/bf/al/bl"
        )

        # for (lo=l; lo<=r; lo++) {
        #   if (alo*x[lo]+blo > -lam) break;
        #   alo += a[lo];
        #   blo += b[lo];
        # }
        def body2(d):
            d["alo"] += a[d["lo"]]
            d["blo"] += b[d["lo"]]
            d["lo"] += 1
            return d

        d = lax.while_loop(
            lambda d: (d["alo"] * x[d["lo"]] + d["blo"] <= -lam) & (d["lo"] <= r),
            body2,
            {"alo": afirst, "blo": bfirst, "lo": ell},
        )
        alo = d["alo"]
        blo = d["blo"]
        lo = d["lo"]

        # // Compute the negative knot
        tm = (-lam - blo) / alo
        ell = lo - 1
        x = x.at[ell].set(tm)

        # // Compute hi: step down from r until the
        # // derivative is less than lam
        # ahi = alast;
        # bhi = blast;
        # for (hi=r; hi>=l; hi--) {
        #   if (-ahi*x[hi]-bhi < lam) break;
        #   ahi += a[hi];
        #   bhi += b[hi];
        # }
        def body3(d):
            d["ahi"] += a[d["hi"]]
            d["bhi"] += b[d["hi"]]
            d["hi"] -= 1
            return d

        # // Compute the positive knot
        ahi = alast
        bhi = blast
        ahi, bhi, r, ell = id_print((ahi, bhi, r, ell), what="ahi/bhi")
        d = lax.while_loop(
            lambda d: (-d["ahi"] * x[d["hi"]] - d["bhi"] >= lam) & (d["hi"] >= ell),
            body3,
            {"ahi": ahi, "bhi": bhi, "hi": r},
        )
        ahi = d["ahi"]
        bhi = d["bhi"]
        hi = d["hi"]
        ahi, bhi, r, ell = id_print((ahi, bhi, r, ell), what="ahi/bhi")

        tp = (lam + bhi) / (-ahi)
        r = hi + 1
        x = x.at[r].set(tp)

        # // Update a and b
        a = a.at[ell].set(alo)
        b = b.at[ell].set(blo + lam)
        a = a.at[r].set(ahi)
        b = b.at[r].set(bhi + lam)

        afirst = 1
        bfirst = -lam - y
        alast = -1
        blast = -lam + y
        return (afirst, bfirst, alast, blast, ell, r, a, b, x), (tm, tp)

    init = (afirst, bfirst, alast, blast, ell, r, a, b, x)
    (afirst, bfirst, alast, blast, ell, r, a, b, x), (tm, tp) = lax.scan(
        body1,
        init,
        (y[2:], lam[1:]),
    )
    tm = jnp.insert(tm, 0, tm0)
    tp = jnp.insert(tp, 0, tp0)
    afirst, alast, bfirst, blast, tm, tp = id_print(
        (afirst, alast, bfirst, blast, tm, tp), what="aabb"
    )

    # // Compute the last coefficient: this is where
    # // the function has zero derivative
    # alo = afirst;
    # blo = bfirst;
    # for (lo=l; lo<=r; lo++) {
    #   if (alo*x[lo]+blo > 0) break;
    #   alo += a[lo];
    #   blo += b[lo];
    # }
    def body2(d):
        d["alo"] += a[d["lo"]]
        d["blo"] += b[d["lo"]]
        d["lo"] += 1
        return d

    d = lax.while_loop(
        lambda d: (d["alo"] * x[d["lo"]] + d["blo"] <= 0) & (d["lo"] <= r),
        body2,
        {"alo": afirst, "blo": bfirst, "lo": ell},
    )
    alo = d["alo"]
    blo = d["blo"]
    alo, blo = id_print((alo, blo), what="alo blo")
    betan = -blo / alo

    # // Compute the rest of the coefficients, by the
    # // back-pointers
    # for (int k=n-2; k>=0; k--) {
    #   if (beta[k+1]>tp[k]) beta[k] = tp[k];
    #   else if (beta[k+1]<tm[k]) beta[k] = tm[k];
    #   else beta[k] = beta[k+1];
    # }
    def body3(beta_k1, tup):
        tpk, tmk = tup
        return (jnp.clip(beta_k1, tmk, tpk),) * 2

    tp, tm = id_print((tp, tm), what="tptm")
    _, beta = lax.scan(body3, betan, (tp, tm), reverse=True)
    return jnp.append(beta, betan)[:n]
