"""Two-stage least squares with absorbed fixed effects.

Demand estimation needs this because price is not handed to the econometrician
as an experiment. An airline that knows a route is unusually desirable this
quarter - a convention in town, a competitor's strike - charges more on it and
also sells more on it. The unobserved desirability sits in the error term and is
correlated with the price, so ordinary least squares reads the two moving
together and concludes demand is less price-sensitive than it is. In airline
data the bias is large enough to flip the sign of the answer to the question the
merger simulation actually asks.

The fix is the usual one: find variation in price that comes from cost or from
market structure rather than from demand. What counts as a valid instrument is
argued in ``06_estimate_demand.py``; this module only does the algebra, and
reports the diagnostics that say whether the instruments were strong enough to
believe the second stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .fe_ols import demean, factorize, feols


@dataclass
class IVResult:
    params: np.ndarray
    vcov: np.ndarray
    names: list[str]
    nobs: int
    nclusters: int
    df_resid: int
    resid: np.ndarray = field(repr=False)
    first_stage_F: dict[str, float] = field(default_factory=dict)
    hansen_j: float = float("nan")
    hansen_p: float = float("nan")
    n_instruments: int = 0
    n_endog: int = 0
    n_overid: int = 0

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0.0, None))

    @property
    def tstat(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.se

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        crit = stats.t.ppf(0.5 + level / 2.0, df=max(self.df_resid, 1))
        return np.column_stack([self.params - crit * self.se,
                                self.params + crit * self.se])

    def get(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return float(self.params[i]), float(self.se[i])

    def summary(self) -> str:
        ci = self.conf_int()
        out = ["  {:<26} {:>10} {:>10} {:>8} {:>22}".format(
            "", "coef", "se", "t", "95% interval")]
        for i, nm in enumerate(self.names):
            out.append("  {:<26} {:>10.4f} {:>10.4f} {:>8.2f} {:>22}".format(
                nm[:26], self.params[i], self.se[i], self.tstat[i],
                f"[{ci[i,0]:.4f}, {ci[i,1]:.4f}]"))
        out.append(f"  N={self.nobs:,}  clusters={self.nclusters:,}")
        for nm, f in self.first_stage_F.items():
            out.append(f"  first-stage F, {nm:<18} {f:>10.1f}")
        if np.isfinite(self.hansen_j):
            out.append(f"  Hansen J ({self.n_overid} overid) "
                       f"{self.hansen_j:.2f}, p = {self.hansen_p:.3f}")
        return "\n".join(out)


def iv2sls(y: np.ndarray, X_exog: np.ndarray | None, X_endog: np.ndarray,
           Z_excl: np.ndarray, fes=None, cluster=None, weights=None,
           names_exog: list[str] | None = None,
           names_endog: list[str] | None = None,
           names_z: list[str] | None = None) -> IVResult:
    """Estimate y = X_exog b1 + X_endog b2 + e, instrumenting X_endog with Z_excl."""
    y = np.asarray(y, float).ravel()
    n = y.size
    X_endog = np.atleast_2d(np.asarray(X_endog, float))
    if X_endog.shape[0] != n:
        X_endog = X_endog.T
    Z_excl = np.atleast_2d(np.asarray(Z_excl, float))
    if Z_excl.shape[0] != n:
        Z_excl = Z_excl.T
    if X_exog is None or (hasattr(X_exog, "size") and np.asarray(X_exog).size == 0):
        X_exog = np.empty((n, 0))
    else:
        X_exog = np.atleast_2d(np.asarray(X_exog, float))
        if X_exog.shape[0] != n:
            X_exog = X_exog.T

    names_exog = list(names_exog or [f"x{i}" for i in range(X_exog.shape[1])])
    names_endog = list(names_endog or [f"e{i}" for i in range(X_endog.shape[1])])
    names_z = list(names_z or [f"z{i}" for i in range(Z_excl.shape[1])])
    w = np.ones(n) if weights is None else np.asarray(weights, float).ravel()
    fes = fes or []

    # Absorb the fixed effects out of everything at once, then work with the
    # residualised data. Frisch-Waugh applies to 2SLS the same way it does to OLS.
    big = np.column_stack([y, X_exog, X_endog, Z_excl])
    big, _ = demean(big, fes, w)
    k_abs = sum(g for _, g in fes) - (len(fes) - 1 if fes else 0)
    i0 = 1
    i1 = i0 + X_exog.shape[1]
    i2 = i1 + X_endog.shape[1]
    yt = big[:, 0]
    Xx, Xn, Zx = big[:, i0:i1], big[:, i1:i2], big[:, i2:]

    X = np.column_stack([Xx, Xn])
    Z = np.column_stack([Xx, Zx])
    names = names_exog + names_endog
    if Z.shape[1] < X.shape[1]:
        raise ValueError(f"under-identified: {Z.shape[1]} instruments for "
                         f"{X.shape[1]} regressors")

    sw = w[:, None]
    ZtWZ = Z.T @ (Z * sw)
    ZtWX = Z.T @ (X * sw)
    ZtWy = Z.T @ (yt * w)
    Pinv = np.linalg.pinv(ZtWZ)
    XPX = ZtWX.T @ Pinv @ ZtWX
    XPy = ZtWX.T @ Pinv @ ZtWy
    bread = np.linalg.pinv(XPX)
    beta = bread @ XPy
    resid = yt - X @ beta

    Xhat = Z @ (Pinv @ ZtWX)          # fitted regressors, for the sandwich meat
    score = Xhat * (w * resid)[:, None]
    k = X.shape[1]
    if cluster is None:
        meat = score.T @ score
        df = max(n - k - k_abs, 1)
        vcov = bread @ meat @ bread * (n / df)
        ng = n
    else:
        codes, ng = factorize(np.asarray(cluster).ravel())
        order = np.argsort(codes, kind="stable")
        sc, cc = score[order], codes[order]
        meat = np.zeros((k, k))
        for chunk in np.split(sc, np.flatnonzero(np.diff(cc)) + 1):
            s = chunk.sum(axis=0)
            meat += np.outer(s, s)
        df = max(n - k - k_abs, 1)
        vcov = bread @ meat @ bread * ((ng / max(ng - 1, 1)) * ((n - 1) / df))

    # First stage, one regression per endogenous regressor. The reported F is
    # the cluster-robust joint test that the excluded instruments do nothing;
    # under ten is the conventional signal that the second stage is unreliable.
    fs: dict[str, float] = {}
    for j, nm in enumerate(names_endog):
        r = feols(Xn[:, j], np.column_stack([Xx, Zx]), fes=None, cluster=cluster,
                  weights=w, names=names_exog + names_z)
        zz = [z for z in names_z if z in r.names]
        if zz:
            fs[nm] = r.wald(zz)[0]

    # Hansen J for overidentification, in its cluster-robust form.
    hj, hp = float("nan"), float("nan")
    n_over = Z.shape[1] - X.shape[1]
    if n_over > 0:
        g = Z * (w * resid)[:, None]
        if cluster is None:
            S = g.T @ g
        else:
            codes, _ = factorize(np.asarray(cluster).ravel())
            order = np.argsort(codes, kind="stable")
            gs, cc = g[order], codes[order]
            S = np.zeros((Z.shape[1], Z.shape[1]))
            for chunk in np.split(gs, np.flatnonzero(np.diff(cc)) + 1):
                s = chunk.sum(axis=0)
                S += np.outer(s, s)
        gbar = g.sum(axis=0)
        hj = float(gbar @ np.linalg.pinv(S) @ gbar)
        hp = float(stats.chi2.sf(hj, n_over))

    return IVResult(params=beta, vcov=vcov, names=names, nobs=n, nclusters=ng,
                    df_resid=max(ng - 1, 1) if cluster is not None else df,
                    resid=resid, first_stage_F=fs, hansen_j=hj, hansen_p=hp,
                    n_instruments=Z_excl.shape[1], n_endog=X_endog.shape[1],
                    n_overid=n_over)
