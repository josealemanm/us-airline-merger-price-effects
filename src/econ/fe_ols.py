"""Weighted least squares with absorbed fixed effects and clustered inference.

The panel here is roughly forty thousand routes by sixty quarters. Estimating
route and quarter effects by building dummy columns would mean a design matrix
with forty thousand columns, which does not fit in memory and does not need to.
The fixed effects are absorbed instead, by the method of alternating
projections: subtract group means for one set of effects, then the other, and
repeat until nothing moves. Frisch-Waugh-Lovell guarantees the slope
coefficients that come out are the ones the dummy regression would have given.

Standard errors are clustered, by default on the same unit that carries the
first fixed effect, because fares on a route are serially correlated and
treating sixty quarters of one route as sixty independent draws would shrink
the intervals to nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# --------------------------------------------------------------------- codes
def factorize(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Map arbitrary labels to 0..G-1 integer codes. Returns codes and G."""
    uniq, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes.astype(np.int64), int(uniq.size)


# ------------------------------------------------------------------ absorbing
def _demean_once(M: np.ndarray, codes: np.ndarray, ngroups: int,
                 w: np.ndarray) -> np.ndarray:
    """Subtract the weighted group mean of every column, in place."""
    wsum = np.bincount(codes, weights=w, minlength=ngroups)
    # A group with zero total weight contributes nothing; guard the divide.
    wsum[wsum == 0] = 1.0
    for j in range(M.shape[1]):
        num = np.bincount(codes, weights=w * M[:, j], minlength=ngroups)
        M[:, j] -= (num / wsum)[codes]
    return M


def demean(M: np.ndarray, fes: list[tuple[np.ndarray, int]],
           w: np.ndarray | None = None, tol: float = 1e-10,
           maxiter: int = 5000) -> tuple[np.ndarray, int]:
    """Absorb every fixed effect in ``fes`` out of the columns of ``M``.

    With one fixed effect a single pass is exact. With two or more, the passes
    interfere and have to be iterated; convergence is guaranteed but the rate
    depends on how tangled the two groupings are. Returns the residualised
    matrix and the number of iterations used.
    """
    M = np.array(M, dtype=np.float64, copy=True)
    if not fes:
        return M, 0
    if w is None:
        w = np.ones(M.shape[0])
    if len(fes) == 1:
        codes, g = fes[0]
        return _demean_once(M, codes, g, w), 1

    scale = np.maximum(np.abs(M).max(axis=0), 1e-12)
    for it in range(1, maxiter + 1):
        prev = M.copy()
        for codes, g in fes:
            M = _demean_once(M, codes, g, w)
        if np.max(np.abs(M - prev) / scale) < tol:
            return M, it
    raise RuntimeError(f"fixed-effect absorption did not converge in {maxiter} passes")


# -------------------------------------------------------------------- results
@dataclass
class FEResult:
    params: np.ndarray
    vcov: np.ndarray
    names: list[str]
    nobs: int
    nclusters: int
    df_resid: int
    r2_within: float
    resid: np.ndarray = field(repr=False)
    fitted_within: np.ndarray = field(repr=False)
    iterations: int = 0

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0.0, None))

    @property
    def tstat(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.se

    @property
    def pvalue(self) -> np.ndarray:
        return 2.0 * stats.t.sf(np.abs(self.tstat), df=max(self.df_resid, 1))

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        crit = stats.t.ppf(0.5 + level / 2.0, df=max(self.df_resid, 1))
        return np.column_stack([self.params - crit * self.se,
                                self.params + crit * self.se])

    def get(self, name: str) -> tuple[float, float]:
        """Coefficient and standard error for one regressor, by name."""
        i = self.names.index(name)
        return float(self.params[i]), float(self.se[i])

    def wald(self, names: list[str]) -> tuple[float, float]:
        """Joint test that the named coefficients are all zero. F and p."""
        idx = [self.names.index(n) for n in names]
        R = np.zeros((len(idx), len(self.params)))
        for r, i in enumerate(idx):
            R[r, i] = 1.0
        Rb = R @ self.params
        mid = R @ self.vcov @ R.T
        f = float(Rb @ np.linalg.solve(mid, Rb) / len(idx))
        return f, float(stats.f.sf(f, len(idx), max(self.df_resid, 1)))

    def summary(self, keep: list[str] | None = None) -> str:
        rows = ["  {:<28} {:>10} {:>10} {:>8} {:>20}".format(
            "", "coef", "se", "t", "95% interval")]
        ci = self.conf_int()
        for i, nm in enumerate(self.names):
            if keep is not None and nm not in keep:
                continue
            rows.append("  {:<28} {:>10.4f} {:>10.4f} {:>8.2f} {:>20}".format(
                nm[:28], self.params[i], self.se[i], self.tstat[i],
                f"[{ci[i,0]:.4f}, {ci[i,1]:.4f}]"))
        rows.append(f"  N={self.nobs:,}  clusters={self.nclusters:,}  "
                    f"within-R2={self.r2_within:.4f}")
        return "\n".join(rows)


# ------------------------------------------------------------------ estimator
def feols(y: np.ndarray, X: np.ndarray, fes: list[tuple[np.ndarray, int]] | None = None,
          cluster: np.ndarray | None = None, weights: np.ndarray | None = None,
          names: list[str] | None = None, drop_collinear: bool = True) -> FEResult:
    """Weighted OLS of y on X, absorbing ``fes``, clustering on ``cluster``.

    ``fes`` and ``cluster`` are integer-coded (see ``factorize``). ``X`` should
    not contain a constant: the fixed effects absorb it.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = y.size
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]
    names = list(names)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    fes = fes or []

    M, iters = demean(np.column_stack([y, X]), fes, w)
    yt, Xt = M[:, 0], M[:, 1:]

    # Absorbed parameters: the group counts, less one redundancy for each
    # additional fixed effect beyond the first (the grand mean is shared).
    k_absorbed = sum(g for _, g in fes) - (len(fes) - 1 if fes else 0)

    # Collinear columns appear when a regressor is a linear combination of the
    # absorbed effects, which happens routinely with event-time dummies.
    keep = np.arange(Xt.shape[1])
    if drop_collinear and Xt.shape[1] > 0:
        sw = np.sqrt(w)[:, None]
        q, r, piv = _qr_pivot(Xt * sw)
        tol = max(Xt.shape) * np.finfo(float).eps * max(abs(np.diag(r)).max(), 1e-300)
        rank = int(np.sum(np.abs(np.diag(r)) > tol))
        keep = np.sort(piv[:rank])
    Xk = Xt[:, keep]
    kept_names = [names[i] for i in keep]

    XtW = Xk * w[:, None]
    XtWX = Xk.T @ XtW
    bread = np.linalg.pinv(XtWX)
    beta = bread @ (XtW.T @ yt)
    resid = yt - Xk @ beta

    k = Xk.shape[1]
    if cluster is None:
        # Heteroskedasticity-robust, the HC1 form.
        meat = (Xk * (w * resid)[:, None]).T @ (Xk * (w * resid)[:, None])
        df = max(n - k - k_absorbed, 1)
        vcov = bread @ meat @ bread * (n / df)
        ng = n
    else:
        cl = np.asarray(cluster).ravel()
        codes, ng = factorize(cl)
        score = Xk * (w * resid)[:, None]
        meat = np.zeros((k, k))
        order = np.argsort(codes, kind="stable")
        sc, cc = score[order], codes[order]
        bounds = np.flatnonzero(np.diff(cc)) + 1
        for chunk in np.split(sc, bounds):
            s = chunk.sum(axis=0)
            meat += np.outer(s, s)
        df = max(n - k - k_absorbed, 1)
        corr = (ng / max(ng - 1, 1)) * ((n - 1) / df)
        vcov = bread @ meat @ bread * corr

    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (yt - np.average(yt, weights=w)) ** 2))
    r2w = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FEResult(
        params=beta, vcov=vcov, names=kept_names, nobs=n,
        nclusters=ng, df_resid=max(ng - 1, 1) if cluster is not None else df,
        r2_within=r2w, resid=resid, fitted_within=Xk @ beta, iterations=iters,
    )


def _qr_pivot(A: np.ndarray):
    """Column-pivoted QR, used only to find and drop collinear regressors."""
    from scipy.linalg import qr
    q, r, p = qr(A, mode="economic", pivoting=True)
    return q, r, p
