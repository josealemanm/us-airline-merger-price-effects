"""Inference that does not lean on the asymptotics being kind.

The stacked design leaves a modest number of treated routes per merger, and
Alaska / Virgin America leaves very few. Cluster-robust standard errors are
consistent as the number of clusters grows, and with two hundred treated routes
that is a promise about a limit nobody is standing near. Two procedures here
check whether the headline result survives without that promise.

The wild cluster bootstrap resamples the residuals at the cluster level under
the null, which is the procedure Cameron, Gelbach and Miller show keeps its size
with few clusters where the plain t-test does not.

Randomization inference asks a different question and is the more honest one for
this design: if the merger had touched a different set of routes chosen at
random, how often would the estimate have been as large as the one observed? It
tests sharp nulls rather than average effects, and it needs no distributional
assumption at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fe_ols import FEResult, factorize, feols


@dataclass
class BootstrapResult:
    point: float
    se: float
    ci_lo: float
    ci_hi: float
    p_value: float
    draws: np.ndarray
    n_draws: int

    def summary(self) -> str:
        return (f"  estimate {self.point:+.4f}   bootstrap se {self.se:.4f}   "
                f"95% [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}]   p = {self.p_value:.4f}")


def wild_cluster_bootstrap(y, X, fes, cluster, weights=None, names=None,
                           test_index: int = 0, n_boot: int = 999,
                           seed: int = 0) -> BootstrapResult:
    """Wild cluster bootstrap-t for one coefficient, imposing the null.

    Residuals are drawn once per cluster with Rademacher signs, so the whole
    within-cluster correlation structure is preserved. The reference
    distribution is of the t-statistic, not the coefficient, which is what gives
    the procedure its refinement.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float).ravel()
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != y.size:
        X = X.T
    names = list(names or [f"x{i}" for i in range(X.shape[1])])

    full = feols(y, X, fes=fes, cluster=cluster, weights=weights, names=names)
    b_hat = full.params[test_index]
    t_hat = b_hat / full.se[test_index]

    # Restricted fit: drop the tested regressor, so the null is imposed.
    keep = [i for i in range(X.shape[1]) if i != test_index]
    if keep:
        r0 = feols(y, X[:, keep], fes=fes, cluster=cluster, weights=weights,
                   names=[names[i] for i in keep])
        resid0 = r0.resid
        fitted0 = y - resid0
    else:
        from .fe_ols import demean
        M, _ = demean(y[:, None], fes or [],
                      np.ones(y.size) if weights is None else np.asarray(weights, float))
        resid0 = M[:, 0]
        fitted0 = y - resid0

    codes, ng = factorize(np.asarray(cluster).ravel())
    ts = np.empty(n_boot)
    for b in range(n_boot):
        signs = rng.choice((-1.0, 1.0), size=ng)[codes]
        yb = fitted0 + resid0 * signs
        rb = feols(yb, X, fes=fes, cluster=cluster, weights=weights, names=names)
        ts[b] = rb.params[test_index] / rb.se[test_index]

    p = float((np.sum(np.abs(ts) >= abs(t_hat)) + 1) / (n_boot + 1))
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    se = full.se[test_index]
    return BootstrapResult(
        point=float(b_hat), se=float(se),
        ci_lo=float(b_hat - hi_t * se), ci_hi=float(b_hat - lo_t * se),
        p_value=p, draws=ts, n_draws=n_boot,
    )


@dataclass
class RandomizationResult:
    point: float
    p_value: float
    null_draws: np.ndarray
    quantile: float

    def summary(self) -> str:
        return (f"  estimate {self.point:+.4f}   placebo p = {self.p_value:.4f}   "
                f"({self.quantile:.1%} of placebo draws are smaller in absolute value)")


def randomization_test(y, X_builder, treated_units: np.ndarray,
                       all_units: np.ndarray, n_perm: int = 500,
                       seed: int = 0) -> RandomizationResult:
    """Re-assign treatment at random over units and re-estimate each time.

    ``X_builder(treated_set) -> (beta,)`` must rebuild the treatment variable
    for an arbitrary set of treated units and return the coefficient. The number
    of treated units is held fixed at the observed count, so each placebo draw
    is a design the data could plausibly have produced.
    """
    rng = np.random.default_rng(seed)
    units = np.unique(all_units)
    k = np.unique(treated_units).size
    obs = X_builder(set(np.unique(treated_units).tolist()))
    draws = np.empty(n_perm)
    for i in range(n_perm):
        pick = set(rng.choice(units, size=k, replace=False).tolist())
        draws[i] = X_builder(pick)
    finite = draws[np.isfinite(draws)]
    p = float((np.sum(np.abs(finite) >= abs(obs)) + 1) / (finite.size + 1))
    q = float(np.mean(np.abs(finite) < abs(obs)))
    return RandomizationResult(point=float(obs), p_value=p,
                               null_draws=finite, quantile=q)


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                         stat, n_boot: int = 2000, seed: int = 0,
                         level: float = 0.95) -> tuple[float, float, float]:
    """Percentile interval for an arbitrary statistic, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    codes, ng = factorize(np.asarray(clusters).ravel())
    order = np.argsort(codes, kind="stable")
    idx_by_cluster = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    point = stat(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, ng, size=ng)
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        draws[b] = stat(values[idx])
    a = (1 - level) / 2
    lo, hi = np.quantile(draws[np.isfinite(draws)], [a, 1 - a])
    return float(point), float(lo), float(hi)
