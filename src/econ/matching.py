"""Making the control group resemble the treated group.

An overlap route for the Delta / Northwest merger is, by construction, a route
where two large network carriers both sold tickets. A route where neither of
them sold anything is a different animal: shorter, thinner, more likely served
by a single low-cost carrier. Comparing the two and calling the difference a
merger effect asks the fixed effects to absorb not just the level gap but the
whole difference in how those two kinds of route were trending, which is more
than a route fixed effect can do.

The fix used here is the standard one for an average effect on the treated.
Estimate the probability that a route is an overlap route given what it looked
like before the merger was announced, then reweight the controls by the odds of
that probability. A control route that looks exactly like a typical treated
route gets a large weight; one that looks nothing like any treated route gets
almost none. Under the usual overlap and selection-on-observables conditions,
the reweighted control group has the same covariate distribution as the treated
group, and the comparison is between like and like.

Whether it worked is a factual question with a factual answer, so
``balance_table`` reports the standardised difference on every covariate before
and after. Those numbers go in the report; a covariate still out of balance
after weighting is a caveat, not something to be quiet about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logit_irls(X: np.ndarray, y: np.ndarray, maxiter: int = 100,
                tol: float = 1e-9, ridge: float = 1e-6) -> np.ndarray:
    """Logistic regression by iteratively reweighted least squares.

    A small ridge term keeps the Hessian invertible when a covariate separates
    the groups almost perfectly, which happens on the thinner mergers.
    """
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(maxiter):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        z = eta + (y - p) / w
        H = X.T @ (X * w[:, None]) + ridge * np.eye(k)
        g = X.T @ (w * z)
        try:
            b_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return b
        if np.max(np.abs(b_new - b)) < tol:
            return b_new
        b = b_new
    return b


@dataclass
class NNMatchResult:
    weights: np.ndarray
    matched_treated: int
    unmatched_treated: int
    n_controls_used: int
    n_effective_controls: float
    max_control_weight_share: float


def nearest_neighbour_att(X: np.ndarray, treated: np.ndarray,
                          unit_weight: np.ndarray, k: int = 5,
                          caliper_sd: float = 0.25) -> NNMatchResult:
    """Match each treated unit to its k nearest controls on the propensity index.

    Inverse-odds weighting is the textbook answer and it fails badly here. A
    handful of control routes sit at propensities near one, their odds run into
    the tens, and once those are multiplied by traffic the control arm collapses
    onto a few routes: effective sample sizes in the single digits, and
    confidence intervals that say nothing. Matching bounds the damage, because a
    control route can only ever be weighted by the treated routes it is actually
    standing in for.

    Distance is measured on the linear propensity index rather than the
    probability, which is the usual recommendation - it spreads out the crowd of
    units near zero and one instead of compressing them. Treated units with no
    control inside ``caliper_sd`` standard deviations are dropped and counted,
    because a treated route with no comparable control should leave the estimate
    rather than be matched to something that is not like it.

    Each treated unit's weight is split equally among its matches, so the
    reweighted control arm reproduces the treated arm's weight distribution.
    """
    X = np.asarray(X, float)
    t = np.asarray(treated).astype(bool)
    uw = np.asarray(unit_weight, float)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    b = _logit_irls(Xs, t.astype(float))
    idx = Xs @ b                       # linear propensity index

    ti = np.flatnonzero(t)
    ci = np.flatnonzero(~t)
    if ti.size == 0 or ci.size == 0:
        return NNMatchResult(np.zeros(len(X)), 0, int(ti.size), 0, 0.0, 0.0)

    caliper = caliper_sd * np.std(idx)
    order = np.argsort(idx[ci], kind="stable")
    ci_sorted, idx_sorted = ci[order], idx[ci][order]

    w = np.zeros(len(X))
    matched = unmatched = 0
    for i in ti:
        pos = np.searchsorted(idx_sorted, idx[i])
        lo, hi = max(0, pos - k - 2), min(len(ci_sorted), pos + k + 2)
        cand = np.arange(lo, hi)
        d = np.abs(idx_sorted[cand] - idx[i])
        ok = d <= caliper
        if not ok.any():
            unmatched += 1
            continue
        cand, d = cand[ok], d[ok]
        pick = cand[np.argsort(d, kind="stable")[:k]]
        w[i] = uw[i]
        w[ci_sorted[pick]] += uw[i] / len(pick)
        matched += 1

    used = w[ci] > 0
    wc = w[ci][used]
    neff = float(wc.sum() ** 2 / np.sum(wc ** 2)) if wc.size else 0.0
    share = float(wc.max() / wc.sum()) if wc.size else 0.0
    return NNMatchResult(weights=w, matched_treated=matched,
                         unmatched_treated=unmatched,
                         n_controls_used=int(used.sum()),
                         n_effective_controls=neff,
                         max_control_weight_share=share)


@dataclass
class MatchResult:
    weights: np.ndarray          # one per row: 1 for treated, odds for controls
    propensity: np.ndarray
    n_effective_controls: float  # Kish effective sample size of the control arm
    trimmed: int


def propensity_att_weights(X: np.ndarray, treated: np.ndarray,
                           trim: tuple[float, float] = (0.01, 0.99)
                           ) -> MatchResult:
    """Weights that make the controls match the treated on ``X``.

    Treated rows get weight one. Control rows get p/(1-p), the odds of being
    treated. Propensities outside ``trim`` are dropped: a control route with a
    propensity of 0.999 would otherwise carry a weight of a thousand and the
    estimate would rest on that single route.
    """
    X = np.asarray(X, float)
    t = np.asarray(treated, float)
    # Standardise so the ridge penalty means the same thing for every covariate,
    # and so IRLS starts somewhere sensible.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    b = _logit_irls(Xs, t)
    p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ b, -30, 30)))

    lo, hi = trim
    ok = (p > lo) & (p < hi)
    w = np.zeros(len(X))
    w[(t == 1) & ok] = 1.0
    ctrl = (t == 0) & ok
    w[ctrl] = p[ctrl] / (1.0 - p[ctrl])
    # Normalise the control arm to the same total weight as the treated arm, so
    # the two sides of the comparison count equally.
    if w[ctrl].sum() > 0:
        w[ctrl] *= w[(t == 1) & ok].sum() / w[ctrl].sum()
    wc = w[ctrl]
    neff = float(wc.sum() ** 2 / np.sum(wc ** 2)) if wc.sum() > 0 else 0.0
    return MatchResult(weights=w, propensity=p,
                       n_effective_controls=neff, trimmed=int((~ok).sum()))


def standardised_difference(x: np.ndarray, treated: np.ndarray,
                            w: np.ndarray | None = None) -> float:
    """Difference in means over the pooled standard deviation, in SD units.

    The convention is that anything under 0.1 in absolute value counts as
    balanced.
    """
    x = np.asarray(x, float)
    t = np.asarray(treated).astype(bool)
    if w is None:
        w = np.ones(len(x))
    w = np.asarray(w, float)
    wt, wc = w[t], w[~t]
    if wt.sum() <= 0 or wc.sum() <= 0:
        return float("nan")
    mt = np.average(x[t], weights=wt)
    mc = np.average(x[~t], weights=wc)
    vt = np.average((x[t] - mt) ** 2, weights=wt)
    vc = np.average((x[~t] - mc) ** 2, weights=wc)
    pooled = np.sqrt((vt + vc) / 2.0)
    return float((mt - mc) / pooled) if pooled > 0 else float("nan")


def balance_table(X: np.ndarray, treated: np.ndarray, names: list[str],
                  w_before: np.ndarray | None = None,
                  w_after: np.ndarray | None = None) -> list[dict]:
    """Standardised differences on every covariate, before and after weighting."""
    X = np.asarray(X, float)
    rows = []
    for j, nm in enumerate(names):
        rows.append({
            "covariate": nm,
            "before": standardised_difference(X[:, j], treated, w_before),
            "after": standardised_difference(X[:, j], treated, w_after),
        })
    return rows
