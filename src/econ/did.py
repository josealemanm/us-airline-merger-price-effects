"""Difference-in-differences estimators for staggered treatment.

Five airline mergers close in five different quarters, which is exactly the
setting where the familiar two-way fixed effects regression stops answering the
question people think it answers. When treatment is staggered, two-way fixed
effects computes a weighted average of every available two-by-two comparison,
and some of those comparisons use *already-treated* routes as the control group
for later-treated ones. If the effect of a merger changes over time, and it
does, those comparisons enter with the wrong sign and can drag the average
anywhere.

This module gives three things:

    event_study      leads and lags around one event, which is what shows
                     whether the pre-trends are flat
    stacked_did      the estimator this project reports: each merger gets its
                     own clean control group, and the stacks are pooled
    att_gt           group-time average treatment effects in the
                     Callaway-Sant'Anna spirit, built from not-yet-treated
                     controls only

``bacon.py`` measures how badly the naive estimator is contaminated, which is
the argument for using the ones here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fe_ols import FEResult, factorize, feols


# ------------------------------------------------------------- event dummies
def event_time_dummies(event_time: np.ndarray, treated: np.ndarray,
                       kmin: int, kmax: int, ref: int = -1,
                       bin_ends: bool = True) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build treated-by-event-time indicators.

    ``ref`` is the omitted period, normally the quarter before the event, so
    every coefficient reads as a change relative to the eve of the merger.
    ``bin_ends`` pools everything past the window edges into the end points,
    which keeps the sample composition from shifting as routes enter and leave.
    """
    et = np.asarray(event_time, dtype=np.int64).copy()
    tr = np.asarray(treated).astype(float)
    if bin_ends:
        et = np.clip(et, kmin, kmax)
    ks = [k for k in range(kmin, kmax + 1) if k != ref]
    cols, names = [], []
    for k in ks:
        cols.append(tr * (et == k))
        names.append(f"k{k:+d}")
    return np.column_stack(cols), names, np.array(ks)


@dataclass
class EventStudy:
    ks: np.ndarray
    coef: np.ndarray
    se: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    ref: int
    result: FEResult
    pretrend_f: float
    pretrend_p: float

    def as_rows(self) -> list[dict]:
        rows = []
        for i, k in enumerate(self.ks):
            rows.append({"k": int(k), "coef": float(self.coef[i]),
                         "se": float(self.se[i]), "ci_lo": float(self.ci_lo[i]),
                         "ci_hi": float(self.ci_hi[i])})
        rows.append({"k": self.ref, "coef": 0.0, "se": 0.0,
                     "ci_lo": 0.0, "ci_hi": 0.0})
        return sorted(rows, key=lambda r: r["k"])


def event_study(y, event_time, treated, fes, cluster, weights=None,
                kmin: int = -8, kmax: int = 12, ref: int = -1,
                extra_X: np.ndarray | None = None,
                extra_names: list[str] | None = None) -> EventStudy:
    """Leads and lags around a single event date, with a joint pre-trend test."""
    D, names, ks = event_time_dummies(event_time, treated, kmin, kmax, ref)
    X, nm = D, list(names)
    if extra_X is not None:
        X = np.column_stack([X, extra_X])
        nm = nm + list(extra_names or [f"c{i}" for i in range(extra_X.shape[1])])
    res = feols(y, X, fes=fes, cluster=cluster, weights=weights, names=nm)

    # Some event-time columns can be dropped as collinear; report only survivors.
    kept = [n for n in names if n in res.names]
    idx = [res.names.index(n) for n in kept]
    kk = np.array([int(n[1:]) for n in kept])
    ci = res.conf_int()
    pre = [n for n in kept if int(n[1:]) < ref]
    f, p = res.wald(pre) if len(pre) > 1 else (float("nan"), float("nan"))
    return EventStudy(ks=kk, coef=res.params[idx], se=res.se[idx],
                      ci_lo=ci[idx, 0], ci_hi=ci[idx, 1], ref=ref, result=res,
                      pretrend_f=f, pretrend_p=p)


# -------------------------------------------------------------- stacked panel
@dataclass
class Stack:
    """A long table with one block per event, ready for a pooled regression."""
    event: np.ndarray        # which merger this row belongs to
    unit: np.ndarray         # route id
    period: np.ndarray       # calendar quarter index
    event_time: np.ndarray   # period - close quarter
    treated: np.ndarray      # 1 if this route is an overlap route for this event
    post: np.ndarray         # 1 if period >= close quarter
    y: np.ndarray
    weight: np.ndarray
    extras: dict[str, np.ndarray]

    def __len__(self) -> int:
        return self.y.size

    @property
    def unit_by_event(self) -> np.ndarray:
        """Route identity within a stack: the same route in two stacks is two
        units, which is what keeps each merger's comparison self-contained."""
        return np.char.add(np.char.add(self.event.astype(str), "|"),
                           self.unit.astype(str))

    @property
    def period_by_event(self) -> np.ndarray:
        return np.char.add(np.char.add(self.event.astype(str), "|"),
                           self.period.astype(str))


def stacked_did(stack: Stack, cluster_on: np.ndarray | None = None) -> FEResult:
    """The pooled stacked estimator: one treatment-by-post coefficient.

    Fixed effects are event-by-route and event-by-quarter, so identification
    comes only from comparisons inside a single merger's window against that
    merger's own clean controls. Clustering defaults to the route, since a route
    can appear in more than one stack.
    """
    did = stack.treated * stack.post
    cl = stack.unit if cluster_on is None else cluster_on
    return feols(
        stack.y, did[:, None],
        fes=[factorize(stack.unit_by_event), factorize(stack.period_by_event)],
        cluster=cl, weights=stack.weight, names=["did"],
    )


def stacked_event_study(stack: Stack, kmin: int = -8, kmax: int = 12,
                        ref: int = -1,
                        cluster_on: np.ndarray | None = None) -> EventStudy:
    """The same stack, but with the effect allowed to differ by quarter."""
    cl = stack.unit if cluster_on is None else cluster_on
    return event_study(
        stack.y, stack.event_time, stack.treated,
        fes=[factorize(stack.unit_by_event), factorize(stack.period_by_event)],
        cluster=cl, weights=stack.weight, kmin=kmin, kmax=kmax, ref=ref,
    )


# --------------------------------------------------- group-time average effects
@dataclass
class ATTGT:
    group: np.ndarray     # cohort, by treatment period
    period: np.ndarray
    att: np.ndarray
    se: np.ndarray
    n_treated: np.ndarray

    def aggregate_dynamic(self, kmin: int, kmax: int) -> dict[int, tuple[float, float]]:
        """Average ATT(g,t) by event time, weighting cohorts by size."""
        out: dict[int, tuple[float, float]] = {}
        k = self.period - self.group
        for kk in range(kmin, kmax + 1):
            m = (k == kk) & np.isfinite(self.att)
            if not m.any():
                continue
            w = self.n_treated[m].astype(float)
            w = w / w.sum()
            est = float(np.sum(w * self.att[m]))
            # Cohort estimates share control routes, so this is an approximation
            # that ignores their covariance; it is reported as a cross-check on
            # the stacked estimator, not as the headline.
            se = float(np.sqrt(np.sum((w * self.se[m]) ** 2)))
            out[kk] = (est, se)
        return out


def att_gt(y: np.ndarray, unit: np.ndarray, period: np.ndarray,
           gvar: np.ndarray, weights: np.ndarray | None = None,
           anticipation: int = 0) -> ATTGT:
    """Group-time average treatment effects against not-yet-treated controls.

    ``gvar`` is the period a unit is first treated, or a large sentinel for
    never-treated units. For each cohort g and period t, the comparison is the
    change in y between t and g-1-anticipation for cohort g, minus the same
    change for units not yet treated at t. No already-treated unit is ever used
    as a control, which is the contamination that ``bacon.py`` quantifies.
    """
    y = np.asarray(y, float)
    unit = np.asarray(unit)
    period = np.asarray(period, np.int64)
    gvar = np.asarray(gvar, np.int64)
    w = np.ones(y.size) if weights is None else np.asarray(weights, float)

    ucodes, nu = factorize(unit)
    periods = np.unique(period)
    cohorts = np.unique(gvar[gvar < periods.max() + 1])
    cohorts = cohorts[cohorts > periods.min()]

    # Wide lookup: value of y for (unit, period).
    pidx = {p: i for i, p in enumerate(periods)}
    Y = np.full((nu, periods.size), np.nan)
    W = np.zeros((nu, periods.size))
    Y[ucodes, [pidx[p] for p in period]] = y
    W[ucodes, [pidx[p] for p in period]] = w
    G = np.full(nu, periods.max() + 1, dtype=np.int64)
    G[ucodes] = gvar

    g_out, t_out, a_out, s_out, n_out = [], [], [], [], []
    for g in cohorts:
        base = g - 1 - anticipation
        if base not in pidx:
            continue
        b = pidx[base]
        treat_mask = G == g
        if treat_mask.sum() < 2:
            continue
        for t in periods:
            if t == base:
                continue
            ti = pidx[t]
            # Controls: units treated strictly after max(t, g), plus never-treated.
            ctrl_mask = G > max(t, g)
            ok_t = np.isfinite(Y[:, ti]) & np.isfinite(Y[:, b])
            tm, cm = treat_mask & ok_t, ctrl_mask & ok_t
            if tm.sum() < 2 or cm.sum() < 2:
                continue
            dt = Y[:, ti] - Y[:, b]
            wt = np.minimum(W[:, ti], W[:, b])
            mt = np.average(dt[tm], weights=wt[tm])
            mc = np.average(dt[cm], weights=wt[cm])
            vt = _wvar(dt[tm], wt[tm]) / max(tm.sum(), 1)
            vc = _wvar(dt[cm], wt[cm]) / max(cm.sum(), 1)
            g_out.append(g); t_out.append(t)
            a_out.append(mt - mc); s_out.append(np.sqrt(vt + vc))
            n_out.append(int(tm.sum()))

    return ATTGT(np.array(g_out), np.array(t_out), np.array(a_out),
                 np.array(s_out), np.array(n_out))


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    m = np.average(x, weights=w)
    return float(np.average((x - m) ** 2, weights=w) * x.size / (x.size - 1))
