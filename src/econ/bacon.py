"""Goodman-Bacon decomposition of a staggered two-way fixed effects estimate.

Goodman-Bacon (2021) proves that the two-way fixed effects coefficient on a
staggered binary treatment is a weighted average of all the two-by-two
difference-in-differences you could form from the data, with weights that depend
only on group sizes and on how much of the window each group spends treated.

Three kinds of comparison show up:

    treated vs never-treated        clean
    earlier-treated vs later-treated, using the later group before it is
                                    treated as the control            clean
    later-treated vs earlier-treated, using the earlier group *after* it is
                                    already treated as the control    not clean

The third kind is the problem. If the earlier merger's effect is still growing
when the later merger closes, that growth enters the control group's change and
is subtracted from the later merger's estimate, with a negative sign. The point
of running this decomposition is to report how much weight sits on those
comparisons instead of asserting that it is small.

The implementation computes the weights from Theorem 1, normalises them, and
then checks the decomposition against the actual regression coefficient. If the
two do not agree the function raises rather than returning a number nobody
should trust.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fe_ols import factorize, feols


@dataclass
class BaconComponent:
    kind: str          # "treated_vs_never", "earlier_vs_later", "later_vs_earlier"
    group_a: int       # treatment period of the group acting as treated
    group_b: int       # treatment period of the comparison group
    beta: float
    weight: float
    n_units_a: int
    n_units_b: int


@dataclass
class BaconResult:
    twfe: float
    components: list[BaconComponent]
    reconstructed: float

    def by_kind(self) -> dict[str, tuple[float, float]]:
        """Total weight and weighted-average beta within each comparison type."""
        out = {}
        for kind in ("treated_vs_never", "earlier_vs_later", "later_vs_earlier"):
            sel = [c for c in self.components if c.kind == kind]
            w = sum(c.weight for c in sel)
            b = sum(c.weight * c.beta for c in sel) / w if w > 1e-12 else float("nan")
            out[kind] = (w, b)
        return out

    def forbidden_weight(self) -> float:
        """Share of the estimate that rests on already-treated controls."""
        return self.by_kind()["later_vs_earlier"][0]

    def summary(self) -> str:
        lines = [f"  two-way FE coefficient      {self.twfe:+.5f}",
                 f"  sum of weighted components  {self.reconstructed:+.5f}",
                 "",
                 "  {:<34} {:>8} {:>10}".format("comparison", "weight", "avg beta")]
        pretty = {"treated_vs_never": "treated vs never-treated",
                  "earlier_vs_later": "earlier vs later (clean)",
                  "later_vs_earlier": "later vs earlier (contaminated)"}
        for kind, (w, b) in self.by_kind().items():
            lines.append("  {:<34} {:>8.3f} {:>10.5f}".format(pretty[kind], w, b))
        return "\n".join(lines)


def _balanced(y, unit, period, gvar):
    """Keep only units observed in every period; the decomposition needs it."""
    unit = np.asarray(unit)
    period = np.asarray(period, np.int64)
    ucodes, nu = factorize(unit)
    periods = np.unique(period)
    pidx = {p: i for i, p in enumerate(periods)}
    Y = np.full((nu, periods.size), np.nan)
    Y[ucodes, [pidx[p] for p in period]] = np.asarray(y, float)
    G = np.zeros(nu, np.int64)
    G[ucodes] = np.asarray(gvar, np.int64)
    keep = np.all(np.isfinite(Y), axis=1)
    return Y[keep], G[keep], periods


def bacon_decompose(y: np.ndarray, unit: np.ndarray, period: np.ndarray,
                    gvar: np.ndarray, never_sentinel: int | None = None,
                    tol: float = 1e-6) -> BaconResult:
    """Decompose the staggered TWFE estimate into its two-by-two parts.

    ``gvar`` is the period each unit is first treated; never-treated units carry
    ``never_sentinel`` (by default, anything past the last period).
    """
    Y, G, periods = _balanced(y, unit, period, gvar)
    T = periods.size
    if never_sentinel is None:
        never_sentinel = int(periods.max()) + 1
    pidx = {p: i for i, p in enumerate(periods)}

    # The regression the decomposition is explaining: unit and period effects,
    # run on the balanced sample so the two objects describe the same data.
    nu = Y.shape[0]
    yl = Y.ravel()
    ul = np.repeat(np.arange(nu), T)
    pl = np.tile(periods, nu)
    dl = (pl >= np.repeat(G, T)).astype(float)
    twfe = float(feols(yl, dl[:, None], fes=[factorize(ul), factorize(pl)],
                       names=["d"]).params[0])

    treat_times = sorted({int(g) for g in G if g != never_sentinel})
    never = G == never_sentinel
    n_total = nu

    def share(mask) -> float:
        return float(mask.sum()) / n_total

    def dbar(tstar: int) -> float:
        """Fraction of the window a group treated at tstar spends treated."""
        return float(np.sum(periods >= tstar)) / T

    def gmean(mask, lo, hi) -> float:
        """Mean of y for these units over periods in [lo, hi)."""
        cols = [pidx[p] for p in periods if lo <= p < hi]
        if not cols or not mask.any():
            return float("nan")
        return float(Y[np.ix_(mask, cols)].mean())

    comps: list[BaconComponent] = []
    hi_end = int(periods.max()) + 1
    lo_end = int(periods.min())

    # ---- treated k against never-treated
    if never.any():
        n_u = share(never)
        for tk in treat_times:
            mk = G == tk
            n_k = share(mk)
            if n_k == 0:
                continue
            nbar = n_k / (n_k + n_u)
            dk = dbar(tk)
            w = (n_k + n_u) ** 2 * nbar * (1 - nbar) * dk * (1 - dk)
            beta = ((gmean(mk, tk, hi_end) - gmean(mk, lo_end, tk))
                    - (gmean(never, tk, hi_end) - gmean(never, lo_end, tk)))
            comps.append(BaconComponent("treated_vs_never", tk, never_sentinel,
                                        beta, w, int(mk.sum()), int(never.sum())))

    # ---- timing pairs
    for i, tk in enumerate(treat_times):
        for tl in treat_times[i + 1:]:
            mk, ml = G == tk, G == tl        # k is treated earlier than l
            n_k, n_l = share(mk), share(ml)
            if n_k == 0 or n_l == 0:
                continue
            nbar = n_k / (n_k + n_l)
            dk, dl_ = dbar(tk), dbar(tl)     # dk > dl_ because tk < tl

            # (a) k treated, l still untreated and acting as control
            w_a = ((n_k + n_l) * (1 - dl_)) ** 2 * nbar * (1 - nbar) \
                * ((dk - dl_) * (1 - dk)) / (1 - dl_) ** 2
            beta_a = ((gmean(mk, tk, tl) - gmean(mk, lo_end, tk))
                      - (gmean(ml, tk, tl) - gmean(ml, lo_end, tk)))
            comps.append(BaconComponent("earlier_vs_later", tk, tl, beta_a, w_a,
                                        int(mk.sum()), int(ml.sum())))

            # (b) l treated, k already treated and acting as control
            w_b = ((n_k + n_l) * dk) ** 2 * nbar * (1 - nbar) \
                * (dl_ * (dk - dl_)) / dk ** 2
            beta_b = ((gmean(ml, tl, hi_end) - gmean(ml, tk, tl))
                      - (gmean(mk, tl, hi_end) - gmean(mk, tk, tl)))
            comps.append(BaconComponent("later_vs_earlier", tl, tk, beta_b, w_b,
                                        int(ml.sum()), int(mk.sum())))

    total_w = sum(c.weight for c in comps)
    if total_w <= 0:
        raise ValueError("no usable two-by-two comparisons")
    for c in comps:
        c.weight /= total_w

    recon = sum(c.weight * c.beta for c in comps if np.isfinite(c.beta))
    if not np.isfinite(recon) or abs(recon - twfe) > max(tol, tol * abs(twfe)):
        raise AssertionError(
            f"decomposition does not reproduce the regression: "
            f"components sum to {recon:.8f}, regression gives {twfe:.8f}")
    return BaconResult(twfe=twfe, components=comps, reconstructed=recon)
