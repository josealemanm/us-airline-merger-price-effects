"""Stage 5 - what the structural screens would have said, before the fact.

This is the ex-ante view. For every route where both merging carriers were
selling, it computes the pre-merger HHI from the baseline shares, the increase
the merger would mechanically produce, and whether the 2023 Merger Guidelines
structural presumption would attach.

Everything here uses only information available before the deal closed:
baseline-period shares, measured over the four quarters ending the quarter
before the merger was announced. No outcome data enters. That is the point - the
screens have to be computable at the time an agency would have had to decide,
or comparing them to what happened afterwards proves nothing.

Writes data/processed/screens.parquet and reports/concentration.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ.concentration import hhi, num_effective_competitors, screen_merger


def main() -> int:
    cp = pd.read_parquet(C.DATA_PROCESSED / "panel_carrier.parquet")
    tr = pd.read_parquet(C.DATA_PROCESSED / "treatment.parquet")

    out = []
    for m in C.MERGERS:
        lo, hi = m.announce_q - C.BASELINE_QUARTERS, m.announce_q - 1
        base = cp[(cp["qindex"] >= lo) & (cp["qindex"] <= hi)]
        if base.empty:
            continue
        # Average share over the baseline window, per carrier per route.
        nq = base.groupby("market_id")["qindex"].nunique()
        sh = (base.groupby(["market_id", "carrier"])["share"].sum()
              / nq.reindex(base.groupby(["market_id", "carrier"]).size().index
                           .get_level_values(0)).to_numpy())
        sh = sh.rename("share").reset_index()
        pax = (base.groupby("market_id")["pax"].sum() / nq).rename("pax")
        dist = base.groupby("market_id")["distance"].mean().rename("distance")
        route = base.groupby("market_id")["route"].first()

        treated = set(tr.loc[(tr["merger"] == m.key) & tr["overlap"], "market_id"])
        a, b = m.carriers
        for mid, g in sh.groupby("market_id"):
            if mid not in treated:
                continue
            carriers = list(g["carrier"])
            shares = g["share"].to_numpy(float)
            # Shares are renormalised because the baseline average can drift off
            # one when a carrier enters or exits mid-window.
            shares = shares / shares.sum()
            if a not in carriers or b not in carriers:
                continue
            ia, ib = carriers.index(a), carriers.index(b)
            sc = screen_merger(shares, ia, ib,
                               hhi_threshold=C.HHI_CONCENTRATED,
                               delta_threshold=C.DELTA_HHI_THRESHOLD,
                               share_threshold=C.SHARE_THRESHOLD)
            out.append({
                "merger": m.key, "market_id": mid,
                "route": route.get(mid, mid),
                "hhi_pre": sc.hhi_pre, "hhi_post": sc.hhi_post,
                "delta_hhi": sc.delta, "combined_share": sc.combined_share,
                "share_a": float(shares[ia]), "share_b": float(shares[ib]),
                "n_carriers": len(carriers),
                "n_effective": num_effective_competitors(shares),
                "highly_concentrated": sc.highly_concentrated,
                "flag_hhi": sc.presumption_hhi,
                "flag_share": sc.presumption_share,
                "flagged": sc.flagged,
                "baseline_pax": float(pax.get(mid, np.nan)),
                "distance": float(dist.get(mid, np.nan)),
                "to_monopoly": bool(len(carriers) == 2),
            })

    sc = pd.DataFrame(out)
    sc.to_parquet(C.DATA_PROCESSED / "screens.parquet", index=False)

    # ------------------------------------------------------------- report
    L = ["# Concentration and the structural screens", "",
         "Computed by `src/05_concentration.py` from baseline shares only: the "
         "four quarters ending the quarter before each merger was announced. "
         "No post-merger information is used.", "",
         "## Overlap routes by merger", "",
         "| merger | overlap routes | median pre-merger HHI | median delta HHI "
         "| median combined share | flagged by the presumption | routes going "
         "to monopoly |", "|---|---|---|---|---|---|---|"]
    for m in C.MERGERS:
        g = sc[sc["merger"] == m.key]
        if g.empty:
            L.append(f"| {m.label} | 0 | | | | | |")
            continue
        L.append(f"| {m.label} | {len(g):,} | {g['hhi_pre'].median():,.0f} | "
                 f"{g['delta_hhi'].median():,.0f} | {g['combined_share'].median():.1%} | "
                 f"{g['flagged'].mean():.0%} | {g['to_monopoly'].sum():,} |")
    L += ["",
          f"Across all five mergers there are **{len(sc):,} overlap routes**. "
          f"The structural presumption attaches on **{sc['flagged'].mean():.0%}** "
          f"of them.", ""]

    L += ["## Where the flags come from", "",
          "The 2023 Guidelines set two structural screens. The first asks "
          f"whether the post-merger HHI exceeds {C.HHI_CONCENTRATED:,} and the "
          f"increase exceeds {C.DELTA_HHI_THRESHOLD}. The second asks whether "
          f"the combined share exceeds {C.SHARE_THRESHOLD:.0%} with the same "
          "increase. A route is flagged if either attaches.", "",
          "| screen | routes flagged | share of overlap routes |", "|---|---|---|",
          f"| HHI above {C.HHI_CONCENTRATED:,} and delta above {C.DELTA_HHI_THRESHOLD} "
          f"| {sc['flag_hhi'].sum():,} | {sc['flag_hhi'].mean():.1%} |",
          f"| combined share above {C.SHARE_THRESHOLD:.0%} and delta above "
          f"{C.DELTA_HHI_THRESHOLD} | {sc['flag_share'].sum():,} | "
          f"{sc['flag_share'].mean():.1%} |",
          f"| either | {sc['flagged'].sum():,} | {sc['flagged'].mean():.1%} |", ""]

    # How discriminating is the screen, really?
    L += ["## How much does the screen actually separate?", "",
          "A screen is only useful if it divides the routes into groups that "
          "differ. These are the overlap routes split by whether the "
          "presumption attaches.", "",
          "| | flagged | not flagged |", "|---|---|---|"]
    f, nf = sc[sc["flagged"]], sc[~sc["flagged"]]
    for label, col, fmt in [("routes", None, None),
                            ("median pre-merger HHI", "hhi_pre", "{:,.0f}"),
                            ("median delta HHI", "delta_hhi", "{:,.0f}"),
                            ("median combined share", "combined_share", "{:.1%}"),
                            ("median carriers on the route", "n_carriers", "{:.0f}"),
                            ("median baseline passengers", "baseline_pax", "{:,.0f}"),
                            ("median distance, miles", "distance", "{:,.0f}")]:
        if col is None:
            L.append(f"| {label} | {len(f):,} | {len(nf):,} |")
        else:
            L.append(f"| {label} | {fmt.format(f[col].median())} | "
                     f"{fmt.format(nf[col].median())} |")
    L.append("")

    dist = sc["delta_hhi"]
    L += ["## The distribution of the HHI increase", "",
          "The threshold sits at 100 points. Where the overlap routes actually "
          "fall relative to it decides how much work the threshold is doing.", "",
          "| delta HHI | routes | share |", "|---|---|---|"]
    bins = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 2500),
            (2500, 1e9)]
    for lo, hi in bins:
        n = int(((dist >= lo) & (dist < hi)).sum())
        hi_s = "and above" if hi > 1e8 else f"to {hi:,.0f}"
        L.append(f"| {lo:,.0f} {hi_s} | {n:,} | {n/len(sc):.1%} |")
    L.append("")
    L.append(f"Median increase across all overlap routes: "
             f"**{dist.median():,.0f} points**. "
             f"**{(dist < C.DELTA_HHI_THRESHOLD).mean():.0%}** of overlap routes "
             f"fall below the {C.DELTA_HHI_THRESHOLD}-point threshold on their own.")
    L.append("")

    (C.REPORTS / "concentration.md").write_text("\n".join(L), encoding="utf-8")

    print(f"  {len(sc):,} overlap routes screened across "
          f"{sc['merger'].nunique()} mergers")
    print(f"  flagged by the structural presumption: {sc['flagged'].sum():,} "
          f"({sc['flagged'].mean():.1%})")
    print(f"  median pre-merger HHI {sc['hhi_pre'].median():,.0f}, "
          f"median delta {sc['delta_hhi'].median():,.0f}")
    print()
    for m in C.MERGERS:
        g = sc[sc["merger"] == m.key]
        if g.empty:
            continue
        print(f"  {m.key:<6} {len(g):>6,} routes  "
              f"median dHHI {g['delta_hhi'].median():>7,.0f}  "
              f"flagged {g['flagged'].mean():>5.0%}  "
              f"to monopoly {int(g['to_monopoly'].sum()):>4,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
