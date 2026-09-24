"""Stage 8 - simulating each merger route by route, before it happened.

For every overlap route this stage builds the market as it stood before the
merger was announced - which carriers were selling, at what fares, with what
shares - and then does what a merger simulation does:

1. Invert demand so the model reproduces the observed shares exactly.
2. Solve the pre-merger Bertrand first-order conditions backwards for marginal
   cost. Every carrier's fare is taken to be a best response to its rivals', so
   the markup it is charging reveals what the ticket costs it.
3. Change the ownership matrix so the two merging carriers maximise joint
   profit, and solve the same conditions forwards for the new equilibrium.
4. Read off the fare change, the value of diverted sales (GUPPI), the upward
   pricing pressure net of an efficiency credit (UPP), the cost saving that
   would be needed to hold fares flat (CMCR), and the loss of consumer surplus.

Nothing here uses any information from after the merger closed. That is the
point: these are predictions, and stage 9 scores them against what happened.

Everything is conditional on the calibration argued for in stage 7. The
sensitivity table at the end is not decoration - it is the honest statement of
how much of the answer is assumption. The result worth carrying forward is that
the ranking of routes is close to invariant to it while the level is not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ.bertrand import simulate_merger
from econ.nested_logit import market_from_estimates

EFFICIENCY_CREDIT = 0.10   # the standard credit in the UPP statistic


def baseline_markets(cp: pd.DataFrame, m) -> dict:
    """Each overlap route as it stood in the four quarters before announcement."""
    lo, hi = m.announce_q - C.BASELINE_QUARTERS, m.announce_q - 1
    w = cp[(cp["qindex"] >= lo) & (cp["qindex"] <= hi)]
    if w.empty:
        return {}
    g = w.groupby(["market_id", "carrier"]).agg(
        pax=("pax", "sum"),
        fare=("fare", lambda s: np.average(s)),
        group=("group", "first"))
    fare = (w.assign(fp=w["fare"] * w["pax"]).groupby(["market_id", "carrier"])
            .apply(lambda x: x["fp"].sum() / x["pax"].sum(), include_groups=False))
    g["fare"] = fare
    mk = w.groupby("market_id").agg(
        market_size=("market_size", "mean"),
        inside=("inside_share", "mean"),
        route=("route", "first"))
    out = {}
    for mid, sub in g.groupby(level=0):
        out[mid] = (sub.droplevel(0), mk.loc[mid])
    return out


def run(cp: pd.DataFrame, screens: pd.DataFrame, alpha: float, sigma: float,
        cost_saving: float = 0.0) -> pd.DataFrame:
    """Simulate every overlap route at one parameter setting."""
    rows = []
    for m in C.MERGERS:
        mkts = baseline_markets(cp, m)
        want = set(screens.loc[screens["merger"] == m.key, "market_id"])
        a, b = m.carriers
        for mid in want & set(mkts):
            sub, meta = mkts[mid]
            carriers = list(sub.index)
            if a not in carriers or b not in carriers or len(carriers) < 2:
                continue
            pax = sub["pax"].to_numpy(float)
            shares_inside = pax / pax.sum()
            inside = float(np.clip(meta["inside"], 1e-6, 0.95))
            shares = shares_inside * inside
            outside = 1.0 - inside
            prices = sub["fare"].to_numpy(float)
            if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
                continue
            groups = (np.zeros(len(carriers), int) if sigma == 0.0
                      else np.array([0 if g == "network" else 1
                                     for g in sub["group"]]))
            try:
                model = market_from_estimates(shares, prices, groups, alpha,
                                              sigma, outside)
                sim = simulate_merger(model, np.array(carriers), (a, b),
                                      market_size=float(meta["market_size"]),
                                      efficiency=EFFICIENCY_CREDIT,
                                      cost_saving=cost_saving)
            except Exception:  # noqa: BLE001 - a market that will not solve is dropped
                continue
            party = np.isin(carriers, [a, b])
            wts = shares_inside / shares_inside.sum()
            wp = shares_inside[party] / shares_inside[party].sum()
            rows.append({
                "merger": m.key, "market_id": mid, "route": str(meta["route"]),
                "n_carriers": len(carriers),
                "sim_dp_market": float(np.sum(wts * sim.pct_price_change)),
                "sim_dp_merging": float(np.sum(wp * sim.pct_price_change[party])),
                "guppi_max": float(sim.guppi.max()),
                "guppi_wavg": float(np.sum(wp * sim.guppi[party])),
                "upp_max": float(sim.upp[party].max()),
                "cmcr_max": (float(np.nanmax(sim.cmcr[party]))
                             if np.any(np.isfinite(sim.cmcr[party])) else np.nan),
                "diversion_to_partner": float(np.sum(wp * sim.diversion_to_partner[party])),
                "margin_pre": float(np.sum(wts * sim.margins_pre)),
                "consumer_harm_q": float(sim.consumer_harm),
                "baseline_pax": float(pax.sum()),
                "converged": bool(sim.converged),
            })
    return pd.DataFrame(rows)


def main() -> int:
    cp = pd.read_parquet(C.DATA_PROCESSED / "panel_carrier.parquet")
    screens = pd.read_parquet(C.DATA_PROCESSED / "screens.parquet")
    params = json.loads((C.DATA_PROCESSED / "demand_params.json").read_text())
    cal = params["calibrated"]
    alpha = cal["alpha_per_dollar"]
    sigma = cal["sigma"]

    sim = run(cp, screens, alpha, sigma)
    ok = sim[sim["converged"]].copy()
    print(f"  simulated {len(sim):,} overlap routes, "
          f"{len(ok):,} converged ({len(ok)/max(len(sim),1):.1%})")

    # Passengers are scaled from the DB1B ten per cent sample, and the harm
    # figure is per quarter, so it is multiplied out to an annual rate.
    ok["consumer_harm_annual"] = (ok["consumer_harm_q"] / C.DB1B_SAMPLE_RATE * 4)
    ok.to_parquet(C.DATA_PROCESSED / "simulation.parquet", index=False)

    print(f"  median simulated fare change, merging carriers: "
          f"{ok['sim_dp_merging'].median()*100:+.2f}%")
    print(f"  median simulated fare change, whole route:      "
          f"{ok['sim_dp_market'].median()*100:+.2f}%")
    print(f"  median GUPPI: {ok['guppi_wavg'].median():.3f}   "
          f"median diversion to partner: {ok['diversion_to_partner'].median():.3f}")
    print(f"  median pre-merger margin: {ok['margin_pre'].median():.3f}")

    # ------------------------------------------------------------ sensitivity
    sens = []
    # A route can be an overlap route for more than one merger, so the key is
    # the pair, not the route.
    base_rank = ok[["merger", "market_id", "sim_dp_market"]].rename(
        columns={"sim_dp_market": "_base"})
    for alt in cal["alternatives"]:
        s = run(cp, screens, alt["alpha_per_dollar"], alt["sigma"])
        s = s[s["converged"]]
        j = s.merge(base_rank, on=["merger", "market_id"], how="inner")
        keep = np.isfinite(j["_base"]) & np.isfinite(j["sim_dp_market"])
        rho = (float(j.loc[keep, "_base"].corr(
            j.loc[keep, "sim_dp_market"], method="spearman"))
            if keep.sum() > 10 else float("nan"))
        sens.append({
            "target_elasticity": alt["target_elasticity"], "sigma": alt["sigma"],
            "median_dp_market": float(s["sim_dp_market"].median()),
            "median_dp_merging": float(s["sim_dp_merging"].median()),
            "median_guppi": float(s["guppi_wavg"].median()),
            "median_margin": float(s["margin_pre"].median()),
            "harm_annual_total": float(
                (s["consumer_harm_q"] / C.DB1B_SAMPLE_RATE * 4).sum()),
            "rank_corr_with_main": rho,
            "n": int(len(s)),
        })
        print(f"    elasticity {alt['target_elasticity']:+.1f}, sigma "
              f"{alt['sigma']:.2f}: median route fare change "
              f"{s['sim_dp_market'].median()*100:+.2f}%, rank correlation with "
              f"the reported run {rho:.3f}")

    summary = {
        "alpha_per_dollar": alpha, "sigma": sigma,
        "n_simulated": int(len(sim)), "n_converged": int(len(ok)),
        "median_dp_market": float(ok["sim_dp_market"].median()),
        "median_dp_merging": float(ok["sim_dp_merging"].median()),
        "median_guppi": float(ok["guppi_wavg"].median()),
        "median_diversion": float(ok["diversion_to_partner"].median()),
        "median_margin": float(ok["margin_pre"].median()),
        "harm_annual_total": float(ok["consumer_harm_annual"].sum()),
        "by_merger": {},
        "sensitivity": sens,
    }
    for m in C.MERGERS:
        g = ok[ok["merger"] == m.key]
        if g.empty:
            continue
        summary["by_merger"][m.key] = {
            "label": m.label, "n": int(len(g)),
            "median_dp_market": float(g["sim_dp_market"].median()),
            "median_guppi": float(g["guppi_wavg"].median()),
            "harm_annual": float(g["consumer_harm_annual"].sum()),
        }
    (C.DATA_PROCESSED / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    # ---------------------------------------------------------------- report
    L = ["# Merger simulation", "",
         "Run by `src/08_merger_sim.py` on "
         f"{len(ok):,} overlap routes, using only pre-announcement fares and "
         "shares. Demand parameters are the calibration argued for in "
         "[`demand.md`](demand.md); every number below is conditional on it.", "",
         "## What the simulation predicts", "",
         "| | median across routes |", "|---|---|",
         f"| fare change, merging carriers | {ok['sim_dp_merging'].median()*100:+.2f}% |",
         f"| fare change, whole route | {ok['sim_dp_market'].median()*100:+.2f}% |",
         f"| GUPPI | {ok['guppi_wavg'].median():.3f} |",
         f"| diversion to the merger partner | {ok['diversion_to_partner'].median():.3f} |",
         f"| pre-merger margin implied by the FOC | {ok['margin_pre'].median():.3f} |",
         f"| cost saving needed to hold fares flat (CMCR) | "
         f"{ok['cmcr_max'].median():.3f} |", "",
         f"Summed over every overlap route, the predicted loss of consumer "
         f"surplus is **${ok['consumer_harm_annual'].sum()/1e9:.2f} billion a "
         f"year**. That figure is the one most exposed to the calibration; see "
         "the sensitivity table.", "",
         "## By merger", "",
         "| merger | routes | median route fare change | median GUPPI | "
         "predicted annual harm |", "|---|---|---|---|---|"]
    for m in C.MERGERS:
        g = ok[ok["merger"] == m.key]
        if g.empty:
            continue
        L.append(f"| {m.label} | {len(g):,} | {g['sim_dp_market'].median()*100:+.2f}% "
                 f"| {g['guppi_wavg'].median():.3f} | "
                 f"${g['consumer_harm_annual'].sum()/1e6:,.0f}M |")
    L += ["", "## How much of this is the calibration?", "",
          "The same simulation at demand parameters either side of the ones "
          "reported. The last column is the rank correlation between each "
          "run's route-level fare predictions and the reported run's.", "",
          "| target elasticity | sigma | median route fare change | median "
          "margin | annual harm | rank correlation |", "|---|---|---|---|---|---|",
          f"| **{cal['target_elasticity']:+.1f}** | **{sigma:.2f}** | "
          f"**{ok['sim_dp_market'].median()*100:+.2f}%** | "
          f"**{ok['margin_pre'].median():.3f}** | "
          f"**${ok['consumer_harm_annual'].sum()/1e9:.2f}B** | **1.000** |"]
    for s in sens:
        L.append(f"| {s['target_elasticity']:+.1f} | {s['sigma']:.2f} | "
                 f"{s['median_dp_market']*100:+.2f}% | {s['median_margin']:.3f} | "
                 f"${s['harm_annual_total']/1e9:.2f}B | "
                 f"{s['rank_corr_with_main']:.3f} |")
    rho_min = min((s["rank_corr_with_main"] for s in sens
                   if np.isfinite(s["rank_corr_with_main"])), default=float("nan"))
    L += ["",
          f"The level moves a great deal and the ordering barely moves: the "
          f"lowest rank correlation across every alternative is "
          f"**{rho_min:.3f}**. Under logit demand the diversion ratio between "
          "two carriers depends on their market shares and not on the price "
          "coefficient, so which routes look worst is settled by market "
          "structure, which is observed, rather than by the calibration, which "
          "is assumed. The dollar figures should be read as scaled to an "
          "assumption. The ranking should not.", "",
          "That distinction is what makes stage 9 possible. It compares "
          "predictions with outcomes by ordering and by bin, not by level.", ""]
    (C.REPORTS / "simulation.md").write_text("\n".join(L), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
