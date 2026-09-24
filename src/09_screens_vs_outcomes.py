"""Stage 9 - scoring the predictions against what actually happened.

Every previous stage exists to make this one possible. Stage 5 computed what the
structural screens would have said before each merger. Stage 8 computed what a
merger simulation would have predicted, from the same pre-merger information.
Stage 6 measured what fares actually did afterwards. This stage puts the three
side by side and asks the question the project is named for: do the tools an
agency reaches for first pick out the routes where fares actually rose?

The comparison has to be made carefully, because a single route's realised
effect is a difference of two noisy quarterly averages and is close to useless
on its own. Three things follow from that.

*Bin, do not rank individually.* Routes are sorted by each prediction and
grouped, and the average realised effect is computed within each group. Noise
averages out inside a bin; signal does not.

*Regress the realised effect on the prediction, not the other way round.* The
prediction is a model output and carries no sampling error; the outcome carries
all of it. In that direction the noise widens the interval and does not bias the
slope. In the other direction it would attenuate it toward zero and manufacture
the conclusion that the screens do not work.

*Report two different nulls.* A slope of zero means the prediction carries no
information about what happened. A slope of one means it is not just informative
but correctly scaled. A screen can be useful and badly scaled at the same time,
and those are different findings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ.fe_ols import factorize, feols

# The predictions being scored, in the order they would be reached for in a
# real review: cheapest first.
PREDICTORS = {
    "delta_hhi": ("HHI increase", "points", 1.0),
    "combined_share": ("combined share of the two parties", "share", 1.0),
    "diversion_to_partner": ("diversion to the merger partner", "share", 1.0),
    "guppi_wavg": ("GUPPI", "share of fare", 1.0),
    "sim_dp_market": ("simulated fare change on the route", "log points", 1.0),
}


def binned(df: pd.DataFrame, col: str, nbins: int = 10) -> pd.DataFrame:
    """Average realised effect within equal-count bins of a prediction."""
    d = df.dropna(subset=[col, "effect"]).copy()
    if len(d) < nbins * 10:
        return pd.DataFrame()
    # Ties at zero are common in the screens, so rank before cutting.
    d["_bin"] = pd.qcut(d[col].rank(method="first"), nbins, labels=False)
    out = []
    for b, g in d.groupby("_bin"):
        eff = g["effect"].to_numpy()
        out.append({
            "bin": int(b) + 1, "n": len(g),
            "pred_mean": float(g[col].mean()),
            "pred_lo": float(g[col].min()), "pred_hi": float(g[col].max()),
            "realised": float(eff.mean()),
            "se": float(eff.std(ddof=1) / np.sqrt(len(eff))),
        })
    return pd.DataFrame(out)


def slope_test(df: pd.DataFrame, col: str) -> dict:
    """Regress realised effect on the prediction, with merger fixed effects."""
    d = df.dropna(subset=[col, "effect"])
    if len(d) < 50:
        return {}
    x = d[col].to_numpy(float)
    # Standardise so slopes across predictors on different scales can be read
    # against each other; the raw slope is reported too.
    sd = x.std()
    r = feols(d["effect"].to_numpy(), x[:, None],
              fes=[factorize(d["merger"])], cluster=d["market_id"].to_numpy(),
              names=[col])
    b, se = float(r.params[0]), float(r.se[0])
    rs = feols(d["effect"].to_numpy(), ((x - x.mean()) / sd)[:, None],
               fes=[factorize(d["merger"])], cluster=d["market_id"].to_numpy(),
               names=[col])
    from scipy import stats as st
    t1 = (b - 1.0) / se if se > 0 else np.nan
    return {
        "slope": b, "se": se,
        "ci_lo": float(r.conf_int()[0][0]), "ci_hi": float(r.conf_int()[0][1]),
        "p_zero": float(r.pvalue[0]),
        "p_one": float(2 * st.t.sf(abs(t1), df=max(r.df_resid, 1)))
        if np.isfinite(t1) else np.nan,
        "slope_per_sd": float(rs.params[0]), "se_per_sd": float(rs.se[0]),
        "n": int(len(d)),
        "spearman": float(d[col].corr(d["effect"], method="spearman")),
    }


def group_means(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Mean realised effect inside and outside a group, and the difference."""
    a = df.loc[mask, "effect"].to_numpy()
    b = df.loc[~mask, "effect"].to_numpy()
    if len(a) < 5 or len(b) < 5:
        return {}
    ma, mb = a.mean(), b.mean()
    sa = a.std(ddof=1) / np.sqrt(len(a))
    sb = b.std(ddof=1) / np.sqrt(len(b))
    diff = ma - mb
    sd = np.sqrt(sa ** 2 + sb ** 2)
    return {"in_mean": float(ma), "in_se": float(sa), "in_n": int(len(a)),
            "out_mean": float(mb), "out_se": float(sb), "out_n": int(len(b)),
            "diff": float(diff), "diff_se": float(sd),
            "diff_lo": float(diff - 1.96 * sd), "diff_hi": float(diff + 1.96 * sd)}


def main() -> int:
    eff = pd.read_parquet(C.DATA_PROCESSED / "route_effects.parquet")
    sc = pd.read_parquet(C.DATA_PROCESSED / "screens.parquet")
    sim = pd.read_parquet(C.DATA_PROCESSED / "simulation.parquet")

    df = (eff.drop(columns=[c for c in ("delta_hhi", "hhi_pre", "combined_share",
                                        "flagged", "n_carriers")
                            if c in eff.columns])
          .merge(sc, on=["merger", "market_id"], how="inner")
          .merge(sim[["merger", "market_id", "diversion_to_partner", "guppi_wavg",
                      "sim_dp_market", "sim_dp_merging", "margin_pre",
                      "consumer_harm_annual"]],
                 on=["merger", "market_id"], how="left"))
    df = df.dropna(subset=["effect"])
    print(f"  {len(df):,} overlap routes with both a prediction and a realised effect")

    RES: dict = {"n_routes": int(len(df))}

    # ------------------------------------------------- 1. does the flag work?
    RES["flag"] = group_means(df, df["flagged"])
    f = RES["flag"]
    print(f"  flagged routes:   {f['in_mean']*100:+.2f}% "
          f"(n={f['in_n']:,})")
    print(f"  unflagged routes: {f['out_mean']*100:+.2f}% (n={f['out_n']:,})")
    print(f"  difference:       {f['diff']*100:+.2f}% "
          f"[{f['diff_lo']*100:+.2f}, {f['diff_hi']*100:+.2f}]")

    # ------------------------------------------------ 2. slope of each predictor
    RES["slopes"] = {}
    print()
    for col in PREDICTORS:
        if col not in df.columns:
            continue
        s = slope_test(df, col)
        if s:
            RES["slopes"][col] = s
            print(f"  {PREDICTORS[col][0]:<38} effect per SD: "
                  f"{s['slope_per_sd']*100:+.3f}% "
                  f"(se {s['se_per_sd']*100:.3f})  rho={s['spearman']:+.3f}")

    # ------------------------------------------------------ 3. binned profiles
    RES["bins"] = {}
    for col in PREDICTORS:
        if col not in df.columns:
            continue
        b = binned(df, col)
        if not b.empty:
            RES["bins"][col] = b.to_dict("records")
    pd.concat(
        [pd.DataFrame(v).assign(predictor=k) for k, v in RES["bins"].items()],
        ignore_index=True
    ).to_parquet(C.DATA_PROCESSED / "calibration_bins.parquet", index=False)

    # ---------------------------------------------- 4. where should the line go?
    # For a range of candidate HHI-increase thresholds, what the flagged group
    # looks like and how much of the realised fare increase it captures.
    thresholds = [0, 50, 100, 200, 300, 500, 800, 1200, 2000]
    rows = []
    tot_pos = df.loc[df["effect"] > 0, "effect"].sum()
    for t in thresholds:
        m = df["delta_hhi"] > t
        if m.sum() < 20 or (~m).sum() < 20:
            continue
        g = group_means(df, m)
        captured = df.loc[m & (df["effect"] > 0), "effect"].sum() / tot_pos \
            if tot_pos > 0 else np.nan
        rows.append({"threshold": t, "flagged": int(m.sum()),
                     "share_flagged": float(m.mean()),
                     "mean_in": g["in_mean"], "mean_out": g["out_mean"],
                     "diff": g["diff"], "diff_se": g["diff_se"],
                     "share_of_increases_captured": float(captured)})
    RES["thresholds"] = rows

    # -------------------------------------------------- 5. predicted vs realised
    have = df.dropna(subset=["sim_dp_market"])
    RES["level"] = {
        "median_predicted": float(have["sim_dp_market"].median()),
        "median_realised": float(have["effect"].median()),
        "mean_predicted": float(have["sim_dp_market"].mean()),
        "mean_realised": float(have["effect"].mean()),
    }

    (C.DATA_PROCESSED / "validation.json").write_text(
        json.dumps(RES, indent=2, default=float))

    # ------------------------------------------------------------- report
    L = ["# Do the screens predict what happened?", "",
         f"Written by `src/09_screens_vs_outcomes.py`. {len(df):,} overlap "
         "routes, each with a prediction computed from pre-merger information "
         "only and a realised fare effect measured against matched control "
         "routes over quarters "
         f"+5 to +12 after closing.", "",
         "## The structural presumption", "",
         "| | routes | average realised fare effect |", "|---|---|---|",
         f"| flagged by the presumption | {f['in_n']:,} | "
         f"{f['in_mean']*100:+.2f}% ({f['in_se']*100:.2f}) |",
         f"| not flagged | {f['out_n']:,} | "
         f"{f['out_mean']*100:+.2f}% ({f['out_se']*100:.2f}) |",
         f"| **difference** | | **{f['diff']*100:+.2f}%** "
         f"[{f['diff_lo']*100:+.2f}%, {f['diff_hi']*100:+.2f}%] |", ""]
    sep = "separates" if abs(f["diff"]) > 1.96 * f["diff_se"] else "does not separate"
    L += [f"On this measure the presumption **{sep}** the routes where fares "
          "rose from the routes where they did not.", ""]

    L += ["## How much information is in each prediction?", "",
          "Each row regresses the realised route-level effect on one "
          "prediction, with merger fixed effects and standard errors clustered "
          "on the route. The middle column is the change in the realised effect "
          "for a one-standard-deviation increase in the prediction, which puts "
          "predictors measured on different scales on the same footing.", "",
          "| prediction | realised effect per SD of prediction | rank "
          "correlation | slope | is the slope zero? | is it one? |",
          "|---|---|---|---|---|---|"]
    for col, s in RES["slopes"].items():
        lab = PREDICTORS[col][0]
        p1 = "-" if not np.isfinite(s.get("p_one", np.nan)) else f"p = {s['p_one']:.3f}"
        L.append(f"| {lab} | {s['slope_per_sd']*100:+.3f}% "
                 f"({s['se_per_sd']*100:.3f}) | {s['spearman']:+.3f} | "
                 f"{s['slope']:+.5f} | p = {s['p_zero']:.3f} | {p1} |")
    L += ["",
          "The slope column is only interpretable for the simulated fare "
          "change, where prediction and outcome are in the same units and a "
          "slope of one would mean the simulation was correctly scaled as well "
          "as informative.", ""]

    if "sim_dp_market" in RES["slopes"]:
        s = RES["slopes"]["sim_dp_market"]
        L += ["## Is the merger simulation calibrated?", "",
              f"> Slope of realised on simulated: **{s['slope']:+.3f}** "
              f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}].", "",
              f"Against the null that the simulation carries no information, "
              f"p = {s['p_zero']:.3f}. Against the null that it is correctly "
              f"scaled, p = {s['p_one']:.3f}.", ""]

    L += ["## Realised effect by decile of each prediction", "",
          "If a prediction works, the realised effect should climb across its "
          "deciles. Each cell is the average realised fare effect for the "
          "routes in that decile, with its standard error.", ""]
    for col, recs in RES["bins"].items():
        b = pd.DataFrame(recs)
        L += [f"### {PREDICTORS[col][0]}", "",
              "| decile | routes | prediction range | realised effect |",
              "|---|---|---|---|"]
        for _, r in b.iterrows():
            if col in ("delta_hhi",):
                rng = f"{r['pred_lo']:,.0f} to {r['pred_hi']:,.0f}"
            else:
                rng = f"{r['pred_lo']:.3f} to {r['pred_hi']:.3f}"
            L.append(f"| {int(r['bin'])} | {int(r['n']):,} | {rng} | "
                     f"{r['realised']*100:+.2f}% ({r['se']*100:.2f}) |")
        top, bot = b.iloc[-1], b.iloc[0]
        gap = top["realised"] - bot["realised"]
        gse = np.sqrt(top["se"] ** 2 + bot["se"] ** 2)
        L += ["",
              f"Top decile minus bottom decile: **{gap*100:+.2f}%** "
              f"[{(gap-1.96*gse)*100:+.2f}%, {(gap+1.96*gse)*100:+.2f}%].", ""]

    if rows:
        L += ["## Where should the threshold sit?", "",
              "The Guidelines put the line at an HHI increase of 100 points. "
              "This is what other lines would have done, on this data: how many "
              "routes each catches, how much the flagged and unflagged groups "
              "differ, and what share of the total realised fare increase falls "
              "inside the flagged group.", "",
              "| HHI increase above | routes flagged | share of overlap routes "
              "| flagged | not flagged | difference | share of realised "
              "increases captured |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            mark = " **(the Guidelines)**" if r["threshold"] == 100 else ""
            L.append(f"| {r['threshold']:,}{mark} | {r['flagged']:,} | "
                     f"{r['share_flagged']:.0%} | {r['mean_in']*100:+.2f}% | "
                     f"{r['mean_out']*100:+.2f}% | {r['diff']*100:+.2f}% "
                     f"({r['diff_se']*100:.2f}) | "
                     f"{r['share_of_increases_captured']:.0%} |")
        L.append("")

    lv = RES["level"]
    L += ["## Level, not just ordering", "",
          "| | simulated | realised |", "|---|---|---|",
          f"| median across routes | {lv['median_predicted']*100:+.2f}% | "
          f"{lv['median_realised']*100:+.2f}% |",
          f"| mean across routes | {lv['mean_predicted']*100:+.2f}% | "
          f"{lv['mean_realised']*100:+.2f}% |", "",
          "The simulated figures are conditional on the calibration in "
          "[`demand.md`](demand.md), so a gap in levels is as much a statement "
          "about that calibration as about the model. The ordering, which "
          "stage 8 showed is close to invariant to the calibration, is the part "
          "that carries weight.", ""]
    (C.REPORTS / "validation.md").write_text("\n".join(L), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
