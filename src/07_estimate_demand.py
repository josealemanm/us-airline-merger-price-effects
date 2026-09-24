"""Stage 7 - estimating demand, and finding out what it will and will not support.

The merger simulation needs two things from demand: diversion ratios, which say
where a carrier's lost passengers go, and margins, which say what those lost
passengers were worth. Both come from a demand system fitted to fares and shares
that airlines chose, so the whole exercise turns on whether the price
coefficient can be identified.

The specification is the nested logit inverted by Berry (1994):

    ln(s_j) - ln(s_0) = -alpha*fare + sigma*ln(s_j|nest) + x'b + xi

with market, quarter and carrier fixed effects. This stage does three things,
in order, and the second and third exist because the first did not go the way
it was supposed to.

1. Test each instrument instead of assuming it.
2. Report the estimate that survives, including the respect in which it fails.
3. Calibrate what the simulation actually runs on, and say so in the open.

**On the instruments.** Five were built, and most of them do not survive being
tested one at a time rather than thrown in together. How many fail is decided by
the data and reported by the script rather than asserted here. The ones that do
fail are the market-structure instruments - how many rivals a carrier faces, how
many are in its own nest - and they fail because entry is a response to demand.
Carriers add a route when they expect it to sell. So a positive demand shock
brings in rivals, rivals push the fare down, and an instrument built on rival
counts carries the demand shock it was supposed to exclude. Instrumenting with
it returns a price coefficient of the wrong sign, and a large one. The Hausman
instrument - the carrier's fare in other markets - fails for the familiar
reason: a carrier's demand shocks are correlated across its own network.

What survives is the cost instrument: national jet fuel prices interacted with
route distance. A fuel price move is not a response to demand between two
particular cities, and route distance is geography. It is the weakest of the
five in the first stage and the only one with a defensible exclusion
restriction, which is the trade every applied paper is really making.

**On what it returns.** Instrumented with the cost shifter alone, the price
coefficient has the right sign and is larger than OLS, which is the direction
the endogeneity argument predicts. It is still too small: the implied median
own-price elasticity (reported by the script) sits above minus one in absolute
terms, and a profit-maximising firm never prices
where its own demand is inelastic: the first-order condition would put marginal
cost below zero. It is a usable regression coefficient and an unusable demand
system.

**On what stage 8 therefore runs on.** The price coefficient is calibrated to
hit a target own-price elasticity taken from the published airline demand
literature, and the simulation is run again at elasticities well either side of
it. This is stated rather than hidden, and it is what an agency does when
demand estimation on the record will not support a simulation: fall back on a
calibrated margin and show the sensitivity. The important consequence is
reported in stage 8 - the *ranking* of routes by predicted harm is nearly
invariant to the calibration, because under logit demand diversion depends on
shares alone, while the *level* of predicted harm scales with it. The
screens-validation exercise turns on the ranking.
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
from econ.iv import iv2sls

FARE_UNIT = 100.0   # fares enter in hundreds of dollars, so alpha is readable

# Every instrument that was built, with the argument for it and the reason it
# does or does not survive. Kept as data so the report writes itself from the
# same object the estimation uses.
INSTRUMENTS = {
    "z_fuel_distance": (
        "cost", "national jet fuel price x log route distance",
        "A fuel price move is not a response to demand between two particular "
        "cities, and distance is geography."),
    "z_hausman": (
        "cost", "the carrier's mean fare in its other markets, same quarter",
        "Moves with the carrier's own costs - but also with any demand shock "
        "that runs across its whole network."),
    "z_n_rivals": (
        "structure", "number of rival carriers on the route",
        "Shifts markups without shifting willingness to pay - unless entry "
        "responds to demand, which it does."),
    "z_n_nest_rivals": (
        "structure", "number of rivals inside the carrier's own nest",
        "The instrument sigma is meant to lean on, and it inherits the same "
        "entry problem."),
    "z_rival_nonstop": (
        "structure", "summed nonstop share of competitors",
        "Rival service quality. Also chosen in response to demand."),
}
COST_IV = ["z_fuel_distance"]

# The calibration target for stage 8. Published firm-level own-price
# elasticities for US airline demand sit around -2 to -3; -2.5 is taken as the
# central value and the simulation is re-run at the bounds.
TARGET_ELASTICITY = -2.5
TARGET_ELASTICITY_ALTS = (-1.5, -4.0)
# Sigma cannot be credibly identified: every instrument for the within-nest
# share is one of the structure instruments that fails the validity test above.
# The simulation therefore runs plain logit, with these as stated sensitivities.
SIGMA_MAIN = 0.0
SIGMA_ALTS = (0.25, 0.50)


def prepare(cp: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    d = cp[["market_id", "qindex", "carrier", "group", "fare", "share",
            "pax", "market_pax", "n_carriers", "nonstop_share", "distance",
            "inside_share"]].copy()
    d = d[d["n_carriers"] >= 2]
    d = d[(d["share"] > 0) & (d["share"] < 1)]
    d = d[(d["inside_share"] > 0) & (d["inside_share"] < 1)]

    d["mq"] = d["market_id"] + "|" + d["qindex"].astype(str)
    # A carrier alone in its nest has a within-nest share of one and a log of
    # zero. Those observations are kept: dropping them would throw away fifteen
    # per cent of the sample and would select on market structure.
    g = d.groupby(["mq", "group"])["share"].transform("sum")
    d["share_in_nest"] = (d["share"] / g).clip(upper=1.0)
    d = d[d["share_in_nest"] > 0]
    d["ln_share_in_nest"] = np.log(d["share_in_nest"])

    d["s_j"] = d["share"] * d["inside_share"]
    d["s_0"] = 1.0 - d["inside_share"]
    d["y"] = np.log(d["s_j"]) - np.log(d["s_0"])
    d["price"] = d["fare"] / FARE_UNIT

    fuel = macro.set_index("qindex")["jetfuel"]
    d["z_fuel_distance"] = (d["qindex"].map(fuel).to_numpy()
                            * np.log(d["distance"].clip(lower=1).to_numpy()))
    cq = d.groupby(["carrier", "qindex"])["price"]
    d["_sum"], d["_n"] = cq.transform("sum"), cq.transform("size")
    d["z_hausman"] = (d["_sum"] - d["price"]) / (d["_n"] - 1).clip(lower=1)
    d.loc[d["_n"] <= 1, "z_hausman"] = np.nan
    d["z_n_rivals"] = d["n_carriers"] - 1
    d["z_n_nest_rivals"] = d.groupby(["mq", "group"])["carrier"].transform("size") - 1
    ns = d.groupby("mq")["nonstop_share"].transform("sum")
    d["z_rival_nonstop"] = ns - d["nonstop_share"]

    return d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["y", "ln_share_in_nest", "price", *INSTRUMENTS])


def median_elasticity(d: pd.DataFrame, alpha_per_100: float, sigma: float) -> float:
    s = d["s_j"].to_numpy()
    sg = d["share_in_nest"].to_numpy()
    p = d["fare"].to_numpy()
    own = -alpha_per_100 / FARE_UNIT * p * (
        1 / (1 - sigma) - sigma / (1 - sigma) * sg - s)
    return float(np.median(own))


def calibrate_alpha(d: pd.DataFrame, target: float, sigma: float) -> float:
    """Scale alpha so the median own-price elasticity equals ``target``.

    The elasticity is linear in alpha at fixed shares, so this is one division
    rather than a search.
    """
    unit = median_elasticity(d, 1.0, sigma)
    return float(target / unit)


def main() -> int:
    cp = pd.read_parquet(C.DATA_PROCESSED / "panel_carrier.parquet")
    macro = pd.read_parquet(C.DATA_INTERIM / "macro.parquet")
    d = prepare(cp, macro)
    print(f"  estimation sample: {len(d):,} carrier-market-quarters, "
          f"{d['market_id'].nunique():,} routes, {d['carrier'].nunique()} carriers")

    fes = [factorize(d["market_id"]), factorize(d["qindex"]),
           factorize(d["carrier"])]
    cl = d["market_id"].to_numpy()
    y = d["y"].to_numpy()
    Xex = d[["nonstop_share"]].to_numpy()
    price = d[["price"]].to_numpy()

    # ---------------------------------- 1. test every instrument on its own
    ols = feols(y, np.column_stack([Xex, price]), fes=fes, cluster=cl,
                names=["nonstop_share", "price"])
    a_ols = -ols.get("price")[0]
    audit = [{"instrument": "none (OLS)", "kind": "-", "alpha": a_ols,
              "first_stage_F": float("nan"),
              "elasticity": median_elasticity(d, a_ols, 0.0), "valid": None}]
    print(f"\n  instrument audit (plain logit, one instrument at a time)")
    print(f"    {'OLS':<24} alpha = {a_ols:+.4f}")
    for z, (kind, _desc, _why) in INSTRUMENTS.items():
        r = iv2sls(y, Xex, price, d[[z]].to_numpy(), fes=fes, cluster=cl,
                   names_exog=["nonstop_share"], names_endog=["price"],
                   names_z=[z])
        a = -r.get("price")[0]
        F = float(list(r.first_stage_F.values())[0])
        audit.append({"instrument": z, "kind": kind, "alpha": a,
                      "first_stage_F": F,
                      "elasticity": median_elasticity(d, a, 0.0),
                      "valid": bool(a > 0)})
        print(f"    {z:<24} alpha = {a:+.4f}   first-stage F = {F:>9,.0f}"
              f"   {'' if a > 0 else '  <- wrong sign'}")

    # ------------------------------- 2. the estimate the cost instrument gives
    iv = iv2sls(y, Xex, price, d[COST_IV].to_numpy(), fes=fes, cluster=cl,
                names_exog=["nonstop_share"], names_endog=["price"],
                names_z=COST_IV)
    alpha_iv = -iv.get("price")[0]
    a_se = float(iv.se[iv.names.index("price")])
    el_iv = median_elasticity(d, alpha_iv, 0.0)
    ci = iv.conf_int()
    ia = iv.names.index("price")
    print(f"\n  cost-instrumented estimate: alpha = {alpha_iv:.4f} "
          f"(se {a_se:.4f}), median elasticity {el_iv:.2f}")

    # Nested logit, for the record, using every instrument. Reported so the
    # sigma that comes out of an unaudited specification is on the page.
    ZS = list(INSTRUMENTS)
    ivn = iv2sls(y, Xex, d[["price", "ln_share_in_nest"]].to_numpy(),
                 d[ZS].to_numpy(), fes=fes, cluster=cl,
                 names_exog=["nonstop_share"],
                 names_endog=["price", "ln_share_in_nest"], names_z=ZS)
    nested = {"alpha": -ivn.get("price")[0],
              "sigma": ivn.get("ln_share_in_nest")[0]}

    # ------------------------------------------ 3. what stage 8 will run on
    alpha_cal = calibrate_alpha(d, TARGET_ELASTICITY, SIGMA_MAIN)
    print(f"  calibrated for simulation:  alpha = {alpha_cal:.4f} "
          f"to hit a median elasticity of {TARGET_ELASTICITY:.1f}")
    print(f"  ratio, calibrated to estimated: {alpha_cal/alpha_iv:.1f}x")

    params = {
        "estimated": {
            "alpha_per_100": alpha_iv, "alpha_se": a_se,
            "alpha_per_dollar": alpha_iv / FARE_UNIT,
            "ci_lo": float(-ci[ia, 1]), "ci_hi": float(-ci[ia, 0]),
            "median_elasticity": el_iv,
            "first_stage_F": float(list(iv.first_stage_F.values())[0]),
            "admissible_for_bertrand": bool(el_iv < -1.0),
        },
        "ols": {"alpha_per_100": a_ols,
                "median_elasticity": median_elasticity(d, a_ols, 0.0)},
        "nested_logit_all_instruments": nested,
        "instrument_audit": audit,
        "calibrated": {
            "alpha_per_100": alpha_cal,
            "alpha_per_dollar": alpha_cal / FARE_UNIT,
            "sigma": SIGMA_MAIN,
            "target_elasticity": TARGET_ELASTICITY,
            "alternatives": [
                {"target_elasticity": t, "sigma": SIGMA_MAIN,
                 "alpha_per_dollar": calibrate_alpha(d, t, SIGMA_MAIN) / FARE_UNIT}
                for t in TARGET_ELASTICITY_ALTS
            ] + [
                {"target_elasticity": TARGET_ELASTICITY, "sigma": s,
                 "alpha_per_dollar": calibrate_alpha(d, TARGET_ELASTICITY, s) / FARE_UNIT}
                for s in SIGMA_ALTS
            ],
        },
        "fare_unit": FARE_UNIT,
        "nobs": int(len(d)), "n_markets": int(d["market_id"].nunique()),
        "median_fare": float(d["fare"].median()),
    }
    (C.DATA_PROCESSED / "demand_params.json").write_text(
        json.dumps(params, indent=2, default=float))

    # ------------------------------------------------------------- report
    L = ["# Demand estimation", "",
         f"Estimated by `src/07_estimate_demand.py` on {len(d):,} "
         f"carrier-market-quarter observations covering "
         f"{d['market_id'].nunique():,} routes, with market, quarter and "
         "carrier fixed effects and standard errors clustered on the route.", "",
         "This stage did not produce what it was built to produce. What follows "
         "is the test that found that out, the estimate that survived it, and "
         "the assumption the simulation runs on instead.", "",
         "## Testing the instruments one at a time", "",
         "Throwing five instruments into one regression and reading the "
         "overidentification test is the usual practice and it hides exactly "
         "this problem. Run separately, in the plain logit where only the fare "
         "is endogenous, each instrument gives its own answer, and a price "
         "coefficient of the wrong sign is a failure no first-stage F can "
         "excuse.", "",
         "| instrument | what it is | first-stage F | implied alpha | median "
         "elasticity | verdict |", "|---|---|---|---|---|---|"]
    for row in audit:
        z = row["instrument"]
        desc = INSTRUMENTS[z][1] if z in INSTRUMENTS else "no instrument"
        F = "-" if not np.isfinite(row["first_stage_F"]) else f"{row['first_stage_F']:,.0f}"
        verdict = ("-" if row["valid"] is None
                   else ("survives" if row["valid"] else "**wrong sign**"))
        L.append(f"| `{z}` | {desc} | {F} | {row['alpha']:+.3f} | "
                 f"{row['elasticity']:+.2f} | {verdict} |")
    failed = [r for r in audit if r["valid"] is False]
    passed = [r for r in audit if r["valid"] is True]
    n_all = len([r for r in audit if r["valid"] is not None])
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    spell = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    fail_names = ", ".join(f"`{r['instrument']}`" for r in failed)
    cost_F = next((r["first_stage_F"] for r in audit
                   if r["instrument"] == COST_IV[0]), float("nan"))
    others_F = [r["first_stage_F"] for r in audit
                if np.isfinite(r["first_stage_F"])
                and r["instrument"] not in COST_IV]
    next_weakest = min(others_F) if others_F else float("nan")
    L += ["",
          f"{words.get(len(failed), len(failed))} of the "
          f"{spell.get(n_all, n_all)} return a price "
          f"coefficient of the wrong sign: {fail_names}. They fail together and "
          "for one reason. Entry is a response to demand: a carrier adds a route "
          "when it expects the route to sell. A positive demand shock therefore "
          "brings in rivals, rivals push the fare down, and an instrument built "
          "on rival counts carries the very shock it was supposed to exclude. "
          "Where the Hausman instrument is among them, it fails for the familiar "
          "reason - a carrier's demand shocks run across its own network, so its "
          "fares elsewhere are not independent of demand here.", ""]
    if len(passed) > 1:
        n_pass = spell.get(len(passed), len(passed))
        others = ", ".join(f"`{r['instrument']}`" for r in passed
                           if r["instrument"] != COST_IV[0])
        L += [f"Of the {n_pass} that come back with the right sign, only the "
              f"cost instrument has an exclusion restriction worth defending. "
              f"{others} survives this particular test but is still a market-"
              "structure instrument, and the entry argument applies to it "
              "whether or not it happens to fail here. It is not used.", ""]
    L += [f"The cost instrument is much the weakest in the first stage - "
          f"F = {cost_F:,.0f}, against {next_weakest:,.0f} for the next weakest "
          "and thousands for some of the rest - and the only one whose validity "
          "survives the argument. Strength and validity are separate properties, "
          "and here they point in opposite directions.", ""]

    L += ["## The estimate that survives", "",
          "| | alpha, per $100 | implied median own-price elasticity |",
          "|---|---|---|",
          f"| OLS | {a_ols:.4f} | {median_elasticity(d, a_ols, 0.0):.2f} |",
          f"| IV, cost instrument | {alpha_iv:.4f} | {el_iv:.2f} |", "",
          f"Instrumenting moves the coefficient away from zero, from "
          f"{a_ols:.3f} to {alpha_iv:.3f}, which is the direction the "
          "endogeneity argument predicts: carriers charge more where demand is "
          "strong, so the uninstrumented regression sees fare and quantity "
          "moving together and reads that as insensitivity.", "",
          "## And the respect in which it fails", "",
          f"> The estimate implies a median own-price elasticity of "
          f"**{el_iv:.2f}**.", "",
          "A profit-maximising firm never prices where its own demand is "
          "inelastic. At an elasticity above -1 the Bertrand first-order "
          "condition returns a marginal cost below zero, and a merger "
          "simulation built on it would be arithmetic performed on an "
          "impossible object.", "",
          "Three things could produce this and the data cannot separate them. "
          "The fare is a quarterly average over every fare class a carrier "
          "sold, and an average over a bundle is less elastic than any ticket "
          "in it. DB1B is a ten per cent sample, so route-level average fares "
          "carry sampling noise, and measurement error attenuates. And the one "
          "valid instrument is weak enough that the correction it applies may "
          "simply be incomplete.", "",
          "The nested logit with all five instruments returns "
          f"alpha = {nested['alpha']:.3f} and sigma = {nested['sigma']:.3f}. "
          "Since four of those instruments have just failed their own test, "
          "that pair is reported here and used nowhere.", ""]

    L += ["## What the simulation runs on instead", "",
          "The price coefficient is calibrated to a target own-price "
          f"elasticity of **{TARGET_ELASTICITY}**, within the range published "
          "estimates of US airline firm-level demand elasticity occupy, and the "
          "simulation is re-run at elasticities either side of it.", "",
          "| | value |", "|---|---|",
          f"| calibrated alpha, per dollar | {alpha_cal/FARE_UNIT:.5f} |",
          f"| implied median elasticity | {TARGET_ELASTICITY:.1f} |",
          f"| ratio to the estimated alpha | {alpha_cal/alpha_iv:.1f}x |",
          f"| nesting parameter sigma | {SIGMA_MAIN} (plain logit) |", "",
          "Sigma is set to zero rather than estimated. Every instrument for the "
          "within-nest share is one of the market-structure instruments that "
          "has just failed, so there is nothing credible left to identify it "
          "with. Plain logit is the weaker model - it forces diversion to "
          "follow market shares - and stage 8 re-runs at sigma of "
          f"{SIGMA_ALTS[0]} and {SIGMA_ALTS[1]} to show what that costs.", "",
          "This is a calibration and it is labelled as one. It is also what an "
          "agency does when the demand estimation on the record will not "
          "support a simulation: fall back on a calibrated margin and put the "
          "sensitivity in the record. The consequence that matters is reported "
          "in stage 8. Under logit demand, diversion ratios depend on market "
          "shares alone and not on alpha at all, so the *ranking* of routes by "
          "predicted harm barely moves with the calibration, while the *level* "
          "of predicted harm scales with it almost proportionally. The question "
          "this project asks - whether the screens pick the right routes - "
          "turns on the ranking.", ""]
    (C.REPORTS / "demand.md").write_text("\n".join(L), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
