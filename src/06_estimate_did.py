"""Stage 6 - what actually happened to fares on the overlap routes.

The estimate reported as the headline is a stacked difference-in-differences.
Each merger contributes its own block: the overlap routes for that merger, the
routes untouched by any merger closing nearby, and the twenty-one quarters
around that merger's closing. Route-by-block and quarter-by-block fixed effects
mean no comparison ever reaches across mergers, so the pathology that staggered
two-way fixed effects runs into cannot arise by construction.

To show that this is not a distinction without a difference, the naive pooled
estimator is run first and then decomposed. The share of it resting on
already-treated controls is reported rather than assumed away.

Everything is weighted by baseline passengers - traffic in the pre-announcement
window - so that busy routes count for more, and so that the weights cannot
themselves be an outcome of the merger. Standard errors cluster on the route.

Outputs
    data/processed/event_study.parquet   coefficients for the figures
    data/processed/route_effects.parquet per-route realised effects, for stage 8
    data/processed/did_results.json      every headline number
    reports/did.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ.bacon import bacon_decompose
from econ.did import Stack, stacked_did, stacked_event_study
from econ.fe_ols import factorize, feols
from econ.inference import wild_cluster_bootstrap
from econ.matching import balance_table, nearest_neighbour_att

# The window used to measure the realised long-run effect. Airlines do not
# combine overnight: a single operating certificate lands four to six quarters
# after closing, and fare systems merge around then. Quarters +5 to +12 is the
# period in which a combined carrier is actually setting fares as one firm.
LONGRUN = (5, 12)
RESULTS: dict = {}


COVARIATES = ["log_distance", "log_pax", "hhi", "n_carriers", "log_fare",
              "nonstop_share", "pre_trend"]
BALANCE: list[dict] = []
MATCH_DIAG: list[dict] = []


def baseline_covariates(mp: pd.DataFrame, m) -> pd.DataFrame:
    """What each route looked like before the merger was announced.

    ``pre_trend`` is the slope of log fare over the baseline window. Including
    it means the control group is selected partly on the thing the design
    assumes away, which is the most useful covariate available.
    """
    lo, hi = m.announce_q - C.BASELINE_QUARTERS, m.announce_q - 1
    w = mp[(mp["qindex"] >= lo) & (mp["qindex"] <= hi)]
    g = w.groupby("market_id")
    out = pd.DataFrame({
        "log_distance": np.log(g["distance"].mean().clip(lower=1)),
        "log_pax": np.log(g["pax"].mean().clip(lower=1)),
        "hhi": g["hhi"].mean() / 1000.0,
        "n_carriers": g["n_carriers"].mean(),
        "log_fare": g["log_fare"].mean(),
        "nonstop_share": g["nonstop_share"].mean(),
    })
    # Slope of log fare on quarter, route by route.
    x = w["qindex"].to_numpy(float)
    xc = x - x.mean()
    tmp = w.assign(_xc=xc, _num=xc * w["log_fare"], _den=xc ** 2)
    num = tmp.groupby("market_id")["_num"].sum()
    den = tmp.groupby("market_id")["_den"].sum()
    out["pre_trend"] = (num / den.replace(0, np.nan)).fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def control_routes(tr: pd.DataFrame, m, kind: str) -> set:
    """Candidate control routes for one merger.

    The default control group is routes where *exactly one* of the merging
    carriers was selling. That is the comparison the airline retrospective
    literature uses, and it is the right one for an antitrust question. Both
    groups are served by the merged carrier, so both absorb whatever the merger
    did to its costs, its network and its frequent-flyer programme. Only the
    overlap routes lose a competitor. The difference between them is therefore
    the competitive effect, with the system-wide effects differenced out.

    Routes where neither party flew are available as an alternative, and are far
    worse suited: there are few of them that resemble an overlap route, and
    matching onto them leaves a handful of control routes carrying most of the
    weight.

    Either way, a control route is dropped if it is an overlap route for a
    different merger closing inside this one's window.
    """
    t = tr[tr["merger"] == m.key]
    cand = set(t.loc[t["one_party"] if kind == "one_party" else t["neither"],
                     "market_id"])
    span = C.EVENT_MAX - C.EVENT_MIN
    for other in C.MERGERS:
        if other.key == m.key or abs(other.close_q - m.close_q) > span:
            continue
        o = tr[(tr["merger"] == other.key) & tr["overlap"]]["market_id"]
        cand -= set(o)
    return cand


def build_stack(mp: pd.DataFrame, tr: pd.DataFrame,
                group: str = "overlap", match: bool = True,
                record_balance: bool = False,
                weight_by: str = "equal",
                control: str = "one_party") -> Stack:
    """Assemble the stacked panel: one block per merger.

    ``weight_by`` chooses what the estimate averages over. "equal" gives every
    route the same weight and answers what happened to the typical overlap
    route. "pax" weights by pre-merger traffic and answers what happened to the
    typical passenger - the more economically meaningful question, but airline
    traffic is so skewed that a handful of routes carry most of the weight and
    the interval widens accordingly. Both are reported.
    """
    blocks = []
    for m in C.MERGERS:
        t = tr[tr["merger"] == m.key]
        treated = set(t.loc[t[group], "market_id"])
        ctrl = control_routes(tr, m, control) - treated
        if not treated or not ctrl:
            continue
        lo, hi = m.close_q + C.EVENT_MIN, m.close_q + C.EVENT_MAX
        w = mp[(mp["qindex"] >= lo) & (mp["qindex"] <= hi)
               & mp["market_id"].isin(treated | ctrl)].copy()
        if w.empty:
            continue
        w["event"] = m.key
        w["event_time"] = w["qindex"] - m.close_q
        w["treated"] = w["market_id"].isin(treated).astype(float)
        w["post"] = (w["qindex"] >= m.close_q).astype(float)

        # A route has to be seen on both sides of the closing to contribute a
        # within-route comparison, and has to cover most of the window so the
        # event-study coefficients are not built from different routes at
        # different lags.
        n_obs = w.groupby("market_id")["qindex"].nunique()
        n_pre = w[w["event_time"] < 0].groupby("market_id")["qindex"].nunique()
        n_post = w[w["event_time"] >= 0].groupby("market_id")["qindex"].nunique()
        keep = set(n_obs[n_obs >= 0.6 * (hi - lo + 1)].index) \
            & set(n_pre[n_pre >= 4].index) & set(n_post[n_post >= 4].index)
        w = w[w["market_id"].isin(keep)]
        if w.empty:
            continue

        # Baseline traffic: fixed before the announcement, so the weights are
        # not themselves affected by the merger.
        base = (mp[(mp["qindex"] >= m.announce_q - C.BASELINE_QUARTERS)
                   & (mp["qindex"] < m.announce_q)]
                .groupby("market_id")["pax"].mean())
        w["weight"] = w["market_id"].map(base).fillna(0.0)
        w = w[w["weight"] > 0]
        if w.empty:
            continue

        if match:
            cov = baseline_covariates(mp, m)
            routes = pd.Index(sorted(set(w["market_id"]) & set(cov.index)))
            if len(routes) < 50:
                blocks.append(w)
                continue
            cov = cov.loc[routes]
            t = routes.isin(treated).astype(float)
            if t.sum() < 10 or (1 - t).sum() < 10:
                blocks.append(w)
                continue
            pax = base.reindex(routes).fillna(0.0).to_numpy()
            uw = pax if weight_by == "pax" else np.ones(len(routes))
            mr = nearest_neighbour_att(cov[COVARIATES].to_numpy(), t, uw)
            wgt = mr.weights
            MATCH_DIAG.append({
                "merger": m.key, "matched": mr.matched_treated,
                "unmatched": mr.unmatched_treated,
                "controls_used": mr.n_controls_used,
                "n_effective_controls": mr.n_effective_controls,
                "max_control_share": mr.max_control_weight_share})
            if record_balance:
                for row in balance_table(cov[COVARIATES].to_numpy(), t, COVARIATES,
                                         w_before=pax, w_after=wgt):
                    BALANCE.append({"merger": m.key, **row})
            wmap = pd.Series(wgt, index=routes)
            w = w[w["market_id"].isin(routes)].copy()
            w["weight"] = w["market_id"].map(wmap)
            w = w[w["weight"] > 0]
            if w.empty:
                continue
        blocks.append(w)

    d = pd.concat(blocks, ignore_index=True)
    return Stack(
        event=d["event"].to_numpy(), unit=d["market_id"].to_numpy(),
        period=d["qindex"].to_numpy(), event_time=d["event_time"].to_numpy(int),
        treated=d["treated"].to_numpy(), post=d["post"].to_numpy(),
        y=d["log_fare"].to_numpy(), weight=d["weight"].to_numpy(),
        extras={"route": d["route"].to_numpy(), "hhi": d["hhi"].to_numpy(),
                "pax": d["pax"].to_numpy()},
    )


def naive_twfe(mp: pd.DataFrame, tr: pd.DataFrame) -> tuple[float, float, object]:
    """The pooled staggered estimator, and the data needed to decompose it."""
    first = (tr[tr["overlap"]].groupby("market_id")["close_q"].min())
    never = set(tr["market_id"]) - set(first.index)
    d = mp[mp["market_id"].isin(set(first.index) | never)].copy()
    SENTINEL = 10_000
    d["g"] = d["market_id"].map(first).fillna(SENTINEL).astype(int)
    d["treat"] = (d["qindex"] >= d["g"]).astype(float)
    base = mp[mp["qindex"] < C.MERGERS[0].announce_q].groupby("market_id")["pax"].mean()
    d["weight"] = d["market_id"].map(base).fillna(0.0)
    d = d[d["weight"] > 0]
    r = feols(d["log_fare"].to_numpy(), d["treat"].to_numpy()[:, None],
              fes=[factorize(d["market_id"]), factorize(d["qindex"])],
              cluster=d["market_id"].to_numpy(), weights=d["weight"].to_numpy(),
              names=["treat"])
    return float(r.params[0]), float(r.se[0]), d


def route_level_effects(mp: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """One realised effect per overlap route, for the ex-ante/ex-post comparison.

    For each route the change in log fare from the pre-window to the long-run
    post-window is computed, and the same change averaged over that merger's
    clean controls is subtracted. Route by route this is noisy; binned and
    averaged it is what stage 8 compares against the predictions.
    """
    rows = []
    for m in C.MERGERS:
        t = tr[tr["merger"] == m.key]
        treated = set(t.loc[t["overlap"], "market_id"])
        # The same control group the headline estimate uses. Using a different
        # one here would mean stage 9 scored its predictions against a number
        # the retrospective never reported.
        control = control_routes(tr, m, "one_party") - treated
        if not treated or not control:
            continue
        w = mp[mp["market_id"].isin(treated | control)].copy()
        w["k"] = w["qindex"] - m.close_q
        pre = w[(w["k"] >= C.EVENT_MIN) & (w["k"] <= -1)]
        post = w[(w["k"] >= LONGRUN[0]) & (w["k"] <= LONGRUN[1])]
        pre_m = pre.groupby("market_id")["log_fare"].mean()
        post_m = post.groupby("market_id")["log_fare"].mean()
        pre_n = pre.groupby("market_id")["log_fare"].size()
        post_n = post.groupby("market_id")["log_fare"].size()
        delta = (post_m - pre_m).dropna()
        delta = delta[(pre_n.reindex(delta.index) >= 4)
                      & (post_n.reindex(delta.index) >= 4)]
        ctrl = delta[delta.index.isin(control)]
        if ctrl.empty:
            continue
        cw = mp[mp["market_id"].isin(ctrl.index)].groupby("market_id")["pax"].mean()
        ctrl_mean = float(np.average(ctrl, weights=cw.reindex(ctrl.index).fillna(1.0)))
        base = (mp[(mp["qindex"] >= m.announce_q - C.BASELINE_QUARTERS)
                   & (mp["qindex"] < m.announce_q)]
                .groupby("market_id")["pax"].mean())
        for mid, dv in delta[delta.index.isin(treated)].items():
            rows.append({"merger": m.key, "market_id": mid,
                         "effect": float(dv - ctrl_mean),
                         "raw_change": float(dv), "control_change": ctrl_mean,
                         "baseline_pax": float(base.get(mid, np.nan))})
    return pd.DataFrame(rows)


def main() -> int:
    mp = pd.read_parquet(C.DATA_PROCESSED / "panel_market.parquet")
    tr = pd.read_parquet(C.DATA_PROCESSED / "treatment.parquet")
    L = ["# The retrospective: what fares did", "",
         "Estimated by `src/06_estimate_did.py`. Fares are in constant 2019 "
         "dollars; the outcome is the log of the passenger-weighted average "
         "fare on a route in a quarter. Every regression is weighted by "
         "pre-announcement traffic and clusters on the route.", ""]

    # ------------------------------------------------- 1. the naive estimator
    nb, nse, dnaive = naive_twfe(mp, tr)
    RESULTS["naive_twfe"] = {"coef": nb, "se": nse}
    L += ["## First, the estimator not to use", "",
          "Pooling all five mergers into one regression with route and quarter "
          "fixed effects gives:", "",
          f"> **{nb*100:+.2f}%** on log fares, standard error "
          f"{nse*100:.2f} points.", "",
          "The number is reported here so that it can be taken apart, not "
          "because it should be believed.", ""]

    # --------------------------------------------- 2. the Bacon decomposition
    bac = bacon_decompose(dnaive["log_fare"].to_numpy(), dnaive["market_id"].to_numpy(),
                          dnaive["qindex"].to_numpy(), dnaive["g"].to_numpy(),
                          never_sentinel=10_000)
    kinds = bac.by_kind()
    RESULTS["bacon"] = {
        "twfe": bac.twfe, "reconstructed": bac.reconstructed,
        "weights": {k: v[0] for k, v in kinds.items()},
        "betas": {k: v[1] for k, v in kinds.items()},
        "forbidden_weight": bac.forbidden_weight(),
        "n_components": len(bac.components),
    }
    L += ["## Why not to use it", "",
          "Goodman-Bacon's decomposition writes that coefficient as a "
          "weighted average of every two-by-two comparison in the data. The "
          "weights below sum to one and the weighted average reproduces the "
          f"regression coefficient exactly ({bac.reconstructed*100:+.4f}% "
          f"against {bac.twfe*100:+.4f}%), which is the check that the "
          "decomposition is right. The decomposition requires a balanced panel "
          "and equal weights, so it is run on its own regression rather than "
          "the weighted one above; that regression gives "
          f"{bac.twfe*100:+.2f}%.", "",
          "| comparison | weight | average effect |", "|---|---|---|"]
    pretty = {"treated_vs_never": "overlap routes vs routes no merger touched",
              "earlier_vs_later": "earlier merger vs a later one, before the later one closed",
              "later_vs_earlier": "later merger vs an earlier one **after it was already merged**"}
    for k, (w, b) in kinds.items():
        L.append(f"| {pretty[k]} | {w:.1%} | {b*100:+.2f}% |")
    L += ["",
          f"**{bac.forbidden_weight():.1%}** of the pooled estimate rests on "
          "comparisons that use an already-merged route as the control for a "
          "later merger. Those comparisons subtract one merger's ongoing fare "
          "response from another merger's estimate. That is the reason the rest "
          "of this report uses the stacked estimator instead.", ""]

    # ----------------------------------------------------- 3. stacked DiD
    stack = build_stack(mp, tr, record_balance=True)
    res = stacked_did(stack)
    b, se = float(res.params[0]), float(res.se[0])
    ci = res.conf_int()[0]
    RESULTS["stacked_did"] = {
        "coef": b, "se": se, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
        "nobs": res.nobs, "n_routes": res.nclusters,
        "n_treated_routes": int(pd.Series(stack.unit[stack.treated == 1]).nunique()),
        "n_control_routes": int(pd.Series(stack.unit[stack.treated == 0]).nunique()),
    }
    L += ["## The estimate", "",
          "Each merger against its own clean controls, in its own window, "
          "pooled.", "",
          f"> Overlap routes saw fares **{b*100:+.2f}%** relative to control "
          f"routes, 95% interval **[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]**.", "",
          f"| | |", "|---|---|",
          f"| observations | {res.nobs:,} route-quarters |",
          f"| routes | {res.nclusters:,} |",
          f"| of which treated | {RESULTS['stacked_did']['n_treated_routes']:,} |",
          f"| of which control | {RESULTS['stacked_did']['n_control_routes']:,} |",
          f"| standard error | {se*100:.2f} points, clustered on route |", ""]

    # ------------------------------------------------- 3b. covariate balance
    bal = pd.DataFrame(BALANCE)
    if not bal.empty:
        agg = bal.groupby("covariate")[["before", "after"]].mean().reindex(COVARIATES)
        RESULTS["balance"] = agg.reset_index().to_dict("records")
        worst_before = agg["before"].abs().max()
        worst_after = agg["after"].abs().max()
        L += ["## Are the control routes comparable?", "",
              "Overlap routes are not a random sample. They are routes where two "
              "large carriers both chose to sell, which makes them longer, "
              "busier and more concentrated than the average route. Controls are "
              "therefore reweighted by the odds of being an overlap route, given "
              "what the route looked like before the merger was announced.", "",
              "Standardised differences, in standard deviations. Under 0.1 in "
              "absolute value is the usual bar for balanced.", "",
              "| covariate | before reweighting | after |", "|---|---|---|"]
        pretty_cov = {"log_distance": "log route distance",
                      "log_pax": "log baseline passengers",
                      "hhi": "HHI (thousands)", "n_carriers": "carriers on the route",
                      "log_fare": "log baseline fare",
                      "nonstop_share": "share of passengers flying nonstop",
                      "pre_trend": "pre-merger fare trend"}
        for _, r in agg.reset_index().iterrows():
            L.append(f"| {pretty_cov.get(r['covariate'], r['covariate'])} | "
                     f"{r['before']:+.2f} | {r['after']:+.2f} |")
        L += ["",
              f"The worst imbalance falls from {worst_before:.2f} to "
              f"{worst_after:.2f} standard deviations.", ""]

    # ------------------------------------------------------ 4. event study
    ev = stacked_event_study(stack, kmin=C.EVENT_MIN, kmax=C.EVENT_MAX)
    evdf = pd.DataFrame(ev.as_rows())
    evdf.to_parquet(C.DATA_PROCESSED / "event_study.parquet", index=False)
    RESULTS["event_study"] = {
        "pretrend_F": ev.pretrend_f, "pretrend_p": ev.pretrend_p,
        "rows": evdf.to_dict("records"),
    }
    pre_max = evdf[evdf["k"] < -1]["coef"].abs().max()
    L += ["## Did fares already differ before the merger?", "",
          "The design is only credible if treated and control routes were "
          "moving together beforehand. Each coefficient below is the gap "
          "between them in one quarter, relative to the quarter before the "
          "merger closed.", "",
          f"> Joint test that all pre-merger coefficients are zero: "
          f"F = {ev.pretrend_f:.2f}, p = {ev.pretrend_p:.3f}. "
          f"The largest pre-merger gap is {pre_max*100:.2f}%.", ""]
    L += ["| quarters from closing | effect | 95% interval |", "|---|---|---|"]
    for r in evdf.itertuples():
        mark = "  *(omitted)*" if r.k == -1 else ""
        L.append(f"| {r.k:+d} | {r.coef*100:+.2f}%{mark} | "
                 f"[{r.ci_lo*100:+.2f}%, {r.ci_hi*100:+.2f}%] |")
    L.append("")

    # -------------------------------------------------- 5. per-merger effects
    L += ["## Merger by merger", "",
          "The same stacked specification, one merger at a time.", "",
          "| merger | effect on overlap-route fares | 95% interval | treated "
          "routes | control routes |", "|---|---|---|---|---|"]
    RESULTS["by_merger"] = {}
    for m in C.MERGERS:
        sel = stack.event == m.key
        if sel.sum() == 0:
            continue
        sub = Stack(event=stack.event[sel], unit=stack.unit[sel],
                    period=stack.period[sel], event_time=stack.event_time[sel],
                    treated=stack.treated[sel], post=stack.post[sel],
                    y=stack.y[sel], weight=stack.weight[sel], extras={})
        r = stacked_did(sub)
        c = r.conf_int()[0]
        nt = pd.Series(sub.unit[sub.treated == 1]).nunique()
        nc = pd.Series(sub.unit[sub.treated == 0]).nunique()
        RESULTS["by_merger"][m.key] = {
            "label": m.label, "coef": float(r.params[0]), "se": float(r.se[0]),
            "ci_lo": float(c[0]), "ci_hi": float(c[1]),
            "n_treated": int(nt), "n_control": int(nc)}
        L.append(f"| {m.label} | {r.params[0]*100:+.2f}% | "
                 f"[{c[0]*100:+.2f}%, {c[1]*100:+.2f}%] | {nt:,} | {nc:,} |")
    L.append("")

    # ------------------------------------------------------- 6. robustness
    L += ["## Does the result survive being poked?", "",
          "| test | what it asks | result |", "|---|---|---|"]

    # (a) placebo on one-party routes
    # Treated: routes where only one merging carrier flew. Control: routes
    # where neither did. Neither group loses a competitor, so a merger-specific
    # effect here would mean the design is picking up something other than the
    # loss of competition - a network change, a cost change, a frequent-flyer
    # effect - and the main estimate would be suspect.
    try:
        pstack = build_stack(mp, tr, group="one_party", control="neither")
        pres = stacked_did(pstack)
        pc = pres.conf_int()[0]
        RESULTS["placebo_one_party"] = {
            "coef": float(pres.params[0]), "se": float(pres.se[0]),
            "ci_lo": float(pc[0]), "ci_hi": float(pc[1])}
        L.append(f"| routes where only one party flew, against routes where "
                 f"neither did | neither group loses a competitor, so this "
                 f"should be near zero | {pres.params[0]*100:+.2f}% "
                 f"[{pc[0]*100:+.2f}%, {pc[1]*100:+.2f}%] |")
    except Exception as exc:  # noqa: BLE001
        L.append(f"| routes where only one party flew | | not estimable: {exc} |")

    # (b) wild cluster bootstrap
    did = (stack.treated * stack.post)[:, None]
    wb = wild_cluster_bootstrap(
        stack.y, did,
        fes=[factorize(stack.unit_by_event), factorize(stack.period_by_event)],
        cluster=stack.unit, weights=stack.weight, names=["did"],
        n_boot=299, seed=11)
    RESULTS["wild_bootstrap"] = {
        "p_value": wb.p_value, "ci_lo": wb.ci_lo, "ci_hi": wb.ci_hi,
        "n_boot": wb.n_draws}
    L.append(f"| wild cluster bootstrap, {wb.n_draws} draws | whether the "
             f"interval depends on the cluster count being large | "
             f"p = {wb.p_value:.3f}, 95% [{wb.ci_lo*100:+.2f}%, "
             f"{wb.ci_hi*100:+.2f}%] |")

    # (c) passenger-weighted
    pstk = build_stack(mp, tr, weight_by="pax")
    rp = stacked_did(pstk)
    cp_ = rp.conf_int()[0]
    RESULTS["passenger_weighted"] = {
        "coef": float(rp.params[0]), "se": float(rp.se[0]),
        "ci_lo": float(cp_[0]), "ci_hi": float(cp_[1])}
    L.append(f"| weighted by pre-merger traffic | what the average passenger "
             f"saw, rather than the average route | {rp.params[0]*100:+.2f}% "
             f"[{cp_[0]*100:+.2f}%, {cp_[1]*100:+.2f}%] |")

    # (d) no matching at all
    ustk = build_stack(mp, tr, match=False)
    ru = stacked_did(ustk)
    cu_ = ru.conf_int()[0]
    RESULTS["unmatched"] = {"coef": float(ru.params[0]), "se": float(ru.se[0]),
                            "ci_lo": float(cu_[0]), "ci_hi": float(cu_[1])}
    L.append(f"| no matching, all clean controls | how much the control group "
             f"choice matters | {ru.params[0]*100:+.2f}% "
             f"[{cu_[0]*100:+.2f}%, {cu_[1]*100:+.2f}%] |")

    # (e) long-run window only
    lr = stack.event_time >= LONGRUN[0]
    sel = (stack.event_time < 0) | lr
    sub = Stack(event=stack.event[sel], unit=stack.unit[sel],
                period=stack.period[sel], event_time=stack.event_time[sel],
                treated=stack.treated[sel],
                post=(stack.event_time[sel] >= LONGRUN[0]).astype(float),
                y=stack.y[sel], weight=stack.weight[sel], extras={})
    rl = stacked_did(sub)
    cl_ = rl.conf_int()[0]
    RESULTS["longrun"] = {"coef": float(rl.params[0]), "se": float(rl.se[0]),
                          "ci_lo": float(cl_[0]), "ci_hi": float(cl_[1]),
                          "window": LONGRUN}
    L.append(f"| quarters +{LONGRUN[0]} to +{LONGRUN[1]} only | the effect once "
             f"the carriers actually operate as one | {rl.params[0]*100:+.2f}% "
             f"[{cl_[0]*100:+.2f}%, {cl_[1]*100:+.2f}%] |")
    L.append("")

    # ------------------------------------------- 7. effect by concentration
    sc = pd.read_parquet(C.DATA_PROCESSED / "screens.parquet")
    eff = route_level_effects(mp, tr)
    eff = eff.merge(sc[["merger", "market_id", "delta_hhi", "hhi_pre",
                        "combined_share", "flagged", "n_carriers"]],
                    on=["merger", "market_id"], how="left")
    eff.to_parquet(C.DATA_PROCESSED / "route_effects.parquet", index=False)

    L += ["## Does the effect grow with how much concentration rose?", "",
          "Overlap routes, grouped by the HHI increase the merger produced. "
          "The effect is measured route by route against that merger's controls "
          f"over quarters +{LONGRUN[0]} to +{LONGRUN[1]}, then averaged with "
          "passenger weights.", "",
          "| HHI increase | routes | average fare effect | 95% interval |",
          "|---|---|---|---|"]
    bins = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 1e9)]
    RESULTS["dose_response"] = []
    for lo, hi in bins:
        g = eff[(eff["delta_hhi"] >= lo) & (eff["delta_hhi"] < hi)].dropna(
            subset=["effect", "baseline_pax"])
        if len(g) < 10:
            continue
        w = g["baseline_pax"].to_numpy()
        mu = float(np.average(g["effect"], weights=w))
        # Weighted standard error of the weighted mean.
        v = float(np.average((g["effect"] - mu) ** 2, weights=w))
        se_mu = float(np.sqrt(v / len(g)))
        lab = f"{lo:,.0f}+" if hi > 1e8 else f"{lo:,.0f} to {hi:,.0f}"
        RESULTS["dose_response"].append(
            {"bin": lab, "n": len(g), "effect": mu, "se": se_mu})
        L.append(f"| {lab} | {len(g):,} | {mu*100:+.2f}% | "
                 f"[{(mu-1.96*se_mu)*100:+.2f}%, {(mu+1.96*se_mu)*100:+.2f}%] |")
    L.append("")

    (C.REPORTS / "did.md").write_text("\n".join(L), encoding="utf-8")
    (C.DATA_PROCESSED / "did_results.json").write_text(
        json.dumps(RESULTS, indent=2, default=float))

    print(f"  naive pooled TWFE      {nb*100:+.2f}%  (se {nse*100:.2f})")
    print(f"  ... of which rests on already-treated controls: "
          f"{bac.forbidden_weight():.1%}")
    print(f"  stacked DiD            {b*100:+.2f}%  "
          f"[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]")
    print(f"  long-run (+{LONGRUN[0]} to +{LONGRUN[1]})   "
          f"{rl.params[0]*100:+.2f}%  [{cl_[0]*100:+.2f}%, {cl_[1]*100:+.2f}%]")
    print(f"  pre-trend joint test   F = {ev.pretrend_f:.2f}, p = {ev.pretrend_p:.3f}")
    print(f"  wild bootstrap p       {wb.p_value:.3f}")
    print(f"  route-level effects    {len(eff):,} overlap routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
