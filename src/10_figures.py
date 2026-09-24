"""Stage 10 - the figures.

Each one is built to answer a single question, and the colour is doing a job
rather than decorating. The rule used throughout: an estimate whose confidence
interval covers zero is drawn in grey. Several of these charts have a minority of
bars that clear zero, and colouring them all alike would invite a reader to take
a ranking off the page that the data does not support.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ import style as S

S.apply()
FIG = C.FIGURES
FIG.mkdir(parents=True, exist_ok=True)
MADE: list[str] = []


def save(fig, name: str) -> None:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    MADE.append(name)
    print(f"  {name}.png")


def figure(fn):
    """Run a figure, and let the rest of the set survive if one input is absent."""
    def wrapper():
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  skipped {fn.__name__}: missing input ({exc.filename})")
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {fn.__name__}: {type(exc).__name__}: {exc}")
    return wrapper


# ------------------------------------------------------------ 1. event study
@figure
def event_study():
    ev = pd.read_parquet(C.DATA_PROCESSED / "event_study.parquet").sort_values("k")
    res = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    k = ev["k"].to_numpy()
    c = ev["coef"].to_numpy() * 100
    lo, hi = ev["ci_lo"].to_numpy() * 100, ev["ci_hi"].to_numpy() * 100
    cols = S.significant_colors(lo, hi)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    S.hgrid(ax)
    ax.axhline(0, color=S.ZERO, lw=1.2, zorder=2)
    ax.axvline(-0.5, color=S.INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    # The window in which the carriers actually start operating as one firm.
    ax.axvspan(4.5, k.max() + 0.5, color=S.SEQ[0], alpha=0.45, zorder=1, lw=0)

    for x, l, h, col in zip(k, lo, hi, cols):
        ax.plot([x, x], [l, h], color=col, lw=2.0, solid_capstyle="round",
                zorder=3, alpha=0.95)
    ax.scatter(k, c, s=46, color=cols, zorder=4, edgecolor=S.SURFACE, linewidth=1.4)

    ax.annotate("merger closes", xy=(-0.5, ax.get_ylim()[1]),
                xytext=(-0.3, ax.get_ylim()[1] * 0.97), fontsize=9,
                color=S.INK_2, va="top")
    ax.annotate("single operating certificate,\nfare systems merge",
                xy=(5, ax.get_ylim()[0] * 0.82), fontsize=9, color=S.INK_2,
                va="bottom")

    S.subtitle(ax, "Fares on overlap routes rise, but only after the airlines "
                   "actually merge",
               "Effect on the average fare of an overlap route relative to "
               "matched control routes, by quarter from closing.\n"
               "Grey where the 95% interval covers zero. The quarter before "
               "closing is the omitted baseline.")
    ax.set_xlabel("quarters from the merger closing")
    ax.set_ylabel("fare effect")
    S.pct(ax)
    ax.set_xticks(range(int(k.min()), int(k.max()) + 1, 2))
    lr = res.get("longrun", {})
    S.source(fig, f"DOT DB1B, {C.YEAR_START}-{C.YEAR_END}. Stacked "
                  f"difference-in-differences, {res['stacked_did']['n_routes']:,} "
                  f"routes, standard errors clustered on the route. "
                  f"Average over quarters +5 to +12: {lr.get('coef',0)*100:+.2f}%.")
    save(fig, "event_study")


# ----------------------------------------------------- 2. Bacon decomposition
@figure
def bacon():
    res = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    b = res["bacon"]
    labels = {"treated_vs_never": "overlap routes vs routes\nno merger touched",
              "earlier_vs_later": "earlier merger vs a later one,\nbefore the later one closed",
              "later_vs_earlier": "later merger vs an earlier one,\nalready merged"}
    colours = {"treated_vs_never": S.BLUE, "earlier_vs_later": S.AQUA,
               "later_vs_earlier": S.ORANGE}
    keys = list(labels)
    w = [b["weights"][k] * 100 for k in keys]
    beta = [b["betas"][k] * 100 for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), width_ratios=[1, 1.15])
    ax = axes[0]
    S.vgrid(ax)
    y = np.arange(len(keys))[::-1]
    ax.barh(y, w, height=0.52, color=[colours[k] for k in keys], zorder=3)
    for yy, ww in zip(y, w):
        ax.text(ww + 1.2, yy, f"{ww:.1f}%", va="center", fontsize=9.5,
                color=S.INK_2)
    ax.set_yticks(y, [labels[k] for k in keys], fontsize=9)
    ax.set_xlim(0, max(w) * 1.25)
    ax.set_xlabel("share of the naive estimate")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("What the naive estimator is averaging", loc="left", fontsize=11.5)

    ax = axes[1]
    S.vgrid(ax)
    ax.axvline(0, color=S.ZERO, lw=1.2, zorder=2)
    ax.barh(y, beta, height=0.52, color=[colours[k] for k in keys], zorder=3)
    for yy, bb in zip(y, beta):
        # Labels sit on the far side of the bar from zero, so they never land on
        # the dashed line marking the naive estimate.
        off = 0.12 if bb >= 0 else -0.12
        ax.text(bb + off, yy + 0.38, f"{bb:+.2f}%", va="bottom",
                ha="left" if bb >= 0 else "right", fontsize=9.5, color=S.INK_2)
    ax.set_yticks(y, ["" for _ in keys])
    ax.axvline(b["twfe"] * 100, color=S.INK_2, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f"the naive estimate,\n{b['twfe']*100:+.2f}%",
                xy=(b["twfe"] * 100, len(keys) - 0.6), fontsize=9,
                color=S.INK_2, ha="center", va="top")
    ax.set_xlabel("fare effect from that comparison")
    ax.set_title("And what each part says", loc="left", fontsize=11.5)
    S.pct(ax, "x")
    lim = max(abs(min(beta)), abs(max(beta))) * 1.6
    ax.set_xlim(-lim, lim)

    fig.suptitle("A staggered two-way fixed effects estimate is an average of "
                 "comparisons, and one of them is backwards",
                 x=0.005, ha="left", fontsize=13, fontweight="semibold", y=1.14)
    fig.text(0.005, 1.005, f"Goodman-Bacon decomposition. "
             f"{b['forbidden_weight']*100:.1f}% of the estimate uses routes "
             "already merged by an earlier deal as the control group for a "
             "later one.", ha="left", va="bottom", fontsize=9.5, color=S.INK_2)
    save(fig, "bacon")


# -------------------------------------------------------- 3. by merger
@figure
def by_merger():
    res = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    rows = list(res["by_merger"].values())
    rows.sort(key=lambda r: r["coef"])
    lab = [r["label"] for r in rows]
    c = np.array([r["coef"] for r in rows]) * 100
    lo = np.array([r["ci_lo"] for r in rows]) * 100
    hi = np.array([r["ci_hi"] for r in rows]) * 100
    n = [r["n_treated"] for r in rows]
    cols = S.significant_colors(lo, hi)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    S.vgrid(ax)
    y = np.arange(len(rows))
    ax.axvline(0, color=S.ZERO, lw=1.2, zorder=2)
    for yy, l, h, col in zip(y, lo, hi, cols):
        ax.plot([l, h], [yy, yy], color=col, lw=2.4, solid_capstyle="round",
                zorder=3)
    ax.scatter(c, y, s=64, color=cols, zorder=4, edgecolor=S.SURFACE, linewidth=1.5)
    for yy, cc, nn in zip(y, c, n):
        ax.text(ax.get_xlim()[1], yy, f"  {nn:,} routes", va="center",
                fontsize=8.5, color=S.MUTED)
    ax.set_yticks(y, lab)
    S.pct(ax, "x")
    ax.set_xlabel("effect on overlap-route fares")
    S.subtitle(ax, "Four of the five mergers raised fares where the carriers "
                   "overlapped",
               "Stacked difference-in-differences, each merger against its own "
               "matched controls. Grey where the interval covers zero.")
    S.source(fig, "DOT DB1B. Alaska / Virgin America had only 16 overlap routes "
                  "that met the sample requirements, which is why its interval "
                  "is so wide.")
    save(fig, "by_merger")


# ------------------------------------------------------------- 4. balance
@figure
def balance():
    res = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    rows = res.get("balance")
    if not rows:
        raise ValueError("no balance table in did_results.json")
    pretty = {"log_distance": "route distance", "log_pax": "baseline passengers",
              "hhi": "concentration (HHI)", "n_carriers": "carriers on the route",
              "log_fare": "baseline fare",
              "nonstop_share": "share flying nonstop",
              "pre_trend": "pre-merger fare trend"}
    rows = sorted(rows, key=lambda r: abs(r["before"]))
    lab = [pretty.get(r["covariate"], r["covariate"]) for r in rows]
    before = [r["before"] for r in rows]
    after = [r["after"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    S.vgrid(ax)
    y = np.arange(len(rows))
    ax.axvline(0, color=S.ZERO, lw=1.2, zorder=2)
    for t in (-0.1, 0.1):
        ax.axvline(t, color=S.MUTED, lw=0.9, ls=(0, (2, 3)), zorder=2)
    for yy, b, a in zip(y, before, after):
        ax.plot([b, a], [yy, yy], color=S.GRID, lw=2.4, zorder=3,
                solid_capstyle="round")
    ax.scatter(before, y, s=58, color=S.MUTED, zorder=4, label="before matching",
               edgecolor=S.SURFACE, linewidth=1.2)
    ax.scatter(after, y, s=58, color=S.BLUE, zorder=5, label="after matching",
               edgecolor=S.SURFACE, linewidth=1.2)
    ax.set_yticks(y, lab)
    ax.set_xlabel("standardised difference, treated minus control (SDs)")
    ax.legend(loc="lower right", ncols=2)
    S.subtitle(ax, "Matching makes the control routes look like the overlap routes",
               "Dotted lines mark plus and minus 0.1 standard deviations, the "
               "usual bar for calling a covariate balanced.")
    save(fig, "balance")


# ------------------------------------------- 5. the screens against the pressure
@figure
def screens_vs_pressure():
    sim = pd.read_parquet(C.DATA_PROCESSED / "simulation.parquet")
    sc = pd.read_parquet(C.DATA_PROCESSED / "screens.parquet")
    d = sim.merge(sc[["merger", "market_id", "delta_hhi", "flagged"]],
                  on=["merger", "market_id"])
    d = d[(d["delta_hhi"] > 0) & (d["guppi_wavg"] > 0)]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    S.hgrid(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(d["delta_hhi"], d["guppi_wavg"] * 100, s=7, alpha=0.18,
               color=S.BLUE, linewidth=0, zorder=3)
    ax.axvline(C.DELTA_HHI_THRESHOLD, color=S.ZERO, lw=1.4, zorder=4)
    ax.axhline(5, color=S.ORANGE, lw=1.4, ls=(0, (4, 3)), zorder=4)

    flagged = d["flagged"].mean()
    pressure = (d["guppi_wavg"] > 0.05).mean()
    ax.annotate(f"the Guidelines threshold:\nHHI increase above "
                f"{C.DELTA_HHI_THRESHOLD}\nflags {flagged:.0%} of overlap routes",
                xy=(C.DELTA_HHI_THRESHOLD, 0.02), xytext=(115, 0.02),
                fontsize=9, color=S.ZERO, va="bottom")
    ax.annotate(f"GUPPI above 5%\nreaches {pressure:.0%}",
                xy=(1.3, 5.6), fontsize=9, color=S.ORANGE, va="bottom")
    ax.set_xlabel("increase in HHI from the merger (points, log scale)")
    ax.set_ylabel("GUPPI (%, log scale)")
    S.subtitle(ax, "The structural screen flags six times as many routes as the "
                   "pricing-pressure analysis",
               "Each dot is one overlap route. Both measures use only "
               "pre-merger information, and they disagree about which routes "
               "deserve attention.")
    S.source(fig, f"{len(d):,} overlap routes across five mergers. GUPPI is "
                  "conditional on the demand calibration in reports/demand.md.")
    save(fig, "screens_vs_pressure")


# --------------------------------------------------------- 6. calibration
@figure
def calibration():
    b = pd.read_parquet(C.DATA_PROCESSED / "calibration_bins.parquet")
    want = [("delta_hhi", "HHI increase"),
            ("guppi_wavg", "GUPPI"),
            ("sim_dp_market", "simulated fare change")]
    want = [(k, t) for k, t in want if k in set(b["predictor"])]
    if not want:
        raise ValueError("no calibration bins")
    fig, axes = plt.subplots(1, len(want), figsize=(4.3 * len(want), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, want):
        g = b[b["predictor"] == key].sort_values("bin")
        x = g["bin"].to_numpy()
        y = g["realised"].to_numpy() * 100
        se = g["se"].to_numpy() * 100
        lo, hi = y - 1.96 * se, y + 1.96 * se
        cols = S.significant_colors(lo, hi)
        S.hgrid(ax)
        ax.axhline(0, color=S.ZERO, lw=1.2, zorder=2)
        for xx, l, h, c in zip(x, lo, hi, cols):
            ax.plot([xx, xx], [l, h], color=c, lw=2.0, zorder=3,
                    solid_capstyle="round")
        ax.scatter(x, y, s=40, color=cols, zorder=4, edgecolor=S.SURFACE,
                   linewidth=1.2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("decile of the prediction")
        ax.set_xticks([1, 5, 10])
    axes[0].set_ylabel("realised fare effect")
    S.pct(axes[0])
    fig.suptitle("Do routes the tools rank as worse actually see bigger fare "
                 "increases?", x=0.005, ha="left", fontsize=13,
                 fontweight="semibold", y=1.13)
    fig.text(0.005, 1.005, "Overlap routes sorted into deciles by each "
             "prediction, with the average realised fare effect in each. Grey "
             "where the interval covers zero.",
             ha="left", va="bottom", fontsize=9.5, color=S.INK_2)
    save(fig, "calibration")


# --------------------------------------------------------- 7. instruments
@figure
def instruments():
    p = json.loads((C.DATA_PROCESSED / "demand_params.json").read_text())
    rows = [r for r in p["instrument_audit"] if r["instrument"] != "none (OLS)"]
    pretty = {"z_fuel_distance": "jet fuel price\nx route distance",
              "z_hausman": "carrier's fare in\nits other markets",
              "z_n_rivals": "number of rivals",
              "z_n_nest_rivals": "rivals in the\nsame nest",
              "z_rival_nonstop": "rivals' nonstop\nservice"}
    rows.sort(key=lambda r: r["alpha"])
    lab = [pretty.get(r["instrument"], r["instrument"]) for r in rows]
    a = [r["alpha"] for r in rows]
    cols = [S.BLUE if v > 0 else S.ORANGE for v in a]

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    S.vgrid(ax)
    y = np.arange(len(rows))
    ax.axvline(0, color=S.ZERO, lw=1.4, zorder=3)
    ax.barh(y, a, height=0.5, color=cols, zorder=3)
    for yy, v, r in zip(y, a, rows):
        off = 0.06 if v >= 0 else -0.06
        ax.text(v + off, yy, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=S.INK_2)
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_xlabel("implied price sensitivity (alpha)")
    # Room on the left so the longest negative label does not sit on the axis.
    lo, hi = min(a), max(a)
    span = hi - lo
    ax.set_xlim(lo - 0.22 * span, hi + 0.18 * span)
    ols = p["ols"]["alpha_per_100"]
    ax.axvline(ols, color=S.INK_2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f"OLS, {ols:+.2f}", xy=(ols, -0.75), fontsize=9,
                color=S.INK_2, ha="center", va="top", annotation_clip=False)
    n_bad = sum(1 for r in rows if r["alpha"] <= 0)
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    spell = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    S.subtitle(ax, f"{words.get(n_bad, n_bad)} of the "
                   f"{spell.get(len(rows), len(rows))} instruments "
                   "return a price coefficient of the wrong sign",
               "A negative alpha says passengers buy more as the fare rises. "
               "Only the cost instrument has an exclusion restriction worth "
               "defending; the market-structure instruments are invalid here "
               "because entry responds to demand.")
    S.source(fig, "Plain logit, one instrument at a time, market, quarter and "
                  "carrier fixed effects. A large first-stage F does not rescue "
                  "an instrument that fails this test.")
    save(fig, "instruments")


# --------------------------------------------------------- 8. fares in context
@figure
def fares_over_time():
    mp = pd.read_parquet(C.DATA_PROCESSED / "panel_market.parquet")
    g = (mp.assign(fp=mp["fare"] * mp["pax"])
         .groupby("qindex").agg(fp=("fp", "sum"), pax=("pax", "sum"),
                                hhi=("hhi", "median")))
    g["fare"] = g["fp"] / g["pax"]
    yrs = C.YEAR_START + (g.index - C.qindex(C.YEAR_START, 1)) / 4

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    S.hgrid(ax)
    for m in C.MERGERS:
        x = C.YEAR_START + (m.close_q - C.qindex(C.YEAR_START, 1)) / 4
        ax.axvline(x, color=S.MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
        ax.annotate(m.key.replace("_", "/"), xy=(x, ax.get_ylim()[1]),
                    xytext=(x + 0.08, 0.97), textcoords=("data", "axes fraction"),
                    fontsize=8, color=S.MUTED, rotation=90, va="top")
    ax.plot(yrs, g["fare"], color=S.BLUE, zorder=4)
    ax.set_ylabel("average fare, 2019 dollars")
    ax.set_xlabel("")
    S.subtitle(ax, "Real domestic fares over the study window",
               "Passenger-weighted average across every route in the sample, in "
               "constant 2019 dollars. Dashed lines mark the five merger "
               "closings.")
    S.source(fig, f"DOT DB1B Market file, {C.YEAR_START}Q1-{C.YEAR_END}Q4, "
                  f"{len(mp):,} route-quarters. Deflated by CPI-U.")
    save(fig, "fares_over_time")


def main() -> int:
    for f in (event_study, bacon, by_merger, balance, screens_vs_pressure,
              calibration, instruments, fares_over_time):
        f()
    print(f"\n  {len(MADE)} figures written to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
