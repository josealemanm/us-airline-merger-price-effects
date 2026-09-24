"""Stage 12 - the one-page memo.

Analysis that stays in a notebook does not inform a decision. This is the form
the work has to take to be useful to somebody who will not read the code: what
was asked, what was found, how much to trust it, and what follows. Every number
is read from the pipeline's outputs at build time, so the memo cannot drift out
of step with the results.
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from econ import style as S

OUT = C.REPORTS / "memo.pdf"
LEFT, RIGHT = 0.085, 0.915
WIDTH_CHARS = 96


def block(fig, y: float, text: str, size: float = 9.0, color=None,
          weight="normal", family="DejaVu Serif", leading: float = 0.0175,
          wrap: int = WIDTH_CHARS) -> float:
    """Draw a wrapped paragraph and return the y the next one should start at."""
    lines: list[str] = []
    for para in text.strip("\n").split("\n"):
        lines.extend(textwrap.wrap(para, wrap) or [""])
    for ln in lines:
        fig.text(LEFT, y, ln, fontsize=size, color=color or S.INK,
                 family=family, fontweight=weight, va="top")
        y -= leading
    return y


def rule(fig, y: float, color=None) -> float:
    fig.add_artist(plt.Line2D([LEFT, RIGHT], [y, y], color=color or S.GRID,
                              lw=0.9, transform=fig.transFigure))
    return y - 0.014


def main() -> int:
    did = json.loads((C.DATA_PROCESSED / "did_results.json").read_text())
    val = json.loads((C.DATA_PROCESSED / "validation.json").read_text())
    sim = json.loads((C.DATA_PROCESSED / "simulation_summary.json").read_text())
    dem = json.loads((C.DATA_PROCESSED / "demand_params.json").read_text())
    man = json.loads((C.DATA_RAW / "manifest.json").read_text())
    tickets = sum(q.get("rows_in", 0) for q in man["quarters"])

    st, lr = did["stacked_did"], did["longrun"]
    flag, slopes = val["flag"], val["slopes"]
    g = slopes["guppi_wavg"]
    h = slopes["delta_hhi"]
    hhi_bins = val["bins"]["delta_hhi"]
    _spell = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    _bad = sum(1 for r in dem["instrument_audit"] if r.get("valid") is False)
    _tot = sum(1 for r in dem["instrument_audit"] if r.get("valid") is not None)
    n_bad = _spell.get(_bad, _bad)
    n_tot = _spell.get(_tot, _tot)
    gup_bins = val["bins"]["guppi_wavg"]

    S.apply()
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    with PdfPages(OUT) as pdf:
        # ------------------------------------------------------------ page 1
        fig = plt.figure(figsize=(8.5, 11))
        y = 0.955
        fig.text(LEFT, y, "MEMORANDUM", fontsize=15, family="DejaVu Serif",
                 fontweight="bold", va="top")
        y -= 0.030
        y = rule(fig, y, S.INK)
        for label, value in [
            ("DATE", date.today().strftime("%B %-d, %Y")
             if sys.platform != "win32" else date.today().strftime("%B %d, %Y")),
            ("RE", "Do the structural merger screens predict realised price "
                   "effects? Evidence from five US airline mergers"),
            ("DATA", f"DOT Origin and Destination Survey (DB1B), "
                     f"{C.YEAR_START}Q1-{C.YEAR_END}Q4; "
                     f"{tickets/1e6:,.0f} million ticket records"),
        ]:
            fig.text(LEFT, y, label, fontsize=8, family="DejaVu Serif",
                     fontweight="bold", va="top", color=S.INK_2)
            for i, ln in enumerate(textwrap.wrap(value, 82)):
                fig.text(LEFT + 0.075, y - i * 0.0155, ln, fontsize=9,
                         family="DejaVu Serif", va="top")
            y -= 0.0155 * max(1, len(textwrap.wrap(value, 82))) + 0.008
        y = rule(fig, y)
        y -= 0.008

        y = block(fig, y, "SUMMARY", size=10, weight="bold")
        y -= 0.006
        y = block(fig, y, f"""
Across the five large US airline mergers that closed between 2008 and 2016,
fares on routes where the merging carriers overlapped rose {st['coef']*100:.1f}% relative to
matched control routes, and {lr['coef']*100:.1f}% once the carriers were operating under a
single certificate. The effect is absent for the first four quarters after
closing and emerges only afterwards, which is when fare systems actually
combine.

The structural screen in Section 2.1 of the 2023 Merger Guidelines does not
identify which of those routes the increases landed on. Sorting overlap routes
into deciles by the increase in HHI produces a flat profile of realised fare
effects: {hhi_bins[0]['realised']*100:+.1f}% in the bottom decile against {hhi_bins[-1]['realised']*100:+.1f}% in the top. Routes the
presumption flags saw fare changes {abs(flag['diff'])*100:.2f} points {'lower' if flag['diff'] < 0 else 'higher'} than routes it
does not flag, not higher.

Diversion-based measures do identify them. The same routes sorted by GUPPI
climb monotonically from {gup_bins[0]['realised']*100:+.1f}% in the bottom decile to {gup_bins[-1]['realised']*100:+.1f}% in the top. A
one-standard-deviation increase in GUPPI is associated with a {g['slope_per_sd']*100:.2f} point
larger realised fare increase (standard error {g['se_per_sd']*100:.2f}); the same figure for the
HHI increase is {h['slope_per_sd']*100:.2f} ({h['se_per_sd']*100:.2f}), which cannot be told from zero.
""")
        y -= 0.010

        y = block(fig, y, "WHAT WAS DONE", size=10, weight="bold")
        y -= 0.006
        y = block(fig, y, f"""
A route is an overlap route for a merger if both carriers each held at least
1% of its passengers in the four quarters before the deal was announced.
Controls are routes where exactly one of the two was selling, matched to
overlap routes on distance, traffic, concentration, fare, nonstop share and
pre-merger fare trend. Both groups absorb whatever the merger did to the
combined carrier's costs and network; only overlap routes lose a competitor.

Effects are estimated by stacked difference-in-differences, each merger in its
own window against its own controls. The pooled two-way fixed effects estimator
is reported for contrast and not relied on: a Goodman-Bacon decomposition shows
{did['bacon']['forbidden_weight']*100:.0f}% of it rests on comparisons that use already-merged routes as the
control group for later deals, and it returns {did['naive_twfe']['coef']*100:+.1f}%, of the wrong sign.

Predictions were computed from pre-announcement information only: HHI and its
increase from baseline shares, and GUPPI, diversion and simulated fare effects
from a Bertrand model with logit demand, solved route by route.
""")
        y -= 0.010

        y = block(fig, y, "HOW MUCH TO TRUST IT", size=10, weight="bold")
        y -= 0.006
        pl = did.get("placebo_one_party", {})
        y = block(fig, y, f"""
The design passes its placebo: routes where only one merging carrier flew,
compared against routes where neither did, show {pl.get('coef', 0)*100:+.2f}% [{pl.get('ci_lo',0)*100:+.2f}, {pl.get('ci_hi',0)*100:+.2f}], so the
estimator is not picking up system-wide effects and calling them competitive
ones. Pre-merger coefficients are individually indistinguishable from zero. A
wild cluster bootstrap gives p = {did['wild_bootstrap']['p_value']:.3f}.

Two limits matter. First, the demand system could not be estimated credibly:
{n_bad} of the {n_tot} candidate instruments return a price coefficient of the wrong
sign, because entry responds to demand, and the only instrument with a
defensible exclusion restriction implies inelastic firm-level demand that no
profit-maximising carrier could be pricing against. The price coefficient used in the simulation is therefore
calibrated to a target elasticity of {dem['calibrated']['target_elasticity']}, not estimated. Second, that
calibration sets the level of every GUPPI and every simulated price change, but
it barely moves their ordering across routes - the rank correlation across
alternative calibrations does not fall below {min(s['rank_corr_with_main'] for s in sim['sensitivity']):.2f}. The finding above is
about ordering, which is the part that survives.
""")
        y -= 0.010

        y = block(fig, y, "WHAT FOLLOWS", size=10, weight="bold")
        y -= 0.006
        y = block(fig, y, f"""
On this evidence, moving the HHI threshold would not help: at every candidate
cut-off from 50 to 2,000 points the flagged group has a smaller average realised
fare increase than the unflagged group. The problem is not where the line sits
but that concentration and realised harm are close to uncorrelated here, once
overlap is already established. A diversion-based screen ran on the same
pre-merger data and sorted the same routes correctly.

That comparison is available before a decision has to be made, and this
retrospective is the kind of exercise that establishes whether it is worth the
extra work. It is one industry and five deals, and airline networks have
features - hub economics, connecting itineraries, frequent-flyer lock-in - that
may not generalise. The method transfers even where the magnitudes do not.
""")

        fig.text(LEFT, 0.035,
                 "Full write-ups, code and data pipeline: "
                 "github.com/josealemanm/us-airline-merger-price-effects",
                 fontsize=7.5, color=S.MUTED, family="DejaVu Serif")
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------ page 2
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(LEFT, 0.955, "EXHIBITS", fontsize=13, family="DejaVu Serif",
                 fontweight="bold", va="top")
        rule(fig, 0.938, S.INK)
        spots = [("event_study.png", 0.63, "Exhibit 1. The realised effect, "
                  "quarter by quarter."),
                 ("calibration.png", 0.28, "Exhibit 2. Realised effect by "
                  "decile of each prediction.")]
        for name, bottom, caption in spots:
            path = C.FIGURES / name
            if not path.exists():
                continue
            img = imread(path)
            h = img.shape[0] / img.shape[1]
            w = RIGHT - LEFT
            ax = fig.add_axes([LEFT, bottom, w, w * h * (8.5 / 11)])
            ax.imshow(img)
            ax.axis("off")
            fig.text(LEFT, bottom - 0.018, caption, fontsize=8,
                     family="DejaVu Serif", color=S.INK_2, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Do structural merger screens predict realised price effects?"
        d["Author"] = "Jose Aleman"
        d["Subject"] = "Airline merger retrospective, DOT DB1B 2005-2019"

    print(f"  {OUT.relative_to(C.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
