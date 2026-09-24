"""One house style for every figure in the project.

Colours are the validated default palette from the data-visualisation reference:
the categorical slots are used in fixed order and never cycled, magnitude is
encoded with the single blue ramp, and the emphasis pattern - one hue against a
de-emphasis grey - carries the charts where a single result is the point.

The rule that does the most work here: an estimate whose interval covers zero is
drawn in grey, not in colour. Several figures in this project have a minority of
bars that clear zero, and colouring all of them equally would let a reader take
a ranking off the page that the data does not support.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- categorical slots, fixed order --------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# --- sequential blue ramp, light to dark ---------------------------------
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]

# --- surfaces and ink -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9aa5ad"          # de-emphasis grey: "cannot be told from zero"
GRID = "#e6e5e1"
ZERO = "#b0413c"           # the reference line, not a series


def apply() -> None:
    """Set the rcParams every figure inherits."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
    })


def subtitle(ax, title: str, sub: str) -> None:
    """A bold title with an explanatory line under it, both left-aligned.

    The title is pushed up by the height the subtitle will take, so a two- or
    three-line subtitle does not print on top of it.
    """
    n = sub.count("\n") + 1
    ax.set_title(title, loc="left", pad=14 + 13 * n)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, color=INK_2,
                linespacing=1.45)


def source(fig, text: str) -> None:
    fig.text(0.005, -0.02, text, ha="left", va="top", fontsize=8, color=MUTED)


def hgrid(ax) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def vgrid(ax) -> None:
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def significant_colors(lo, hi, colour: str = BLUE, muted: str = MUTED) -> list:
    """Colour an estimate only when its interval clears zero."""
    return [colour if (l > 0 or h < 0) else muted for l, h in zip(lo, hi)]


def pct(ax, axis: str = "y") -> None:
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
