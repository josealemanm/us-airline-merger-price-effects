"""Shared configuration: the data window, the mergers, and the sample filters.

Everything the rest of the pipeline treats as a fixed choice lives here, so that
a reader can see every judgment call in one file instead of hunting through
scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# ---------------------------------------------------------------- data window
# DB1B begins in 1993, but the first merger studied here closes in 2008Q4 and we
# want eight quarters of pre-period before it. The window stops at 2019Q4: the
# pandemic makes 2020 fares uninterpretable as a control period.
YEAR_START, QUARTER_START = 2005, 1
YEAR_END, QUARTER_END = 2019, 4


def quarters() -> list[tuple[int, int]]:
    """Every (year, quarter) in the study window, in order."""
    out = []
    y, q = YEAR_START, QUARTER_START
    while (y, q) <= (YEAR_END, QUARTER_END):
        out.append((y, q))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def qindex(year: int, quarter: int) -> int:
    """Quarters since 2000Q1. Gives arithmetic on calendar quarters."""
    return (year - 2000) * 4 + (quarter - 1)


# -------------------------------------------------------------------- mergers
@dataclass(frozen=True)
class Merger:
    key: str
    label: str
    acquirer: str
    target: str
    announced: tuple[int, int]  # quarter the deal became public
    closed: tuple[int, int]     # quarter the deal legally closed
    certificate: tuple[int, int]  # quarter of the single operating certificate
    brand_retired: tuple[int, int]  # quarter the acquired brand stopped selling

    @property
    def close_q(self) -> int:
        return qindex(*self.closed)

    @property
    def announce_q(self) -> int:
        return qindex(*self.announced)

    @property
    def carriers(self) -> tuple[str, str]:
        return (self.acquirer, self.target)


# The five large domestic passenger-airline mergers that clear inside the window.
# Announcement dates matter because carriers can reshuffle networks between
# announcement and closing; the pre-merger baseline is measured before the
# announcement so that it is not contaminated by the deal itself.
MERGERS: list[Merger] = [
    # The certificate date and the date the acquired brand stopped selling
    # tickets are not the same event, and for Southwest / AirTran they are
    # almost three years apart: the certificates merged in 2012 but AirTran-coded
    # tickets were still being sold into late 2014. The validation stage checks
    # the ticketing data against the second of these, which is the one that
    # shows up in DB1B.
    Merger("DL_NW", "Delta / Northwest",       "DL", "NW", (2008, 2), (2008, 4), (2009, 4), (2010, 1)),
    Merger("UA_CO", "United / Continental",    "UA", "CO", (2010, 2), (2010, 4), (2011, 4), (2012, 1)),
    Merger("WN_FL", "Southwest / AirTran",     "WN", "FL", (2010, 3), (2011, 2), (2012, 1), (2014, 4)),
    Merger("AA_US", "American / US Airways",   "AA", "US", (2013, 1), (2013, 4), (2015, 2), (2015, 3)),
    Merger("AS_VX", "Alaska / Virgin America", "AS", "VX", (2016, 2), (2016, 4), (2018, 1), (2018, 2)),
]
MERGERS_BY_KEY = {m.key: m for m in MERGERS}

# ------------------------------------------------------------- sample filters
# DB1B is a 10% sample of ticket coupons, so passenger counts are multiplied by
# ten to read as levels. Nothing downstream depends on the scaling except the
# consumer-harm dollars.
DB1B_SAMPLE_RATE = 0.10

MIN_FARE_NOMINAL = 25.0      # below this is almost always a mileage redemption
MAX_FARE_NOMINAL = 2500.0    # above this is almost always a coding error
MAX_COUPONS = 2              # nonstop, or one connection, in this direction
MIN_MARKET_PAX_PER_Q = 20    # in sample terms: 200 real passengers a quarter
MIN_CARRIER_SHARE = 0.01     # a carrier must hold 1% of a market to count as in it

# Event-study window, in quarters relative to the closing quarter.
EVENT_MIN, EVENT_MAX = -8, 12
# Quarters before the announcement used to measure who was in a market.
BASELINE_QUARTERS = 4

# 2023 Merger Guidelines structural thresholds (Section 2.1).
HHI_CONCENTRATED = 1800
DELTA_HHI_THRESHOLD = 100
# The Guidelines' second structural screen: share above 30% with delta above 100.
SHARE_THRESHOLD = 0.30

# ------------------------------------------------------------ carrier groups
# The nest in the demand model. Network carriers run connecting hubs and sell a
# differentiated product with assigned seats, lounges and interlining; low-cost
# carriers run point-to-point and compete mainly on fare. Passengers are taken
# to substitute more readily inside one of these groups than across them, which
# is what the nesting parameter measures and what the data is asked to confirm.
NETWORK_CARRIERS = {
    "AA", "DL", "UA", "US", "CO", "NW", "TW", "HP", "AS", "HA",
}


def carrier_group(code: str) -> str:
    return "network" if code in NETWORK_CARRIERS else "lowcost"


# ------------------------------------------------------------- market size
# The nested logit needs a potential market, not just realised passengers,
# because the outside option has to have a size. There is no passenger count for
# people who did not fly, so it is proxied: a city pair's potential is taken to
# be proportional to the geometric mean of total origin-and-destination traffic
# at its two endpoints, a gravity form. KAPPA scales that into a level. The
# choice matters for the dollar value of consumer harm and barely at all for the
# simulated percentage price changes; 07 and 08 report the sensitivity.
# The share of its potential market the MEDIAN route is taken to sell.
#
# This is not a free dial. Under logit demand the ratio of the aggregate market
# elasticity to a firm's own elasticity is approximately the outside share, so
# fixing the outside share fixes how much of a carrier's lost traffic leaves the
# market rather than switching to a rival - which is the number the whole merger
# simulation turns on. Published estimates put US airline market-level demand
# elasticity near -1.2 and firm-level demand near -2.5, and their ratio is about
# one half. The potential market is therefore set so the median route sells to
# half of it.
#
# The earlier choice of 0.10 came from wanting the outside share to "look
# reasonable" and was wrong for a concrete reason: it implied that 93 per cent
# of the passengers priced off a carrier stop flying altogether rather than
# switching airlines, which made every diversion ratio, every GUPPI and every
# simulated price effect far too small.
INSIDE_SHARE_TARGET = 0.50
INSIDE_SHARE_TARGET_ALTS = (0.35, 0.70)
