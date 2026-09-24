# Airline merger screens: do they predict which routes get more expensive?

I measured what happened to fares after five US airline mergers, then checked
whether the tools antitrust agencies use to flag routes *before* a merger would
have picked out the routes where fares actually rose.

The data is the Department of Transportation's Origin and Destination Survey
(DB1B), a 10% sample of every domestic ticket sold: 344 million records over 60
quarters, 2005 to 2019. I estimate the fare effects with a stacked
difference-in-differences design, and generate the predictions with a merger
simulation built from scratch: demand estimation, cost recovery, and a Bertrand
pricing equilibrium.

**[Open the interactive dashboard](https://josealemanm.github.io/us-airline-merger-price-effects/dashboard.html)**
It runs in the browser, no installation.

![Realised fare effect by decile of each prediction](reports/figures/calibration.png)

Sorted by how much a merger raised concentration, fares afterwards are flat
across the whole range. Sorted by the value of sales diverted to the merger
partner, they climb from −1.9% to +7.9%. Both numbers were computable before the
mergers closed.

## The question

An agency has to decide which routes a merger will hurt before it closes. For
airlines that means picking out the **overlap routes**: city pairs where both
merging carriers were already selling tickets, so the merger removes a
competitor. I find 8,009 of them across the five deals.

The 2023 Merger Guidelines rank those routes by concentration. Compute the
**Herfindahl-Hirschman Index**, the sum of squared market shares on a 0–10,000
scale, before and after. If the merger leaves the route concentrated and raises
the index by more than 100 points, harm is presumed. That screen needs only
passenger counts. It flags 74% of these routes.

The alternative is **GUPPI** (gross upward pricing pressure). Estimate how
passengers substitute between carriers, work out what share of the passengers
priced off one merging carrier would switch to the other, and multiply by what
those passengers were worth. That share is the **diversion ratio**. Much more
work, and it needs a demand model.

Both are predictions. I have fifteen years of data on what happened next.

## What I found

**Fares rose on overlap routes.** Against matched control routes, **+1.31%**
(95% interval [+0.42%, +2.19%]), and **+2.28%** [+1.30%, +3.26%] once the
carriers were flying under a single operating certificate.

**The effect takes five quarters to appear.**

![Event study](reports/figures/event_study.png)

Every coefficient from the closing quarter through quarter +4 sits between −1.6%
and +0.8%, none distinguishable from zero. The effect shows up at quarter +5 and
stays. That timing is not noise: two carriers that have legally merged still fly
under separate operating certificates and file separate fares for four to six
quarters afterwards. Until the certificates combine, there is no single firm
setting one price. A retrospective with a one-year window would find nothing
here.

**The concentration screen does not sort the routes.**

| | routes | average fare effect |
|---|---|---|
| flagged by the presumption | 5,864 | +2.01% |
| not flagged | 2,072 | +2.72% |
| **difference** | | **−0.71%** [−1.41%, −0.01%] |

Flagged routes saw *smaller* increases than unflagged ones. The interval barely
clears zero, so I would not lean on the sign, but the screen is plainly not
finding the routes where harm landed. By decile of the HHI increase the profile
is flat: +2.31% at the bottom, +1.79% at the top.

Moving the threshold does not help. At every cut-off I tested, from 50 points to
2,000, the flagged group has the smaller average increase.

**Diversion-based measures do sort them.** Same 7,936 routes, same outcomes,
ranked differently:

| prediction | fare effect per SD of the prediction | rank correlation |
|---|---|---|
| GUPPI | **+2.77%** (0.22) | +0.189 |
| diversion to the merger partner | +2.26% (0.19) | +0.118 |
| simulated fare change | +2.20% (0.22) | +0.116 |
| increase in HHI | +0.24% (0.20) | −0.017 |
| combined share of the two carriers | −0.59% (0.19) | −0.060 |

A one-standard-deviation increase in GUPPI is worth 2.77 percentage points of
realised fare increase. For the HHI increase the same figure is 0.24 points with
a standard error of 0.20, indistinguishable from zero (p = 0.25). Combined share
points the wrong way.

All of these use pre-merger data only. The information needed to rank the routes
correctly was available at the time.

**The simulation ranks well and scales badly.** Regressing realised effects on
simulated ones gives a slope of **0.57** [0.46, 0.68]. That rejects the null
that the simulation is uninformative, and also the null that it is correctly
scaled. It overstates the size of the effect by about a factor of 1.8 while
still ordering routes correctly, which is what I would expect from calibrating
the price coefficient instead of estimating it.

**The mergers differ.**

![Effect by merger](reports/figures/by_merger.png)

Four of five raised fares where the carriers overlapped. Delta/Northwest lowered
them by 1.74% [−3.09%, −0.39%]. It is the earliest deal and the one with the
most overlap routes. Alaska/Virgin America had only 16 overlap routes meeting my
sample requirements, so its interval is uninformative.

## Three things that broke

**The naive estimator returns the wrong sign.** Pooling all five mergers into one
regression with route and quarter fixed effects gives **−2.41%**. Two problems
produce that. One is the control group, below. The other is staggered timing.

![Goodman-Bacon decomposition](reports/figures/bacon.png)

Goodman-Bacon (2021) showed that a two-way fixed effects coefficient on
staggered treatment is a weighted average of every two-by-two comparison in the
data, and that some of those comparisons use *already-treated* units as the
control group for later-treated ones. Here **14.3%** of the weight sits on
comparisons that use routes already hit by an earlier merger as the
counterfactual for a later one. Since merger effects grow over the first two
years, those comparisons subtract one merger's ongoing increase from another's
estimate. My implementation in `src/econ/bacon.py` checks itself: the weighted
components must reproduce the regression coefficient, or it raises.

**My first control group was wrong.** I started by comparing overlap routes
against routes where *neither* merging carrier flew. That looks like the
cleanest comparison and is not. A route where both American and US Airways sold
tickets is long, busy and concentrated by construction; a route where neither
sold anything is short, thin and often served by one low-cost carrier.

I switched to routes where **exactly one** of the two carriers was selling. Both
groups absorb whatever the merger did to the combined airline's costs and
network. Only overlap routes lose a competitor. Together with matching, that
choice moves the answer more than anything else:

| control group | without matching | with matching |
|---|---|---|
| routes where neither carrier flew | −2.63% [−4.11, −1.14] | +1.11% [−1.33, +3.55] |
| routes where exactly one carrier flew | −0.11% [−1.26, +1.05] | **+1.31% [+0.42, +2.19]** |

The top-left cell is what I would have reported without checking: a significant
*fall* in fares. Matching brings both control groups into agreement on sign, but
matching onto routes neither carrier served leaves few comparable controls and
an interval more than twice as wide. I report the bottom-right cell.

**Inverse-odds weighting collapsed the control group.** Weighting controls by
the odds of treatment is the textbook approach. It gave me confidence intervals
of ±8 percentage points on tens of thousands of observations. A few control
routes sat at propensity scores near one, their odds ran into the tens, and
after multiplying by passenger traffic one merger's control arm had an effective
sample size of **3.4 routes**, with ten routes carrying 97% of the weight.

Nearest-neighbour matching bounds this, since a control route can only be
weighted by the treated routes it stands in for. Effective control counts went
from single digits to hundreds, and intervals fell to about ±1 point.

![Covariate balance](reports/figures/balance.png)

## Where the demand estimation failed

GUPPI needs a demand system, which means instrumenting the fare: airlines charge
more where demand is strong, so the raw price-quantity relationship understates
how price-sensitive passengers are. I built five instruments and tested each one
separately rather than pooling them and reading the overidentification test.

![Instrument audit](reports/figures/instruments.png)

Three return a price coefficient of the **wrong sign**, implying passengers buy
more as fares rise. They fail for one reason: entry responds to demand. A
carrier adds a route when it expects the route to sell, so a demand shock brings
in rivals, rivals push the fare down, and an instrument built on rival counts
carries the shock it was supposed to exclude. Their first-stage F statistics run
into the hundreds and thousands, so the usual strength diagnostics would not
have caught this.

What survives is jet fuel prices interacted with route distance, a cost shifter
with no demand story attached. It is much the weakest in the first stage (F = 23,
against 707 for the next weakest) and gives the right sign with a coefficient
larger than OLS, as the endogeneity argument predicts. But it implies a median
own-price elasticity of **−1.25**, and a profit-maximising firm never prices
where its own demand is inelastic: the Bertrand first-order condition would
return a negative marginal cost.

So the simulation does not run on an estimate. I **calibrate** the price
coefficient to a median own-price elasticity of −2.5, within the range of
published estimates for US airline demand, and re-run everything at −1.5 and
−4.0. This is labelled as a calibration everywhere it appears.

That matters less for the main result than it might. Under logit demand,
diversion ratios depend on market shares, not on the price coefficient. So the
*level* of every GUPPI scales with the calibration while the *ordering* of
routes barely moves: the rank correlation between the reported run and every
alternative stays above **0.92**. My finding is about ordering. The dollar
figures in [`reports/simulation.md`](reports/simulation.md) are not, and should
be read as scaled to an assumption.

## Robustness

| test | what it asks | result |
|---|---|---|
| placebo on one-carrier routes | is the design picking up non-competitive effects? | −0.04% [−0.98%, +0.90%] |
| wild cluster bootstrap, 299 draws | does the interval need the cluster count to be large? | p = 0.003 |
| weighted by traffic | average route, or average passenger? | +1.75% |
| matching switched off | how much is the matching doing? | −0.11% |
| quarters +5 to +12 only | what happens once the carriers really are one firm? | +2.28% [+1.30%, +3.26%] |

The placebo is the one I care about most. Routes where only one merging carrier
flew, against routes where neither did, give −0.04% [−0.98%, +0.90%]. Neither
group loses a competitor, so an effect there would mean the design was picking
up something other than lost competition.

Pre-merger coefficients are individually indistinguishable from zero and the
largest is 1.5%, but the joint test across all of them rejects at p < 0.001.
With 6,400 clusters that test can detect very small deviations, and it is
detecting one. The coefficients bounce rather than trend, which is the pattern I
want, but the design is not perfectly clean.

## Terms

**Overlap route** — a city pair where both merging carriers each held at least
1% of passengers, averaged over the four quarters before the merger was
*announced* rather than before it closed. Carriers move capacity once a deal is
public, so a later baseline would already reflect the merger.

**HHI** — Herfindahl-Hirschman Index, the sum of squared market shares on a
0–10,000 scale. A monopoly is 10,000; four equal firms is 2,500. A merger raises
it by exactly twice the product of the two shares.

**Diversion ratio** — of the passengers who stop buying carrier A's ticket when
its fare rises, the fraction who buy carrier B's. Not observable; it comes from
an estimated demand system.

**GUPPI** — the value of sales A diverts to B when A raises its fare, as a
fraction of A's fare. A GUPPI of 5% means the merger gives A an incentive to
price as though its marginal cost had risen 5%.

**Realised effect** — the change in a route's average real fare from the eight
quarters before closing to quarters +5 through +12, minus the same change on its
matched controls.

**Stacked difference-in-differences** — each merger gets its own block: its
overlap routes, its controls, its own 21-quarter window. Route-by-block and
quarter-by-block fixed effects mean no comparison crosses mergers, so the
contamination the Bacon decomposition measures cannot arise.

**95% interval** — clustered on the route, since fares on a route are correlated
quarter to quarter. In every chart, an estimate whose interval covers zero is
grey.

## The data

| | |
|---|---|
| Source | US DOT Bureau of Transportation Statistics, Airline Origin and Destination Survey (DB1B), Market file |
| Window | 60 quarters, 2005Q1 to 2019Q4 |
| Volume | 344 million ticket records, 5.2 GB downloaded |
| After filtering | 19,189 routes, 696,963 route-quarters, 2.3 million carrier-route-quarters |
| Overlap routes | 8,009 across five mergers |
| Deflator | CPI-U from FRED; fares in constant 2019 dollars |
| Cost instrument | US Gulf Coast jet fuel spot price, from FRED |

DB1B is the dataset the agencies themselves use for airline competition
analysis. Each record carries origin, destination, ticketing carrier, fare and
passenger count.

The window stops at 2019Q4 on purpose: the pandemic makes 2020 fares
uninterpretable as a control period, and the last merger needs twelve quarters
of follow-up.

Mergers: Delta/Northwest (closed 2008Q4), United/Continental (2010Q4),
Southwest/AirTran (2011Q2), American/US Airways (2013Q4), Alaska/Virgin America
(2016Q4).

I do not take the event dates on trust. `src/03_validate.py` checks them against
the data: each acquired carrier should stop appearing as a ticketing carrier
when its brand is retired, and each one does.

## How it works

Twelve stages, each runnable on its own.

1. **Pull** (`01_pull_db1b.py`) — 60 quarterly files from BTS with retry and
   backoff. Each unpacks to about six million ticket records, too much to keep
   sixty times over, so every quarter is aggregated on arrival and the raw file
   deleted. The manifest records the URL, byte count and SHA-256 of each file.
2. **Macro** (`02_pull_macro.py`) — CPI-U and jet fuel from FRED.
3. **Validate** (`03_validate.py`) — completeness, duplicates, impossible
   values, and the merger-timing check. Writes
   [`reports/data_quality.md`](reports/data_quality.md), exits non-zero on a
   hard failure.
4. **Panel** (`04_build_panel.py`) — route-quarter and carrier-route-quarter
   panels, real fares, treatment definition.
5. **Concentration** (`05_concentration.py`) — shares, HHI and the Guidelines
   screens from pre-announcement data.
   → [`reports/concentration.md`](reports/concentration.md)
6. **Retrospective** (`06_estimate_did.py`) — matching, stacked DiD, event
   study, Goodman-Bacon decomposition, placebo, wild cluster bootstrap.
   → [`reports/did.md`](reports/did.md)
7. **Demand** (`07_estimate_demand.py`) — nested logit, instrument audit,
   calibration. → [`reports/demand.md`](reports/demand.md)
8. **Simulation** (`08_merger_sim.py`) — marginal costs from the pre-merger
   first-order conditions, post-merger Bertrand equilibrium, GUPPI, UPP, CMCR
   and consumer surplus, route by route.
   → [`reports/simulation.md`](reports/simulation.md)
9. **Scoring** (`09_screens_vs_outcomes.py`) — predictions against outcomes.
   → [`reports/validation.md`](reports/validation.md)
10. **Figures** (`10_figures.py`)
11. **Dashboard** (`11_dashboard.py`) — one self-contained HTML file, no CDN.
12. **Memo** (`12_memo.py`) — [`reports/memo.pdf`](reports/memo.pdf), two pages
    for someone who will not read any of this.

The econometrics is in `src/econ/`, written from numpy and scipy rather than
pulled from a package, so I know what every standard error is doing:

| module | what it does |
|---|---|
| `fe_ols.py` | weighted least squares absorbing high-dimensional fixed effects by alternating projections, cluster-robust |
| `iv.py` | 2SLS with absorbed effects, first-stage F statistics, Hansen's J |
| `did.py` | event studies, the stacked estimator, group-time average treatment effects |
| `bacon.py` | the Goodman-Bacon decomposition, self-checking |
| `matching.py` | propensity scores, nearest-neighbour matching, balance tables |
| `nested_logit.py` | Berry inversion, elasticities, diversion, consumer surplus |
| `bertrand.py` | cost recovery, equilibrium solving, GUPPI, UPP, CMCR |
| `inference.py` | wild cluster bootstrap, randomization inference |
| `concentration.py` | HHI and the Guidelines screens |

## Tests

83 tests, run in CI on every push.

The estimators are checked against implementations I did not write. Coefficients
*and* cluster-robust standard errors match `statsmodels` to eight decimal
places, weighted and unweighted, one-way and two-way. The IV estimator matches
`linearmodels` to the same tolerance. `linearmodels` is a test dependency only;
the pipeline never imports it.

The demand and simulation code is checked against closed forms. At a nesting
parameter of zero, the nested logit must reproduce plain logit diversion ratios,
own-price elasticities and single-product Lerner margins exactly. Re-solving the
pre-merger equilibrium must return the observed prices. A merger must raise the
merging carriers' prices, rivals must respond upward by less, and a merger to
monopoly must do more than a partial one.

Two tests encode mistakes rather than theory. One constructs poor overlap
between treated and control units and asserts that inverse-odds weighting
degenerates while matching does not. The other asserts that treated units with
no comparable control are dropped *and counted*, not quietly matched to
something unlike them.

```bash
python -m pytest tests/ -q
```

## Running it

```bash
git clone https://github.com/josealemanm/us-airline-merger-price-effects
cd us-airline-merger-price-effects
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
```

On macOS or Linux, `make setup` then `make all`. On Windows there is no `make`,
so run the stages directly:

```bash
.venv/Scripts/python.exe src/01_pull_db1b.py
```

then `02_pull_macro.py` through `12_memo.py` in order. The pull takes about
forty minutes and 5.2 GB; everything after it runs in a few minutes. The two
large panels and the raw quarterly files are not in the repository, since stage 4
rebuilds them, but every analysis output behind the numbers above is committed.
The numbers can be checked without running the pull.

## Limits

One industry, five deals. Airlines have features that may not carry elsewhere:
hub economics, connecting itineraries, frequent-flyer lock-in. My result is
about airline routes with established overlap, not a general claim about
concentration.

The demand system is calibrated, not estimated. The ordering of routes survives
that; the dollar figures do not.

And I compare a screen against an outcome, not against the decision an agency
should have made. A screen that flags too many routes costs investigation time;
one that misses routes costs consumers. Which error to prefer is not a question
this data answers.

## Source

US Department of Transportation, Bureau of Transportation Statistics, Airline
Origin and Destination Survey (DB1B), Market file, 2005Q1–2019Q4. CPI-U and US
Gulf Coast jet fuel spot prices from the Federal Reserve Bank of St Louis
(FRED). All public, no key required.
