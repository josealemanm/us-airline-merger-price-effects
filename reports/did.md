# The retrospective: what fares did

Estimated by `src/06_estimate_did.py`. Fares are in constant 2019 dollars; the outcome is the log of the passenger-weighted average fare on a route in a quarter. Every regression is weighted by pre-announcement traffic and clusters on the route.

## First, the estimator not to use

Pooling all five mergers into one regression with route and quarter fixed effects gives:

> **-2.41%** on log fares, standard error 0.81 points.

The number is reported here so that it can be taken apart, not because it should be believed.

## Why not to use it

Goodman-Bacon's decomposition writes that coefficient as a weighted average of every two-by-two comparison in the data. The weights below sum to one and the weighted average reproduces the regression coefficient exactly (+1.8964% against +1.8964%), which is the check that the decomposition is right. The decomposition requires a balanced panel and equal weights, so it is run on its own regression rather than the weighted one above; that regression gives +1.90%.

| comparison | weight | average effect |
|---|---|---|
| overlap routes vs routes no merger touched | 77.7% | +2.41% |
| earlier merger vs a later one, before the later one closed | 7.9% | +1.59% |
| later merger vs an earlier one **after it was already merged** | 14.3% | -0.72% |

**14.3%** of the pooled estimate rests on comparisons that use an already-merged route as the control for a later merger. Those comparisons subtract one merger's ongoing fare response from another merger's estimate. That is the reason the rest of this report uses the stacked estimator instead.

## The estimate

Each merger against its own clean controls, in its own window, pooled.

> Overlap routes saw fares **+1.31%** relative to control routes, 95% interval **[+0.42%, +2.19%]**.

| | |
|---|---|
| observations | 247,050 route-quarters |
| routes | 7,556 |
| of which treated | 4,931 |
| of which control | 2,642 |
| standard error | 0.45 points, clustered on route |

## Are the control routes comparable?

Overlap routes are not a random sample. They are routes where two large carriers both chose to sell, which makes them longer, busier and more concentrated than the average route. Controls are therefore reweighted by the odds of being an overlap route, given what the route looked like before the merger was announced.

Standardised differences, in standard deviations. Under 0.1 in absolute value is the usual bar for balanced.

| covariate | before reweighting | after |
|---|---|---|
| log route distance | +1.25 | +0.04 |
| log baseline passengers | +0.04 | -0.25 |
| HHI (thousands) | -1.17 | -0.20 |
| carriers on the route | +1.83 | -0.01 |
| log baseline fare | +0.91 | +0.07 |
| share of passengers flying nonstop | -0.59 | -0.37 |
| pre-merger fare trend | +0.06 | -0.27 |

The worst imbalance falls from 1.83 to 0.37 standard deviations.

## Did fares already differ before the merger?

The design is only credible if treated and control routes were moving together beforehand. Each coefficient below is the gap between them in one quarter, relative to the quarter before the merger closed.

> Joint test that all pre-merger coefficients are zero: F = 5.62, p = 0.000. The largest pre-merger gap is 1.50%.

| quarters from closing | effect | 95% interval |
|---|---|---|
| -8 | -1.00% | [-2.57%, +0.56%] |
| -7 | +0.62% | [-1.36%, +2.60%] |
| -6 | +1.50% | [-0.14%, +3.14%] |
| -5 | +0.34% | [-1.29%, +1.97%] |
| -4 | -0.64% | [-2.11%, +0.84%] |
| -3 | +0.22% | [-1.21%, +1.66%] |
| -2 | +0.66% | [-0.37%, +1.68%] |
| -1 | +0.00%  *(omitted)* | [+0.00%, +0.00%] |
| +0 | -1.62% | [-2.63%, -0.61%] |
| +1 | -0.36% | [-1.69%, +0.98%] |
| +2 | +0.57% | [-0.58%, +1.73%] |
| +3 | +0.84% | [-0.27%, +1.96%] |
| +4 | +0.53% | [-0.85%, +1.90%] |
| +5 | +2.38% | [+0.71%, +4.04%] |
| +6 | +2.76% | [+1.45%, +4.07%] |
| +7 | +0.70% | [-0.70%, +2.10%] |
| +8 | +1.48% | [+0.01%, +2.95%] |
| +9 | +3.84% | [+2.27%, +5.42%] |
| +10 | +3.88% | [+2.45%, +5.32%] |
| +11 | +2.46% | [+1.01%, +3.92%] |
| +12 | +2.32% | [+0.80%, +3.84%] |

## Merger by merger

The same stacked specification, one merger at a time.

| merger | effect on overlap-route fares | 95% interval | treated routes | control routes |
|---|---|---|---|---|
| Delta / Northwest | -1.74% | [-3.09%, -0.39%] | 3,471 | 1,817 |
| United / Continental | +3.65% | [+1.47%, +5.83%] | 1,626 | 996 |
| Southwest / AirTran | +5.85% | [+3.20%, +8.50%] | 290 | 114 |
| American / US Airways | +3.78% | [+2.37%, +5.18%] | 2,143 | 1,396 |
| Alaska / Virgin America | +3.91% | [-1.53%, +9.36%] | 16 | 35 |

## Does the result survive being poked?

| test | what it asks | result |
|---|---|---|
| routes where only one party flew, against routes where neither did | neither group loses a competitor, so this should be near zero | -0.04% [-0.98%, +0.90%] |
| wild cluster bootstrap, 299 draws | whether the interval depends on the cluster count being large | p = 0.003, 95% [+0.53%, +2.18%] |
| weighted by pre-merger traffic | what the average passenger saw, rather than the average route | +1.75% [+0.40%, +3.11%] |
| no matching, all clean controls | how much the control group choice matters | -0.11% [-1.26%, +1.05%] |
| quarters +5 to +12 only | the effect once the carriers actually operate as one | +2.28% [+1.30%, +3.26%] |


### The control group and the matching

Two choices move this estimate more than anything else, and the damage comes from their combination rather than from either alone. Routes where neither merging carrier flew are shorter, thinner and served by different carriers than routes where both did; comparing the two without first making them comparable returns a fare *decrease*.

| control group | without matching | with matching |
|---|---|---|
| routes where neither party flew | -2.63% [-4.11, -1.14] | +1.11% [-1.33, +3.55] |
| routes where exactly one party flew | -0.11% [-1.26, +1.05] | **+1.31% [+0.42, +2.19]** |

Matching alone brings the two control groups into agreement on sign. It does not make them equally informative: matching onto routes neither carrier served leaves few comparable controls and an interval more than twice as wide. The reported specification is the bottom-right cell.

## Does the effect grow with how much concentration rose?

Overlap routes, grouped by the HHI increase the merger produced. The effect is measured route by route against that merger's controls over quarters +5 to +12, then averaged with passenger weights.

| HHI increase | routes | average fare effect | 95% interval |
|---|---|---|---|
| 0 to 100 | 2,018 | +0.95% | [+0.46%, +1.44%] |
| 100 to 200 | 1,405 | +0.56% | [-0.07%, +1.18%] |
| 200 to 500 | 2,037 | +1.36% | [+0.85%, +1.87%] |
| 500 to 1,000 | 1,196 | +2.20% | [+1.52%, +2.88%] |
| 1,000+ | 1,280 | -2.62% | [-3.57%, -1.66%] |
