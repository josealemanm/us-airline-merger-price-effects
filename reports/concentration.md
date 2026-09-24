# Concentration and the structural screens

Computed by `src/05_concentration.py` from baseline shares only: the four quarters ending the quarter before each merger was announced. No post-merger information is used.

## Overlap routes by merger

| merger | overlap routes | median pre-merger HHI | median delta HHI | median combined share | flagged by the presumption | routes going to monopoly |
|---|---|---|---|---|---|---|
| Delta / Northwest | 3,667 | 3,529 | 409 | 44.4% | 83% | 20 |
| United / Continental | 1,779 | 2,819 | 168 | 23.8% | 63% | 1 |
| Southwest / AirTran | 328 | 2,572 | 169 | 26.7% | 62% | 0 |
| American / US Airways | 2,216 | 3,590 | 195 | 28.2% | 70% | 0 |
| Alaska / Virgin America | 19 | 3,327 | 105 | 22.0% | 53% | 0 |

Across all five mergers there are **8,009 overlap routes**. The structural presumption attaches on **74%** of them.

## Where the flags come from

The 2023 Guidelines set two structural screens. The first asks whether the post-merger HHI exceeds 1,800 and the increase exceeds 100. The second asks whether the combined share exceeds 30% with the same increase. A route is flagged if either attaches.

| screen | routes flagged | share of overlap routes |
|---|---|---|
| HHI above 1,800 and delta above 100 | 5,927 | 74.0% |
| combined share above 30% and delta above 100 | 4,271 | 53.3% |
| either | 5,929 | 74.0% |

## How much does the screen actually separate?

A screen is only useful if it divides the routes into groups that differ. These are the overlap routes split by whether the presumption attaches.

| | flagged | not flagged |
|---|---|---|
| routes | 5,929 | 2,080 |
| median pre-merger HHI | 3,307 | 3,587 |
| median delta HHI | 401 | 50 |
| median combined share | 41.7% | 12.4% |
| median carriers on the route | 6 | 6 |
| median baseline passengers | 248 | 647 |
| median distance, miles | 1,071 | 1,103 |

## The distribution of the HHI increase

The threshold sits at 100 points. Where the overlap routes actually fall relative to it decides how much work the threshold is doing.

| delta HHI | routes | share |
|---|---|---|
| 0 to 100 | 2,026 | 25.3% |
| 100 to 200 | 1,410 | 17.6% |
| 200 to 500 | 2,050 | 25.6% |
| 500 to 1,000 | 1,206 | 15.1% |
| 1,000 to 2,500 | 1,024 | 12.8% |
| 2,500 and above | 293 | 3.7% |

Median increase across all overlap routes: **254 points**. **25%** of overlap routes fall below the 100-point threshold on their own.
