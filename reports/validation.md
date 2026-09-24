# Do the screens predict what happened?

Written by `src/09_screens_vs_outcomes.py`. 7,936 overlap routes, each with a prediction computed from pre-merger information only and a realised fare effect measured against matched control routes over quarters +5 to +12 after closing.

## The structural presumption

| | routes | average realised fare effect |
|---|---|---|
| flagged by the presumption | 5,864 | +2.01% (0.20) |
| not flagged | 2,072 | +2.72% (0.30) |
| **difference** | | **-0.71%** [-1.41%, -0.01%] |

On this measure the presumption **separates** the routes where fares rose from the routes where they did not.

## How much information is in each prediction?

Each row regresses the realised route-level effect on one prediction, with merger fixed effects and standard errors clustered on the route. The middle column is the change in the realised effect for a one-standard-deviation increase in the prediction, which puts predictors measured on different scales on the same footing.

| prediction | realised effect per SD of prediction | rank correlation | slope | is the slope zero? | is it one? |
|---|---|---|---|---|---|
| HHI increase | +0.235% (0.202) | -0.017 | +0.00000 | p = 0.246 | p = 0.000 |
| combined share of the two parties | -0.589% (0.186) | -0.060 | -0.02544 | p = 0.002 | p = 0.000 |
| diversion to the merger partner | +2.258% (0.187) | +0.118 | +0.27694 | p = 0.000 | p = 0.000 |
| GUPPI | +2.772% (0.217) | +0.189 | +0.52944 | p = 0.000 | p = 0.000 |
| simulated fare change on the route | +2.200% (0.217) | +0.116 | +0.57014 | p = 0.000 | p = 0.000 |

The slope column is only interpretable for the simulated fare change, where prediction and outcome are in the same units and a slope of one would mean the simulation was correctly scaled as well as informative.

## Is the merger simulation calibrated?

> Slope of realised on simulated: **+0.570** [+0.460, +0.681].

Against the null that the simulation carries no information, p = 0.000. Against the null that it is correctly scaled, p = 0.000.

## Realised effect by decile of each prediction

If a prediction works, the realised effect should climb across its deciles. Each cell is the average realised fare effect for the routes in that decile, with its standard error.

### HHI increase

| decile | routes | prediction range | realised effect |
|---|---|---|---|
| 1 | 794 | 2 to 37 | +2.31% (0.49) |
| 2 | 794 | 37 to 76 | +3.21% (0.49) |
| 3 | 793 | 76 to 122 | +2.21% (0.47) |
| 4 | 794 | 122 to 180 | +2.40% (0.50) |
| 5 | 793 | 180 to 252 | +2.21% (0.52) |
| 6 | 794 | 252 to 355 | +2.49% (0.50) |
| 7 | 793 | 355 to 526 | +2.13% (0.51) |
| 8 | 794 | 526 to 816 | +2.28% (0.53) |
| 9 | 793 | 817 to 1,447 | +0.91% (0.59) |
| 10 | 794 | 1,449 to 4,972 | +1.79% (0.59) |

Top decile minus bottom decile: **-0.52%** [-2.03%, +0.99%].

### combined share of the two parties

| decile | routes | prediction range | realised effect |
|---|---|---|---|
| 1 | 794 | 0.022 to 0.105 | +1.88% (0.50) |
| 2 | 794 | 0.105 to 0.163 | +3.72% (0.47) |
| 3 | 793 | 0.163 to 0.216 | +3.45% (0.50) |
| 4 | 794 | 0.216 to 0.269 | +2.93% (0.47) |
| 5 | 793 | 0.269 to 0.326 | +3.74% (0.47) |
| 6 | 794 | 0.327 to 0.393 | +3.06% (0.49) |
| 7 | 793 | 0.393 to 0.478 | +2.23% (0.53) |
| 8 | 794 | 0.478 to 0.582 | +0.74% (0.56) |
| 9 | 793 | 0.582 to 0.724 | +0.42% (0.52) |
| 10 | 794 | 0.724 to 1.000 | -0.22% (0.64) |

Top decile minus bottom decile: **-2.10%** [-3.68%, -0.52%].

### diversion to the merger partner

| decile | routes | prediction range | realised effect |
|---|---|---|---|
| 1 | 787 | 0.000 to 0.008 | -0.62% (0.48) |
| 2 | 787 | 0.008 to 0.013 | -0.46% (0.52) |
| 3 | 786 | 0.013 to 0.018 | +1.47% (0.51) |
| 4 | 787 | 0.018 to 0.023 | +1.67% (0.47) |
| 5 | 787 | 0.023 to 0.030 | +1.92% (0.48) |
| 6 | 786 | 0.030 to 0.039 | +2.39% (0.50) |
| 7 | 787 | 0.039 to 0.053 | +2.72% (0.52) |
| 8 | 786 | 0.053 to 0.078 | +3.16% (0.55) |
| 9 | 787 | 0.078 to 0.131 | +3.54% (0.54) |
| 10 | 787 | 0.132 to 0.809 | +6.01% (0.60) |

Top decile minus bottom decile: **+6.63%** [+5.13%, +8.13%].

### GUPPI

| decile | routes | prediction range | realised effect |
|---|---|---|---|
| 1 | 787 | 0.000 to 0.003 | -1.89% (0.48) |
| 2 | 787 | 0.003 to 0.004 | -0.99% (0.50) |
| 3 | 786 | 0.004 to 0.006 | -0.69% (0.48) |
| 4 | 787 | 0.006 to 0.009 | +2.33% (0.50) |
| 5 | 787 | 0.009 to 0.012 | +1.64% (0.49) |
| 6 | 786 | 0.012 to 0.015 | +1.10% (0.50) |
| 7 | 787 | 0.015 to 0.022 | +3.51% (0.51) |
| 8 | 786 | 0.022 to 0.034 | +4.24% (0.55) |
| 9 | 787 | 0.034 to 0.060 | +4.63% (0.54) |
| 10 | 787 | 0.060 to 0.877 | +7.92% (0.56) |

Top decile minus bottom decile: **+9.81%** [+8.36%, +11.26%].

### simulated fare change on the route

| decile | routes | prediction range | realised effect |
|---|---|---|---|
| 1 | 787 | 0.000 to 0.000 | -0.45% (0.49) |
| 2 | 787 | 0.000 to 0.001 | +0.99% (0.46) |
| 3 | 786 | 0.001 to 0.002 | +0.98% (0.47) |
| 4 | 787 | 0.002 to 0.002 | +1.90% (0.48) |
| 5 | 787 | 0.002 to 0.004 | +1.98% (0.49) |
| 6 | 786 | 0.004 to 0.005 | +1.12% (0.54) |
| 7 | 787 | 0.005 to 0.009 | +1.87% (0.54) |
| 8 | 786 | 0.009 to 0.015 | +3.61% (0.55) |
| 9 | 787 | 0.015 to 0.032 | +2.89% (0.55) |
| 10 | 787 | 0.032 to 0.755 | +6.91% (0.58) |

Top decile minus bottom decile: **+7.36%** [+5.87%, +8.85%].

## Where should the threshold sit?

The Guidelines put the line at an HHI increase of 100 points. This is what other lines would have done, on this data: how many routes each catches, how much the flagged and unflagged groups differ, and what share of the total realised fare increase falls inside the flagged group.

| HHI increase above | routes flagged | share of overlap routes | flagged | not flagged | difference | share of realised increases captured |
|---|---|---|---|---|---|---|
| 50 | 6,899 | 87% | +2.19% | +2.26% | -0.08% (0.46) | 88% |
| 100 **(the Guidelines)** | 5,918 | 75% | +2.03% | +2.68% | -0.65% (0.36) | 75% |
| 200 | 4,513 | 57% | +1.94% | +2.53% | -0.58% (0.33) | 58% |
| 300 | 3,534 | 45% | +1.89% | +2.44% | -0.55% (0.33) | 46% |
| 500 | 2,476 | 31% | +1.61% | +2.46% | -0.85% (0.37) | 32% |
| 800 | 1,613 | 20% | +1.44% | +2.39% | -0.94% (0.45) | 22% |
| 1,200 | 1,051 | 13% | +1.42% | +2.31% | -0.89% (0.54) | 14% |
| 2,000 | 478 | 6% | +1.78% | +2.22% | -0.44% (0.83) | 7% |

## Level, not just ordering

| | simulated | realised |
|---|---|---|
| median across routes | +0.36% | +3.21% |
| mean across routes | +1.44% | +2.18% |

The simulated figures are conditional on the calibration in [`demand.md`](demand.md), so a gap in levels is as much a statement about that calibration as about the model. The ordering, which stage 8 showed is close to invariant to the calibration, is the part that carries weight.
