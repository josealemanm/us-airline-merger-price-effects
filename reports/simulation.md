# Merger simulation

Run by `src/08_merger_sim.py` on 7,932 overlap routes, using only pre-announcement fares and shares. Demand parameters are the calibration argued for in [`demand.md`](demand.md); every number below is conditional on it.

## What the simulation predicts

| | median across routes |
|---|---|
| fare change, merging carriers | +1.11% |
| fare change, whole route | +0.36% |
| GUPPI | 0.012 |
| diversion to the merger partner | 0.030 |
| pre-merger margin implied by the FOC | 0.421 |
| cost saving needed to hold fares flat (CMCR) | 0.056 |

Summed over every overlap route, the predicted loss of consumer surplus is **$1.02 billion a year**. That figure is the one most exposed to the calibration; see the sensitivity table.

## By merger

| merger | routes | median route fare change | median GUPPI | predicted annual harm |
|---|---|---|---|---|
| Delta / Northwest | 3,612 | +0.67% | 0.016 | $181M |
| United / Continental | 1,772 | +0.23% | 0.010 | $109M |
| Southwest / AirTran | 326 | +0.35% | 0.014 | $398M |
| American / US Airways | 2,203 | +0.23% | 0.008 | $260M |
| Alaska / Virgin America | 19 | +0.17% | 0.008 | $72M |

## How much of this is the calibration?

The same simulation at demand parameters either side of the ones reported. The last column is the rank correlation between each run's route-level fare predictions and the reported run's.

| target elasticity | sigma | median route fare change | median margin | annual harm | rank correlation |
|---|---|---|---|---|---|
| **-2.5** | **0.00** | **+0.36%** | **0.421** | **$1.02B** | **1.000** |
| -1.5 | 0.00 | +0.60% | 0.701 | $1.70B | 1.000 |
| -4.0 | 0.00 | +0.23% | 0.263 | $0.64B | 1.000 |
| -2.5 | 0.25 | +0.70% | 0.426 | $1.65B | 0.964 |
| -2.5 | 0.50 | +1.04% | 0.436 | $2.56B | 0.922 |

The level moves a great deal and the ordering barely moves: the lowest rank correlation across every alternative is **0.922**. Under logit demand the diversion ratio between two carriers depends on their market shares and not on the price coefficient, so which routes look worst is settled by market structure, which is observed, rather than by the calibration, which is assumed. The dollar figures should be read as scaled to an assumption. The ranking should not.

That distinction is what makes stage 9 possible. It compares predictions with outcomes by ordering and by bin, not by level.
