# Demand estimation

Estimated by `src/07_estimate_demand.py` on 2,195,079 carrier-market-quarter observations covering 17,071 routes, with market, quarter and carrier fixed effects and standard errors clustered on the route.

This stage did not produce what it was built to produce. What follows is the test that found that out, the estimate that survived it, and the assumption the simulation runs on instead.

## Testing the instruments one at a time

Throwing five instruments into one regression and reading the overidentification test is the usual practice and it hides exactly this problem. Run separately, in the plain logit where only the fare is endogenous, each instrument gives its own answer, and a price coefficient of the wrong sign is a failure no first-stage F can excuse.

| instrument | what it is | first-stage F | implied alpha | median elasticity | verdict |
|---|---|---|---|---|---|
| `none (OLS)` | no instrument | - | +0.183 | -0.46 | - |
| `z_fuel_distance` | national jet fuel price x log route distance | 23 | +0.503 | -1.25 | survives |
| `z_hausman` | the carrier's mean fare in its other markets, same quarter | 6,583 | -0.252 | +0.63 | **wrong sign** |
| `z_n_rivals` | number of rival carriers on the route | 1,513 | -1.845 | +4.60 | **wrong sign** |
| `z_n_nest_rivals` | number of rivals inside the carrier's own nest | 1,176 | -3.273 | +8.16 | **wrong sign** |
| `z_rival_nonstop` | summed nonstop share of competitors | 707 | +0.794 | -1.98 | survives |

Three of the five return a price coefficient of the wrong sign: `z_hausman`, `z_n_rivals`, `z_n_nest_rivals`. They fail together and for one reason. Entry is a response to demand: a carrier adds a route when it expects the route to sell. A positive demand shock therefore brings in rivals, rivals push the fare down, and an instrument built on rival counts carries the very shock it was supposed to exclude. Where the Hausman instrument is among them, it fails for the familiar reason - a carrier's demand shocks run across its own network, so its fares elsewhere are not independent of demand here.

Of the two that come back with the right sign, only the cost instrument has an exclusion restriction worth defending. `z_rival_nonstop` survives this particular test but is still a market-structure instrument, and the entry argument applies to it whether or not it happens to fail here. It is not used.

The cost instrument is much the weakest in the first stage - F = 23, against 707 for the next weakest and thousands for some of the rest - and the only one whose validity survives the argument. That trade is the one every applied paper is really making, and it is usually made quietly.

## The estimate that survives

| | alpha, per $100 | implied median own-price elasticity |
|---|---|---|
| OLS | 0.1827 | -0.46 |
| IV, cost instrument | 0.5035 | -1.25 |

Instrumenting moves the coefficient away from zero, from 0.183 to 0.503, which is the direction the endogeneity argument predicts: carriers charge more where demand is strong, so the uninstrumented regression sees fare and quantity moving together and reads that as insensitivity.

## And the respect in which it fails

> The estimate implies a median own-price elasticity of **-1.25**.

A profit-maximising firm never prices where its own demand is inelastic. At an elasticity above -1 the Bertrand first-order condition returns a marginal cost below zero, and a merger simulation built on it would be arithmetic performed on an impossible object.

Three things could produce this and the data cannot separate them. The fare is a quarterly average over every fare class a carrier sold, and an average over a bundle is less elastic than any ticket in it. DB1B is a ten per cent sample, so route-level average fares carry sampling noise, and measurement error attenuates. And the one valid instrument is weak enough that the correction it applies may simply be incomplete.

The nested logit with all five instruments returns alpha = -0.022 and sigma = 0.438. Since four of those instruments have just failed their own test, that pair is reported here and used nowhere.

## What the simulation runs on instead

The price coefficient is calibrated to a target own-price elasticity of **-2.5**, within the range published estimates of US airline firm-level demand elasticity occupy, and the simulation is re-run at elasticities either side of it.

| | value |
|---|---|
| calibrated alpha, per dollar | 0.01003 |
| implied median elasticity | -2.5 |
| ratio to the estimated alpha | 2.0x |
| nesting parameter sigma | 0.0 (plain logit) |

Sigma is set to zero rather than estimated. Every instrument for the within-nest share is one of the market-structure instruments that has just failed, so there is nothing credible left to identify it with. Plain logit is the weaker model - it forces diversion to follow market shares - and stage 8 re-runs at sigma of 0.25 and 0.5 to show what that costs.

This is a calibration and it is labelled as one. It is also what an agency does when the demand estimation on the record will not support a simulation: fall back on a calibrated margin and put the sensitivity in the record. The consequence that matters is reported in stage 8. Under logit demand, diversion ratios depend on market shares alone and not on alpha at all, so the *ranking* of routes by predicted harm barely moves with the calibration, while the *level* of predicted harm scales with it almost proportionally. The question this project asks - whether the screens pick the right routes - turns on the ranking.
