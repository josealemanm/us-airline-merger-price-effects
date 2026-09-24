"""Econometric machinery for the merger analysis.

Written from numpy and scipy rather than assembled from a package, because the
point of the exercise is to know exactly what every standard error is doing.
Every estimator here is checked in ``tests/`` against a closed-form answer or
against an established implementation.
"""

from .concentration import (
    concentration_ratio, delta_hhi, hhi, num_effective_competitors,
    screen_merger, shares_from_quantities,
)
from .fe_ols import FEResult, factorize, feols
from .iv import IVResult, iv2sls
from .did import (
    ATTGT, EventStudy, Stack, att_gt, event_study, event_time_dummies,
    stacked_did, stacked_event_study,
)
from .bacon import BaconResult, bacon_decompose
from .nested_logit import NestedLogit, berry_delta, market_from_estimates, within_group_shares
from .bertrand import (
    MergerSimulation, cmcr, equilibrium_prices, guppi, marginal_costs,
    margins, ownership_matrix, simulate_merger, upp,
)
from .inference import (
    BootstrapResult, RandomizationResult, cluster_bootstrap_ci,
    randomization_test, wild_cluster_bootstrap,
)

__all__ = [
    "concentration_ratio", "delta_hhi", "hhi", "num_effective_competitors",
    "screen_merger", "shares_from_quantities",
    "FEResult", "factorize", "feols", "IVResult", "iv2sls",
    "ATTGT", "EventStudy", "Stack", "att_gt", "event_study",
    "event_time_dummies", "stacked_did", "stacked_event_study",
    "BaconResult", "bacon_decompose",
    "NestedLogit", "berry_delta", "market_from_estimates", "within_group_shares",
    "MergerSimulation", "cmcr", "equilibrium_prices", "guppi", "marginal_costs",
    "margins", "ownership_matrix", "simulate_merger", "upp",
    "BootstrapResult", "RandomizationResult", "cluster_bootstrap_ci",
    "randomization_test", "wild_cluster_bootstrap",
]
