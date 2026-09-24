"""Bertrand equilibrium, merger simulation, and the pricing-pressure screens.

The logic a merger simulation runs on is short. Before the merger, each airline
sets fares to maximise its own profit, taking rivals' fares as given. That gives
a first-order condition per product, and since fares and shares are observed and
demand has been estimated, the condition can be solved *backwards* for the one
thing nobody observes: marginal cost. This is the standard trick, and it is
worth being clear about what it assumes. It assumes the observed fares really
are a Nash equilibrium in prices. Where they are not - where fares are set by
long-term contract, or by a carrier not maximising short-run profit - the
recovered cost absorbs the error.

After the merger the two carriers set fares jointly. A fare rise on one of them
now pushes passengers onto a product the same owner holds, so the loss from
raising it is smaller and the profit-maximising fare is higher. Re-solving the
same first-order conditions under the new ownership gives the predicted
post-merger fares, and the difference is the simulated price effect.

Two cheaper screens fall out of the same objects and are what agencies actually
compute first:

    GUPPI  the value of sales diverted to the partner, per dollar of own price:
           the upward pricing pressure a merger creates before any efficiency
    UPP    the same, net of a cost saving the merger is credited with

Both are reported here because the point of the project is to ask whether they
predict what the retrospective finds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nested_logit import NestedLogit


def ownership_matrix(firms: np.ndarray) -> np.ndarray:
    """1 where two products share an owner, 0 otherwise."""
    firms = np.asarray(firms)
    return (firms[:, None] == firms[None, :]).astype(float)


def _foc_matrix(model: NestedLogit, omega: np.ndarray,
                prices: np.ndarray | None = None) -> np.ndarray:
    """A[j, k] = omega[j, k] * d s_k / d p_j, the matrix in the pricing FOC."""
    J = model.jacobian(prices)          # J[j, k] = d s_j / d p_k
    return omega * J.T


def marginal_costs(model: NestedLogit, omega: np.ndarray) -> np.ndarray:
    """Back out marginal cost from the observed prices and shares.

    From  s + A (p - mc) = 0  we get  mc = p + A^-1 s.
    """
    s = model.shares()
    A = _foc_matrix(model, omega)
    return model.prices + np.linalg.solve(A, s)


def margins(model: NestedLogit, omega: np.ndarray) -> np.ndarray:
    """Lerner index implied by the first-order conditions."""
    mc = marginal_costs(model, omega)
    return (model.prices - mc) / model.prices


def equilibrium_prices(model: NestedLogit, omega: np.ndarray, mc: np.ndarray,
                       p0: np.ndarray | None = None, tol: float = 1e-10,
                       maxiter: int = 2000, damping: float = 0.5
                       ) -> tuple[np.ndarray, bool, int]:
    """Solve for the Nash equilibrium in prices under ownership ``omega``.

    Iterates p <- mc - A(p)^-1 s(p) with damping. The map is a contraction near
    the equilibrium for the demand systems used here, but damping is kept
    because undamped iteration overshoots on routes where one carrier holds
    almost the whole market. Returns the prices, whether it converged, and the
    iteration count; callers are expected to check the flag rather than trust
    the prices blindly.
    """
    p = model.prices.copy() if p0 is None else np.asarray(p0, float).copy()
    for it in range(1, maxiter + 1):
        s = model.shares(p)
        A = _foc_matrix(model, omega, p)
        try:
            step = mc - np.linalg.solve(A, s)
        except np.linalg.LinAlgError:
            return p, False, it
        if not np.all(np.isfinite(step)):
            return p, False, it
        # Fares cannot go below cost in this model; clamping keeps a bad
        # iterate from walking the solver into a region with no equilibrium.
        step = np.maximum(step, mc * 1.000001)
        newp = (1 - damping) * p + damping * step
        if np.max(np.abs(newp - p)) < tol * max(1.0, np.max(np.abs(p))):
            return newp, True, it
        p = newp
    return p, False, maxiter


# ------------------------------------------------------------------- screens
def guppi(model: NestedLogit, mc: np.ndarray, firms: np.ndarray,
          merging: tuple, ) -> np.ndarray:
    """Gross upward pricing pressure for every product of the merging firms.

    For product j owned by one merging party, GUPPI_j is the value of the sales
    diverted to the *other* party when j raises its fare, expressed as a
    fraction of j's own fare. Products not owned by a merging party get zero.
    """
    firms = np.asarray(firms)
    D = model.diversion()
    p = model.prices
    margin_dollars = p - mc
    out = np.zeros_like(p)
    a, b = merging
    for j in range(len(p)):
        if firms[j] == a:
            partner = firms == b
        elif firms[j] == b:
            partner = firms == a
        else:
            continue
        out[j] = float(np.sum(D[j, partner] * margin_dollars[partner])) / p[j]
    return out


def upp(model: NestedLogit, mc: np.ndarray, firms: np.ndarray,
        merging: tuple, efficiency: float = 0.10) -> np.ndarray:
    """Upward pricing pressure, in dollars, net of an efficiency credit.

    ``efficiency`` is the proportional marginal-cost saving the merger is
    credited with; 10% is the figure Farrell and Shapiro use as a default
    standard, not an estimate of what any particular merger achieved.
    """
    firms = np.asarray(firms)
    D = model.diversion()
    margin_dollars = model.prices - mc
    out = np.zeros_like(model.prices)
    a, b = merging
    for j in range(len(model.prices)):
        if firms[j] == a:
            partner = firms == b
        elif firms[j] == b:
            partner = firms == a
        else:
            continue
        out[j] = float(np.sum(D[j, partner] * margin_dollars[partner])) \
            - efficiency * mc[j]
    return out


def cmcr(model: NestedLogit, mc: np.ndarray, firms: np.ndarray,
         merging: tuple) -> np.ndarray:
    """Compensating marginal cost reduction.

    The proportional cost saving that would have to arrive for the merged firm
    to leave its fares where they were. A large number means the merger would
    need implausible efficiencies to be harmless.
    """
    firms = np.asarray(firms)
    D = model.diversion()
    p, out = model.prices, np.zeros_like(model.prices)
    a, b = merging
    for j in range(len(p)):
        if firms[j] == a:
            partner = firms == b
        elif firms[j] == b:
            partner = firms == a
        else:
            continue
        # Symmetric two-product form; with several products per party the
        # reported value is the own-product term, which is the standard
        # first-order approximation.
        num = float(np.sum(D[j, partner] * (p[partner] - mc[partner])))
        # A non-positive recovered marginal cost means the first-order condition
        # did not describe this market; the statistic is undefined rather than
        # infinite, and callers drop it.
        out[j] = num / mc[j] if mc[j] > 0 else np.nan
    return out


# ---------------------------------------------------------------- simulation
@dataclass
class MergerSimulation:
    prices_pre: np.ndarray
    prices_post: np.ndarray
    shares_pre: np.ndarray
    shares_post: np.ndarray
    marginal_costs: np.ndarray
    margins_pre: np.ndarray
    guppi: np.ndarray
    upp: np.ndarray
    cmcr: np.ndarray
    diversion_to_partner: np.ndarray
    converged: bool
    iterations: int
    cv_per_consumer: float          # dollars of harm per potential passenger
    market_size: float

    @property
    def price_change(self) -> np.ndarray:
        return self.prices_post - self.prices_pre

    @property
    def pct_price_change(self) -> np.ndarray:
        return self.prices_post / self.prices_pre - 1.0

    def weighted_price_change(self) -> float:
        """Share-weighted average percentage fare change across all carriers."""
        w = self.shares_pre / self.shares_pre.sum()
        return float(np.sum(w * self.pct_price_change))

    @property
    def consumer_harm(self) -> float:
        """Total dollars of harm in this market, per period."""
        return self.cv_per_consumer * self.market_size


def simulate_merger(model: NestedLogit, firms: np.ndarray, merging: tuple,
                    market_size: float, efficiency: float = 0.10,
                    cost_saving: float = 0.0) -> MergerSimulation:
    """Run the whole exercise for one market.

    ``cost_saving`` is a proportional reduction applied to the merging parties'
    marginal costs in the post-merger equilibrium. It defaults to zero, so the
    headline simulation asks what happens with no efficiencies at all; the
    sensitivity to that assumption is reported separately.
    """
    firms = np.asarray(firms)
    omega_pre = ownership_matrix(firms)
    mc = marginal_costs(model, omega_pre)

    firms_post = np.where(firms == merging[1], merging[0], firms)
    omega_post = ownership_matrix(firms_post)
    mc_post = mc.copy()
    if cost_saving:
        party = (firms == merging[0]) | (firms == merging[1])
        mc_post[party] = mc[party] * (1.0 - cost_saving)

    p_post, ok, iters = equilibrium_prices(model, omega_post, mc_post)

    D = model.diversion()
    to_partner = np.zeros(len(firms))
    a, b = merging
    for j in range(len(firms)):
        if firms[j] == a:
            to_partner[j] = D[j, firms == b].sum()
        elif firms[j] == b:
            to_partner[j] = D[j, firms == a].sum()

    return MergerSimulation(
        prices_pre=model.prices.copy(), prices_post=p_post,
        shares_pre=model.shares(), shares_post=model.shares(p_post),
        marginal_costs=mc, margins_pre=(model.prices - mc) / model.prices,
        guppi=guppi(model, mc, firms, merging),
        upp=upp(model, mc, firms, merging, efficiency),
        cmcr=cmcr(model, mc, firms, merging),
        diversion_to_partner=to_partner, converged=ok, iterations=iters,
        cv_per_consumer=model.compensating_variation(p_post),
        market_size=market_size,
    )
