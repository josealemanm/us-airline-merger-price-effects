"""Nested logit demand: inversion, substitution, diversion, consumer surplus.

A merger simulation is only as good as its answer to one question: when an
airline raises its fare, where do the passengers who leave actually go? If they
go to the other merging airline, the merged firm recaptures them and the
incentive to raise the fare is large. If they go to a third carrier, or stop
flying, the incentive is small. That fraction is the diversion ratio, and it is
the single number that drives everything downstream.

Plain logit answers the question badly, because it forces diversion to be
proportional to market share. It implies that a passenger priced off a
Spirit ticket is as likely to buy a first-class Delta seat as their shares
suggest, which nobody believes. The nested logit relaxes that by letting
products inside a nest be closer substitutes for each other than for products
outside it, at the cost of one extra parameter.

The model here is the standard one. Consumer i on route m picks the option with
the highest

    u_ij = x_j'b - a*p_j + xi_j + zeta_ig + (1 - sigma)*eps_ij

where the nest term zeta_ig is common to every product in nest g and sigma
governs how tightly the nest binds. sigma = 0 collapses to plain logit; sigma
approaching 1 makes products inside a nest perfect substitutes. The outside
option - not flying, driving, not making the trip - sits alone with utility
normalised to zero, which is what makes the level of demand identified at all.

Berry (1994) showed this inverts in closed form, which is why it is estimable on
sixty quarters of route data without a nested fixed-point solver:

    ln(s_j) - ln(s_0) = x_j'b - a*p_j + sigma*ln(s_j|g) + xi_j

That is a linear regression. Both p_j and ln(s_j|g) are endogenous, so it is a
linear regression that has to be run with instruments; see ``iv.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ inversion
def within_group_shares(shares: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Share of each product within its own nest."""
    shares = np.asarray(shares, float)
    groups = np.asarray(groups)
    out = np.empty_like(shares)
    for g in np.unique(groups):
        m = groups == g
        tot = shares[m].sum()
        out[m] = shares[m] / tot if tot > 0 else 0.0
    return out


def berry_delta(shares: np.ndarray, outside_share: float) -> np.ndarray:
    """The left-hand side of the inversion, ln(s_j) - ln(s_0)."""
    return np.log(np.asarray(shares, float)) - np.log(outside_share)


# -------------------------------------------------------------------- the model
@dataclass(frozen=True)
class NestedLogit:
    """One market: products, their prices, their nests, and the parameters.

    ``alpha`` is the magnitude of the price coefficient, so it is positive and
    higher means more price-sensitive. ``sigma`` is the nesting parameter, one
    per nest or a scalar shared by all of them.
    """
    delta: np.ndarray        # mean utility excluding price: x'b + xi
    prices: np.ndarray
    groups: np.ndarray
    alpha: float
    sigma: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.sigma < 1.0):
            raise ValueError(f"sigma must be in [0, 1), got {self.sigma}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    # -- shares ------------------------------------------------------------
    def _utilities(self, prices: np.ndarray | None = None) -> np.ndarray:
        p = self.prices if prices is None else np.asarray(prices, float)
        return self.delta - self.alpha * p

    def shares(self, prices: np.ndarray | None = None) -> np.ndarray:
        """Market shares including the outside good, which is not returned."""
        v = self._utilities(prices)
        s = np.empty_like(v)
        one_minus = 1.0 - self.sigma
        # exp() of a large utility overflows; subtracting the max inside each
        # nest is the usual log-sum-exp guard and cancels out of the ratios.
        iv = {}
        for g in np.unique(self.groups):
            m = self.groups == g
            z = v[m] / one_minus
            zmax = z.max()
            e = np.exp(z - zmax)
            denom = e.sum()
            s[m] = e / denom                      # within-nest shares
            iv[g] = one_minus * (np.log(denom) + zmax)   # inclusive value
        ivs = np.array(list(iv.values()))
        ivmax = max(ivs.max(), 0.0)
        num = np.exp(ivs - ivmax)
        outside = np.exp(-ivmax)
        total = outside + num.sum()
        for i, g in enumerate(iv):
            m = self.groups == g
            s[m] *= num[i] / total
        return s

    def outside_share(self, prices: np.ndarray | None = None) -> float:
        return float(1.0 - self.shares(prices).sum())

    def inclusive_value(self, prices: np.ndarray | None = None) -> float:
        """log(1 + sum over nests of exp(IV_g)), the consumer-surplus core."""
        v = self._utilities(prices)
        one_minus = 1.0 - self.sigma
        ivs = []
        for g in np.unique(self.groups):
            m = self.groups == g
            z = v[m] / one_minus
            zmax = z.max()
            ivs.append(one_minus * (np.log(np.exp(z - zmax).sum()) + zmax))
        ivs = np.array(ivs)
        mx = max(ivs.max(), 0.0)
        return float(mx + np.log(np.exp(-mx) + np.exp(ivs - mx).sum()))

    # -- substitution ------------------------------------------------------
    def jacobian(self, prices: np.ndarray | None = None) -> np.ndarray:
        """``J[j, k] = d s_j / d p_k``.

        Own-price terms are negative; cross-price terms are positive, larger
        inside a nest than across nests. At sigma = 0 this reduces to the plain
        logit derivatives, which is the first thing the tests check.
        """
        s = self.shares(prices)
        sg = within_group_shares(s, self.groups)
        a, sig = self.alpha, self.sigma
        same = self.groups[:, None] == self.groups[None, :]
        # d s_j / d p_k for k != j
        J = a * s[:, None] * (np.where(same, sig / (1 - sig) * sg[None, :], 0.0)
                              + s[None, :])
        own = -a * s * (1.0 / (1 - sig) - sig / (1 - sig) * sg - s)
        np.fill_diagonal(J, own)
        return J

    def elasticities(self, prices: np.ndarray | None = None) -> np.ndarray:
        """``E[j, k] = d ln s_j / d ln p_k``."""
        p = self.prices if prices is None else np.asarray(prices, float)
        s = self.shares(prices)
        return self.jacobian(prices) * p[None, :] / s[:, None]

    def diversion(self, prices: np.ndarray | None = None) -> np.ndarray:
        """``D[j, k]``: share of the sales j loses to a price rise that go to k.

        Rows sum to less than one; the remainder leaves for the outside good.
        """
        J = self.jacobian(prices)
        own = np.diag(J).copy()
        D = -J.T / own[None, :]     # D[j, k] = -(ds_k/dp_j) / (ds_j/dp_j)
        D = D.T
        np.fill_diagonal(D, 0.0)
        return D

    def diversion_to_outside(self, prices: np.ndarray | None = None) -> np.ndarray:
        return 1.0 - self.diversion(prices).sum(axis=1)

    # -- welfare -----------------------------------------------------------
    def consumer_surplus(self, prices: np.ndarray | None = None) -> float:
        """Expected surplus per consumer, in dollars, up to a constant.

        Only differences are meaningful, which is all the harm calculation uses.
        """
        return self.inclusive_value(prices) / self.alpha

    def compensating_variation(self, new_prices: np.ndarray) -> float:
        """Dollars per consumer needed to undo a price change. Positive = harm."""
        return self.consumer_surplus(self.prices) - self.consumer_surplus(new_prices)


# ------------------------------------------------- building one from estimates
def market_from_estimates(shares: np.ndarray, prices: np.ndarray,
                          groups: np.ndarray, alpha: float, sigma: float,
                          outside_share: float) -> NestedLogit:
    """Recover delta so the model reproduces the observed shares exactly.

    Estimation gives alpha and sigma; the residual xi is whatever makes the
    model fit this market's data. Inverting rather than predicting is what keeps
    the simulation anchored to the fares and shares actually observed.
    """
    shares = np.asarray(shares, float)
    sg = within_group_shares(shares, groups)
    # Inverting the share equation: delta = ln s_j - ln s_0 - sigma ln s_j|g + a p
    delta = (np.log(shares) - np.log(outside_share) - sigma * np.log(sg)
             + alpha * np.asarray(prices, float))
    m = NestedLogit(delta=delta, prices=np.asarray(prices, float),
                    groups=np.asarray(groups), alpha=alpha, sigma=sigma)
    check = m.shares()
    if not np.allclose(check, shares, rtol=1e-6, atol=1e-10):
        raise AssertionError("inversion failed to reproduce the observed shares")
    return m
