"""Demand and merger simulation, against closed forms.

Nested logit at sigma = 0 is plain logit, and plain logit has closed-form
elasticities, diversion ratios and single-product margins. Those identities are
the tests: if the code reproduces them exactly, the algebra is right, and the
nested cases can then be checked for the qualitative behaviour nesting is
supposed to produce.
"""
from __future__ import annotations

import numpy as np
import pytest

from econ.bertrand import (cmcr, equilibrium_prices, guppi, marginal_costs,
                           margins, ownership_matrix, simulate_merger, upp)
from econ.concentration import (concentration_ratio, delta_hhi, hhi,
                                num_effective_competitors, screen_merger,
                                shares_from_quantities)
from econ.nested_logit import (NestedLogit, market_from_estimates,
                               within_group_shares)


@pytest.fixture
def logit_market():
    return NestedLogit(delta=np.array([2.0, 1.5, 1.0, 0.5]),
                       prices=np.array([300.0, 250.0, 200.0, 180.0]),
                       groups=np.array([0, 0, 1, 1]), alpha=0.01, sigma=0.0)


@pytest.fixture
def nested_market(logit_market):
    return NestedLogit(delta=logit_market.delta, prices=logit_market.prices,
                       groups=logit_market.groups, alpha=0.01, sigma=0.7)


class TestShares:
    def test_shares_are_a_probability_distribution(self, logit_market):
        s = logit_market.shares()
        assert np.all(s > 0)
        assert s.sum() < 1.0
        assert logit_market.outside_share() == pytest.approx(1 - s.sum())

    def test_within_group_shares_sum_to_one_per_group(self, nested_market):
        s = nested_market.shares()
        sg = within_group_shares(s, nested_market.groups)
        for g in np.unique(nested_market.groups):
            assert sg[nested_market.groups == g].sum() == pytest.approx(1.0)

    def test_raising_a_price_lowers_that_share(self, logit_market):
        s0 = logit_market.shares()
        p = logit_market.prices.copy()
        p[0] += 50
        s1 = logit_market.shares(p)
        assert s1[0] < s0[0]
        assert np.all(s1[1:] > s0[1:])      # rivals gain

    def test_no_overflow_at_extreme_utilities(self):
        m = NestedLogit(delta=np.array([900.0, -900.0]),
                        prices=np.array([100.0, 100.0]),
                        groups=np.array([0, 0]), alpha=0.01, sigma=0.5)
        s = m.shares()
        assert np.all(np.isfinite(s))
        assert s.sum() <= 1.0

    def test_sigma_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError):
            NestedLogit(np.array([1.0]), np.array([100.0]), np.array([0]),
                        alpha=0.01, sigma=1.0)

    def test_non_positive_alpha_is_rejected(self):
        with pytest.raises(ValueError):
            NestedLogit(np.array([1.0]), np.array([100.0]), np.array([0]),
                        alpha=0.0, sigma=0.0)


class TestClosedForms:
    def test_logit_diversion(self, logit_market):
        """D_jk = s_k / (1 - s_j) when sigma is zero."""
        s = logit_market.shares()
        want = s[None, :] / (1 - s[:, None])
        np.fill_diagonal(want, 0.0)
        assert np.allclose(logit_market.diversion(), want)

    def test_logit_own_elasticity(self, logit_market):
        """e_jj = -alpha * p_j * (1 - s_j)."""
        s = logit_market.shares()
        want = -logit_market.alpha * logit_market.prices * (1 - s)
        assert np.allclose(np.diag(logit_market.elasticities()), want)

    def test_diversion_rows_account_for_everything(self, nested_market):
        D = nested_market.diversion()
        assert np.allclose(D.sum(axis=1) + nested_market.diversion_to_outside(),
                           1.0)

    def test_single_product_lerner(self, logit_market):
        """Margin = 1 / (alpha * p * (1 - s)) for a single-product firm."""
        om = ownership_matrix(np.arange(4))
        s = logit_market.shares()
        want = 1.0 / (logit_market.alpha * logit_market.prices * (1 - s))
        assert np.allclose(margins(logit_market, om), want)

    def test_inversion_round_trips(self, nested_market):
        s = nested_market.shares()
        rebuilt = market_from_estimates(s, nested_market.prices,
                                        nested_market.groups,
                                        nested_market.alpha, nested_market.sigma,
                                        nested_market.outside_share())
        assert np.allclose(rebuilt.shares(), s)


class TestNesting:
    def test_nesting_raises_within_group_diversion(self, logit_market, nested_market):
        assert nested_market.diversion()[0, 1] > logit_market.diversion()[0, 1]

    def test_nesting_lowers_cross_group_diversion(self, logit_market, nested_market):
        assert nested_market.diversion()[0, 2] < logit_market.diversion()[0, 2]

    def test_sigma_zero_equals_plain_logit(self, logit_market):
        same = NestedLogit(logit_market.delta, logit_market.prices,
                           logit_market.groups, logit_market.alpha, 0.0)
        assert np.allclose(same.jacobian(), logit_market.jacobian())


class TestMergerSimulation:
    def test_marginal_costs_are_below_price(self, logit_market):
        om = ownership_matrix(np.arange(4))
        mc = marginal_costs(logit_market, om)
        assert np.all(mc < logit_market.prices)
        assert np.all(mc > 0)

    def test_equilibrium_reproduces_observed_prices(self, logit_market):
        """Re-solving with the pre-merger ownership must return to the data."""
        om = ownership_matrix(np.arange(4))
        mc = marginal_costs(logit_market, om)
        p, ok, _ = equilibrium_prices(logit_market, om, mc)
        assert ok
        assert np.allclose(p, logit_market.prices, rtol=1e-6)

    def test_merger_raises_the_parties_prices(self, logit_market):
        sim = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6)
        assert sim.converged
        assert np.all(sim.pct_price_change[:2] > 0)

    def test_rivals_respond_upward_but_by_less(self, logit_market):
        sim = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6)
        assert np.all(sim.pct_price_change[2:] > 0)
        assert sim.pct_price_change[2:].max() < sim.pct_price_change[:2].min()

    def test_merger_to_monopoly_raises_prices_more(self, logit_market):
        partial = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6)
        duo = NestedLogit(logit_market.delta[:2], logit_market.prices[:2],
                          logit_market.groups[:2], logit_market.alpha, 0.0)
        mono = simulate_merger(duo, np.arange(2), (0, 1), 1e6)
        assert mono.pct_price_change.mean() > partial.pct_price_change[:2].mean()

    def test_efficiencies_soften_the_price_rise(self, logit_market):
        no_eff = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6,
                                 cost_saving=0.0)
        with_eff = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6,
                                   cost_saving=0.15)
        assert with_eff.weighted_price_change() < no_eff.weighted_price_change()

    def test_consumer_harm_is_positive_and_scales_with_the_market(self, logit_market):
        small = simulate_merger(logit_market, np.arange(4), (0, 1), 1e5)
        big = simulate_merger(logit_market, np.arange(4), (0, 1), 1e6)
        assert small.cv_per_consumer > 0
        assert big.consumer_harm == pytest.approx(10 * small.consumer_harm)

    def test_a_merger_of_distant_products_does_less(self, logit_market):
        """Merging across nests should bite less than merging within one."""
        m = NestedLogit(logit_market.delta, logit_market.prices,
                        logit_market.groups, logit_market.alpha, 0.7)
        within = simulate_merger(m, np.arange(4), (0, 1), 1e6)
        across = simulate_merger(m, np.arange(4), (0, 2), 1e6)
        assert within.guppi.max() > across.guppi.max()


class TestScreens:
    def test_guppi_is_zero_for_non_merging_firms(self, logit_market):
        om = ownership_matrix(np.arange(4))
        mc = marginal_costs(logit_market, om)
        g = guppi(logit_market, mc, np.arange(4), (0, 1))
        assert np.allclose(g[2:], 0.0)
        assert np.all(g[:2] > 0)

    def test_upp_is_guppi_less_the_efficiency_credit(self, logit_market):
        om = ownership_matrix(np.arange(4))
        mc = marginal_costs(logit_market, om)
        g = guppi(logit_market, mc, np.arange(4), (0, 1))
        u0 = upp(logit_market, mc, np.arange(4), (0, 1), efficiency=0.0)
        u1 = upp(logit_market, mc, np.arange(4), (0, 1), efficiency=0.10)
        assert np.allclose(u0[:2], g[:2] * logit_market.prices[:2])
        assert np.all(u1[:2] < u0[:2])

    def test_cmcr_is_positive_for_the_parties(self, logit_market):
        om = ownership_matrix(np.arange(4))
        mc = marginal_costs(logit_market, om)
        c = cmcr(logit_market, mc, np.arange(4), (0, 1))
        assert np.all(c[:2] > 0)


class TestConcentration:
    def test_monopoly_is_ten_thousand(self):
        assert hhi(np.array([1.0])) == pytest.approx(10_000.0)

    def test_n_equal_firms(self):
        for n in (2, 4, 10):
            assert hhi(np.ones(n) / n) == pytest.approx(10_000.0 / n)

    def test_delta_is_twice_the_product_of_shares(self):
        a, b = 0.3, 0.2
        assert delta_hhi(a, b) == pytest.approx(2 * 100 * a * 100 * b)

    def test_delta_matches_recomputing_the_index(self):
        s = np.array([0.4, 0.3, 0.2, 0.1])
        merged = np.array([0.7, 0.2, 0.1])
        assert hhi(merged) - hhi(s) == pytest.approx(delta_hhi(0.4, 0.3))

    def test_shares_from_quantities(self):
        assert np.allclose(shares_from_quantities(np.array([1.0, 3.0])),
                           [0.25, 0.75])

    def test_concentration_ratio_and_effective_count(self):
        s = np.array([0.4, 0.3, 0.2, 0.09, 0.01])
        assert concentration_ratio(s, 4) == pytest.approx(0.99)
        assert num_effective_competitors(s, floor=0.05) == 4

    def test_guidelines_screen_flags_a_concentrated_merger(self):
        s = np.array([0.45, 0.35, 0.20])
        sc = screen_merger(s, 0, 1)
        assert sc.highly_concentrated
        assert sc.presumption_hhi
        assert sc.presumption_share
        assert sc.flagged

    def test_guidelines_screen_clears_a_trivial_merger(self):
        s = np.concatenate([[0.30], np.full(70, 0.01)])
        s = s / s.sum()
        sc = screen_merger(s, 1, 2)
        assert not sc.flagged
