"""Matching, and the configuration the whole pipeline reads its choices from."""
from __future__ import annotations

import numpy as np
import pytest

import config as C
from econ.matching import (balance_table, nearest_neighbour_att,
                           propensity_att_weights, standardised_difference)


@pytest.fixture
def imbalanced(rng):
    """Treated units systematically larger than controls on two covariates."""
    n_t, n_c = 200, 600
    Xt = np.column_stack([rng.normal(1.0, 1.0, n_t), rng.normal(0.6, 1.0, n_t)])
    Xc = np.column_stack([rng.normal(0.0, 1.0, n_c), rng.normal(0.0, 1.0, n_c)])
    X = np.vstack([Xt, Xc])
    t = np.concatenate([np.ones(n_t), np.zeros(n_c)])
    return X, t


class TestNearestNeighbour:
    def test_improves_balance(self, imbalanced):
        X, t = imbalanced
        r = nearest_neighbour_att(X, t, np.ones(len(X)))
        for j in range(X.shape[1]):
            before = abs(standardised_difference(X[:, j], t))
            after = abs(standardised_difference(X[:, j], t, r.weights))
            assert after < before

    def test_treated_keep_their_own_weight(self, imbalanced):
        X, t = imbalanced
        uw = np.abs(np.random.default_rng(0).normal(5, 1, len(X)))
        r = nearest_neighbour_att(X, t, uw)
        matched = (t == 1) & (r.weights > 0)
        assert np.allclose(r.weights[matched], uw[matched])

    def test_the_two_arms_carry_equal_total_weight(self, imbalanced):
        X, t = imbalanced
        r = nearest_neighbour_att(X, t, np.ones(len(X)))
        assert r.weights[t == 1].sum() == pytest.approx(r.weights[t == 0].sum())

    def test_no_control_dominates(self, imbalanced):
        """The failure mode that odds weighting produced: one unit taking it all."""
        X, t = imbalanced
        r = nearest_neighbour_att(X, t, np.ones(len(X)))
        assert r.max_control_weight_share < 0.10
        assert r.n_effective_controls > 50

    def test_matching_bounds_control_weights_where_odds_weighting_does_not(self, rng):
        """The failure this project actually hit, reproduced in miniature.

        With good overlap, inverse-odds weighting is fine and can even retain a
        larger effective sample than matching. The problem appears when overlap
        is poor - which is the airline case, where routes two big carriers both
        served look nothing like routes neither served. There a few controls sit
        at propensities near one, their odds run into the tens, and the control
        arm collapses onto them. Matching cannot do that, because a control is
        only ever weighted by the treated units it stands in for.
        """
        n_t, n_c = 300, 300
        # Treated and control barely overlap: separated by three SDs.
        Xt = rng.normal(3.0, 1.0, (n_t, 2))
        Xc = rng.normal(0.0, 1.0, (n_c, 2))
        X = np.vstack([Xt, Xc])
        t = np.concatenate([np.ones(n_t), np.zeros(n_c)])
        pax = np.abs(rng.lognormal(0, 1.5, len(X)))   # skewed, as traffic is

        odds = propensity_att_weights(X, t)
        ow = odds.weights * pax
        ow_ctrl = ow[t == 0]
        odds_top = ow_ctrl.max() / ow_ctrl.sum()

        nn = nearest_neighbour_att(X, t, pax)

        assert odds_top > 0.25, "expected odds weighting to degenerate here"
        assert nn.max_control_weight_share < odds_top
        # Matching bounds the damage; it does not repair genuinely disjoint
        # support. With the two groups three standard deviations apart, few
        # controls fall inside the caliper and the effective sample is small
        # either way. That is the honest limit of the technique, and it is why
        # the pipeline reports the effective control count rather than only the
        # point estimate.
        assert nn.n_effective_controls < 50

    def test_unmatchable_treated_units_are_counted_not_hidden(self, rng):
        """A treated unit with no comparable control must be dropped and reported."""
        Xc = rng.normal(0, 1, (300, 2))
        Xt = rng.normal(0, 1, (50, 2))
        Xt[:10] += 25.0                       # far outside the control support
        X = np.vstack([Xt, Xc])
        t = np.concatenate([np.ones(50), np.zeros(300)])
        r = nearest_neighbour_att(X, t, np.ones(len(X)))
        assert r.unmatched_treated >= 10
        assert r.matched_treated + r.unmatched_treated == 50

    def test_balance_table_reports_both_columns(self, imbalanced):
        X, t = imbalanced
        r = nearest_neighbour_att(X, t, np.ones(len(X)))
        rows = balance_table(X, t, ["a", "b"], w_before=np.ones(len(X)),
                             w_after=r.weights)
        assert [row["covariate"] for row in rows] == ["a", "b"]
        for row in rows:
            assert abs(row["after"]) < abs(row["before"])

    def test_standardised_difference_is_zero_on_identical_groups(self, rng):
        x = rng.normal(size=400)
        t = np.tile([0, 1], 200)
        assert abs(standardised_difference(x, t)) < 0.2


class TestConfig:
    def test_quarter_sequence_is_complete_and_ordered(self):
        qs = C.quarters()
        assert qs[0] == (C.YEAR_START, C.QUARTER_START)
        assert qs[-1] == (C.YEAR_END, C.QUARTER_END)
        assert len(qs) == 60
        assert qs == sorted(qs)

    def test_qindex_is_monotonic_and_unit_spaced(self):
        idx = [C.qindex(y, q) for y, q in C.quarters()]
        assert np.all(np.diff(idx) == 1)

    def test_every_merger_closes_after_it_is_announced(self):
        for m in C.MERGERS:
            assert m.announce_q < m.close_q
            assert m.close_q <= C.qindex(*m.certificate)
            assert C.qindex(*m.certificate) <= C.qindex(*m.brand_retired)

    def test_every_event_window_fits_inside_the_data(self):
        lo = C.qindex(C.YEAR_START, C.QUARTER_START)
        hi = C.qindex(C.YEAR_END, C.QUARTER_END)
        for m in C.MERGERS:
            assert m.close_q + C.EVENT_MIN >= lo, f"{m.key} pre-window is truncated"
            assert m.close_q + C.EVENT_MAX <= hi, f"{m.key} post-window is truncated"
            assert m.announce_q - C.BASELINE_QUARTERS >= lo

    def test_merger_keys_are_unique(self):
        keys = [m.key for m in C.MERGERS]
        assert len(keys) == len(set(keys))

    def test_carrier_groups_cover_the_merging_carriers(self):
        for m in C.MERGERS:
            for c in m.carriers:
                assert C.carrier_group(c) in {"network", "lowcost"}

    def test_guidelines_thresholds_match_the_2023_text(self):
        assert C.HHI_CONCENTRATED == 1800
        assert C.DELTA_HHI_THRESHOLD == 100
        assert C.SHARE_THRESHOLD == 0.30

    def test_the_inside_share_target_is_in_range(self):
        assert 0.0 < C.INSIDE_SHARE_TARGET < 1.0
        assert all(0.0 < a < 1.0 for a in C.INSIDE_SHARE_TARGET_ALTS)
