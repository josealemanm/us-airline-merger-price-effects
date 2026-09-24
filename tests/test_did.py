"""Difference-in-differences: the decomposition identity and recovery of a
known effect from data where the answer is built in.
"""
from __future__ import annotations

import numpy as np
import pytest

from econ.bacon import bacon_decompose
from econ.did import Stack, att_gt, event_time_dummies, stacked_did, stacked_event_study
from econ.fe_ols import factorize
from econ.inference import wild_cluster_bootstrap


class TestBaconDecomposition:
    def test_weights_sum_to_one(self, staggered):
        r = bacon_decompose(staggered["y"], staggered["unit"],
                            staggered["period"], staggered["gvar"],
                            never_sentinel=staggered["never"])
        assert np.isclose(sum(c.weight for c in r.components), 1.0)

    def test_components_reproduce_the_regression(self, staggered):
        """The identity the whole decomposition rests on."""
        r = bacon_decompose(staggered["y"], staggered["unit"],
                            staggered["period"], staggered["gvar"],
                            never_sentinel=staggered["never"])
        assert np.isclose(r.reconstructed, r.twfe, atol=1e-8)

    def test_every_comparison_type_appears(self, staggered):
        r = bacon_decompose(staggered["y"], staggered["unit"],
                            staggered["period"], staggered["gvar"],
                            never_sentinel=staggered["never"])
        kinds = {c.kind for c in r.components}
        assert kinds == {"treated_vs_never", "earlier_vs_later",
                         "later_vs_earlier"}

    def test_dynamic_effects_contaminate_the_forbidden_comparisons(self, staggered):
        """With an effect that grows, already-treated controls bias downward.

        This is the whole reason the project does not report the naive
        estimator: the contaminated comparisons carry a smaller effect than the
        clean ones, and they drag the average with them.
        """
        r = bacon_decompose(staggered["y"], staggered["unit"],
                            staggered["period"], staggered["gvar"],
                            never_sentinel=staggered["never"])
        kinds = r.by_kind()
        assert r.forbidden_weight() > 0.05
        assert kinds["later_vs_earlier"][1] < kinds["treated_vs_never"][1]

    def test_no_contamination_without_staggering(self, rng):
        """One treatment date means no timing pairs, so nothing is forbidden."""
        T = 20
        rows, uid = [], 0
        for g, count in {10: 50, 10_000: 50}.items():
            for _ in range(count):
                au = rng.normal(0, 1)
                for t in range(T):
                    eff = 0.4 if (g < 1000 and t >= g) else 0.0
                    rows.append((uid, t, g, au + eff + rng.normal(0, 0.2)))
                uid += 1
        u, t, g, y = (np.array(a) for a in zip(*rows))
        r = bacon_decompose(y, u, t, g, never_sentinel=10_000)
        assert r.forbidden_weight() == pytest.approx(0.0, abs=1e-12)
        assert r.twfe == pytest.approx(0.4, abs=0.05)


class TestEventStudy:
    def test_reference_period_is_omitted(self):
        et = np.array([-2, -1, 0, 1])
        tr = np.ones(4)
        _, names, ks = event_time_dummies(et, tr, -2, 1, ref=-1)
        assert "k-1" not in names
        assert set(ks) == {-2, 0, 1}

    def test_ends_are_binned(self):
        et = np.array([-99, 0, 99])
        tr = np.ones(3)
        D, names, _ = event_time_dummies(et, tr, -2, 2, ref=-1, bin_ends=True)
        assert D[0, names.index("k-2")] == 1.0
        assert D[2, names.index("k+2")] == 1.0

    def test_controls_get_zero_in_every_column(self):
        et = np.array([0, 0])
        tr = np.array([1.0, 0.0])
        D, _, _ = event_time_dummies(et, tr, -1, 1, ref=-1)
        assert np.all(D[1] == 0.0)


def make_stack(rng, effect=0.05, n_treat=120, n_ctrl=120, T=16, close=8):
    """A single-event stack where the true post-period effect is ``effect``."""
    rows = []
    for i in range(n_treat + n_ctrl):
        treated = i < n_treat
        au = rng.normal(0, 0.5)
        for t in range(T):
            post = t >= close
            y = au + 0.01 * t + (effect if treated and post else 0.0) \
                + rng.normal(0, 0.1)
            rows.append(("E", f"u{i}", t, t - close, float(treated),
                         float(post), y, 1.0))
    ev, u, p, k, tr, po, y, w = zip(*rows)
    return Stack(event=np.array(ev), unit=np.array(u), period=np.array(p),
                 event_time=np.array(k), treated=np.array(tr),
                 post=np.array(po), y=np.array(y), weight=np.array(w),
                 extras={})


class TestStackedEstimator:
    def test_recovers_a_known_effect(self, rng):
        st = make_stack(rng, effect=0.05)
        r = stacked_did(st)
        assert r.params[0] == pytest.approx(0.05, abs=0.01)
        lo, hi = r.conf_int()[0]
        assert lo < 0.05 < hi

    def test_finds_nothing_when_there_is_nothing(self, rng):
        st = make_stack(rng, effect=0.0)
        r = stacked_did(st)
        lo, hi = r.conf_int()[0]
        assert lo < 0.0 < hi

    def test_event_study_pre_periods_are_flat(self, rng):
        """No pre-trend when the data has none.

        The magnitude bound is stated relative to the true effect rather than as
        an absolute number: what matters is that pre-period movement is small
        next to the thing being measured, and the joint test is the real check.
        """
        effect = 0.05
        st = make_stack(rng, effect=effect)
        ev = stacked_event_study(st, kmin=-6, kmax=6)
        pre = ev.coef[ev.ks < -1]
        assert np.max(np.abs(pre)) < 0.5 * effect
        assert ev.pretrend_p > 0.01

    def test_event_study_post_periods_find_the_effect(self, rng):
        st = make_stack(rng, effect=0.05)
        ev = stacked_event_study(st, kmin=-6, kmax=6)
        post = ev.coef[ev.ks >= 0]
        assert np.mean(post) == pytest.approx(0.05, abs=0.015)

    def test_event_study_detects_a_pre_trend_that_is_there(self, rng):
        """A design that is genuinely broken should fail the pre-trend test."""
        rows = []
        for i in range(240):
            treated = i < 120
            au = rng.normal(0, 0.5)
            for t in range(16):
                trend = 0.02 * t if treated else 0.0   # diverging beforehand
                rows.append(("E", f"u{i}", t, t - 8, float(treated),
                             float(t >= 8), au + trend + rng.normal(0, 0.1), 1.0))
        ev_, u, p, k, tr, po, y, w = zip(*rows)
        st = Stack(event=np.array(ev_), unit=np.array(u), period=np.array(p),
                   event_time=np.array(k), treated=np.array(tr),
                   post=np.array(po), y=np.array(y), weight=np.array(w),
                   extras={})
        ev = stacked_event_study(st, kmin=-6, kmax=6)
        assert ev.pretrend_p < 0.01


class TestATTGT:
    def test_group_time_effects_track_the_truth(self, staggered):
        r = att_gt(staggered["y"], staggered["unit"], staggered["period"],
                   staggered["gvar"])
        dyn = r.aggregate_dynamic(0, 6)
        # The simulated effect is 0.05 per quarter since treatment, capped at 8.
        for k in (0, 2, 4):
            if k in dyn:
                assert dyn[k][0] == pytest.approx(0.05 * k, abs=0.06)

    def test_pre_period_effects_are_near_zero(self, staggered):
        """Small next to the post-treatment effects, which reach 0.40."""
        r = att_gt(staggered["y"], staggered["unit"], staggered["period"],
                   staggered["gvar"])
        dyn = r.aggregate_dynamic(-4, -2)
        for k, (est, _) in dyn.items():
            assert abs(est) < 0.10, f"pre-period {k} moved by {est:.3f}"


class TestWildBootstrap:
    def test_rejects_when_an_effect_is_present(self, rng):
        st = make_stack(rng, effect=0.08)
        did = (st.treated * st.post)[:, None]
        r = wild_cluster_bootstrap(
            st.y, did, fes=[factorize(st.unit_by_event),
                            factorize(st.period_by_event)],
            cluster=st.unit, names=["did"], n_boot=199, seed=3)
        assert r.p_value < 0.05

    def test_does_not_reject_when_there_is_no_effect(self, rng):
        st = make_stack(rng, effect=0.0)
        did = (st.treated * st.post)[:, None]
        r = wild_cluster_bootstrap(
            st.y, did, fes=[factorize(st.unit_by_event),
                            factorize(st.period_by_event)],
            cluster=st.unit, names=["did"], n_boot=199, seed=3)
        assert r.p_value > 0.05
