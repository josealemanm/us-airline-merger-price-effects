"""The regression machinery, checked against established implementations.

Writing your own estimator is only defensible if you show it agrees with one
somebody else wrote. These tests compare against statsmodels and linearmodels to
eight decimal places on coefficients *and* standard errors - the standard errors
are the part that is easy to get subtly wrong and impossible to notice.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from econ.fe_ols import demean, factorize, feols


def dummies(codes, n):
    return np.eye(n)[codes][:, 1:]


class TestAbsorption:
    def test_one_way_matches_dummies(self, panel):
        D = dummies(panel["unit"], panel["n_unit"])
        X = np.column_stack([panel["x1"], panel["x2"]])
        ref = sm.OLS(panel["y"], np.column_stack([X, D, np.ones(len(X))])).fit()
        got = feols(panel["y"], X, fes=[factorize(panel["unit"])],
                    names=["x1", "x2"])
        assert np.allclose(got.params, ref.params[:2], atol=1e-10)

    def test_two_way_matches_dummies(self, panel):
        Du = dummies(panel["unit"], panel["n_unit"])
        Dt = dummies(panel["time"], panel["n_time"])
        X = np.column_stack([panel["x1"], panel["x2"]])
        ref = sm.OLS(panel["y"],
                     np.column_stack([X, Du, Dt, np.ones(len(X))])).fit()
        got = feols(panel["y"], X, fes=[factorize(panel["unit"]),
                                        factorize(panel["time"])],
                    names=["x1", "x2"])
        assert np.allclose(got.params, ref.params[:2], atol=1e-9)

    def test_demeaned_columns_are_orthogonal_to_the_effects(self, panel):
        M, _ = demean(np.column_stack([panel["y"], panel["x1"]]),
                      [factorize(panel["unit"]), factorize(panel["time"])])
        for codes, g in (factorize(panel["unit"]), factorize(panel["time"])):
            means = np.bincount(codes, weights=M[:, 1], minlength=g) / \
                np.bincount(codes, minlength=g)
            assert np.allclose(means, 0.0, atol=1e-8)

    def test_absorption_recovers_the_true_slope(self, panel):
        got = feols(panel["y"], np.column_stack([panel["x1"], panel["x2"]]),
                    fes=[factorize(panel["unit"]), factorize(panel["time"])],
                    names=["x1", "x2"])
        assert np.allclose(got.params, panel["beta"], atol=0.15)


class TestClusteredErrors:
    def test_clustered_se_matches_statsmodels(self, panel):
        Du = dummies(panel["unit"], panel["n_unit"])
        Dt = dummies(panel["time"], panel["n_time"])
        X = np.column_stack([panel["x1"], panel["x2"]])
        ref = sm.WLS(panel["y"], np.column_stack([X, Du, Dt, np.ones(len(X))]),
                     weights=np.ones(len(X))).fit(
            cov_type="cluster", cov_kwds={"groups": panel["unit"]})
        got = feols(panel["y"], X, fes=[factorize(panel["unit"]),
                                        factorize(panel["time"])],
                    cluster=panel["unit"], names=["x1", "x2"])
        assert np.allclose(got.se, ref.bse[:2], rtol=1e-8)

    def test_weighted_clustered_se_matches_statsmodels(self, panel):
        Du = dummies(panel["unit"], panel["n_unit"])
        Dt = dummies(panel["time"], panel["n_time"])
        X = np.column_stack([panel["x1"], panel["x2"]])
        ref = sm.WLS(panel["y"], np.column_stack([X, Du, Dt, np.ones(len(X))]),
                     weights=panel["w"]).fit(
            cov_type="cluster", cov_kwds={"groups": panel["unit"]})
        got = feols(panel["y"], X, fes=[factorize(panel["unit"]),
                                        factorize(panel["time"])],
                    cluster=panel["unit"], weights=panel["w"], names=["x1", "x2"])
        assert np.allclose(got.params, ref.params[:2], rtol=1e-9)
        assert np.allclose(got.se, ref.bse[:2], rtol=1e-8)

    def test_hc1_matches_statsmodels(self, panel):
        D = dummies(panel["unit"], panel["n_unit"])
        X = np.column_stack([panel["x1"], panel["x2"]])
        ref = sm.OLS(panel["y"],
                     np.column_stack([X, D, np.ones(len(X))])).fit(cov_type="HC1")
        got = feols(panel["y"], X, fes=[factorize(panel["unit"])],
                    names=["x1", "x2"])
        assert np.allclose(got.se, ref.bse[:2], rtol=1e-8)

    def test_clustering_widens_the_interval(self, panel):
        """Serially correlated errors make the naive interval too narrow."""
        X = panel["x1"][:, None]
        naive = feols(panel["y"], X, fes=[factorize(panel["time"])],
                      names=["x1"])
        clust = feols(panel["y"], X, fes=[factorize(panel["time"])],
                      cluster=panel["unit"], names=["x1"])
        assert clust.se[0] > naive.se[0]


class TestCollinearity:
    def test_a_duplicated_column_is_dropped(self, panel):
        X = np.column_stack([panel["x1"], panel["x2"], panel["x1"]])
        got = feols(panel["y"], X, fes=[factorize(panel["unit"])],
                    names=["x1", "x2", "x1_copy"])
        assert len(got.names) == 2
        assert "x1_copy" not in got.names

    def test_a_column_inside_the_fixed_effects_is_dropped(self, panel):
        unit_mean = np.bincount(panel["unit"], weights=panel["y"]) / \
            np.bincount(panel["unit"])
        X = np.column_stack([panel["x1"], unit_mean[panel["unit"]]])
        got = feols(panel["y"], X, fes=[factorize(panel["unit"])],
                    names=["x1", "absorbed"])
        assert "absorbed" not in got.names


class TestWald:
    def test_wald_on_a_true_null_usually_passes(self, rng):
        n = 4000
        cl = np.repeat(np.arange(200), 20)
        X = rng.normal(size=(n, 3))
        y = 2.0 * X[:, 0] + rng.normal(size=n)
        r = feols(y, X, fes=[factorize(cl)], cluster=cl,
                  names=["a", "b", "c"])
        _, p = r.wald(["b", "c"])
        assert p > 0.01

    def test_wald_rejects_a_false_null(self, rng):
        n = 4000
        cl = np.repeat(np.arange(200), 20)
        X = rng.normal(size=(n, 3))
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(size=n)
        r = feols(y, X, fes=[factorize(cl)], cluster=cl, names=["a", "b", "c"])
        _, p = r.wald(["b", "c"])
        assert p < 1e-6


class TestIV:
    """The 2SLS path, against linearmodels."""

    def setup_data(self, rng):
        n = 3000
        cl = np.repeat(np.arange(150), 20)
        xi = rng.normal(0, 1, n)
        z1, z2, z3 = (rng.normal(0, 1, n) for _ in range(3))
        p = 0.9 * z1 + 0.6 * z2 + 0.4 * z3 + 1.1 * xi + rng.normal(0, 0.5, n)
        x1 = rng.normal(0, 1, n)
        y = -1.5 * p + 0.7 * x1 + xi + rng.normal(0, 0.3, n)
        return n, cl, p, x1, y, np.column_stack([z1, z2, z3])

    def test_matches_linearmodels(self, rng):
        pd = pytest.importorskip("pandas")
        lm = pytest.importorskip("linearmodels.iv")
        from econ.iv import iv2sls
        n, cl, p, x1, y, Z = self.setup_data(rng)
        exog = pd.DataFrame({"x1": x1, "const": np.ones(n)})
        ref = lm.IV2SLS(pd.Series(y, name="y"), exog,
                        pd.DataFrame({"price": p}),
                        pd.DataFrame({"z1": Z[:, 0], "z2": Z[:, 1],
                                      "z3": Z[:, 2]})
                        ).fit(cov_type="clustered", clusters=cl, debiased=True)
        got = iv2sls(y, np.column_stack([x1, np.ones(n)]), p, Z, cluster=cl,
                     names_exog=["x1", "const"], names_endog=["price"],
                     names_z=["z1", "z2", "z3"])
        assert np.allclose(got.params[got.names.index("price")],
                           ref.params["price"], rtol=1e-8)
        assert np.allclose(got.se[got.names.index("price")],
                           ref.std_errors["price"], rtol=1e-7)

    def test_iv_beats_ols_when_price_is_endogenous(self, rng):
        from econ.iv import iv2sls
        n, cl, p, x1, y, Z = self.setup_data(rng)
        ols = feols(y, np.column_stack([p, x1]), cluster=cl,
                    names=["price", "x1"])
        iv = iv2sls(y, np.column_stack([x1, np.ones(n)]), p, Z, cluster=cl,
                    names_exog=["x1", "const"], names_endog=["price"],
                    names_z=["z1", "z2", "z3"])
        truth = -1.5
        ols_err = abs(ols.params[0] - truth)
        iv_err = abs(iv.params[iv.names.index("price")] - truth)
        assert iv_err < ols_err
        assert iv_err < 0.1

    def test_overidentification_test_accepts_valid_instruments(self, rng):
        from econ.iv import iv2sls
        n, cl, p, x1, y, Z = self.setup_data(rng)
        got = iv2sls(y, np.column_stack([x1, np.ones(n)]), p, Z, cluster=cl,
                     names_exog=["x1", "const"], names_endog=["price"],
                     names_z=["z1", "z2", "z3"])
        assert got.n_overid == 2
        assert got.hansen_p > 0.01

    def test_overidentification_test_rejects_an_invalid_instrument(self, rng):
        """One instrument correlated with the error should be caught."""
        from econ.iv import iv2sls
        n, cl, p, x1, y, Z = self.setup_data(rng)
        # Rebuild y so we know the error, then contaminate one instrument.
        err = y - (-1.5 * p + 0.7 * x1)
        bad = np.column_stack([Z[:, 0], Z[:, 1], Z[:, 2] + 1.5 * err])
        got = iv2sls(y, np.column_stack([x1, np.ones(n)]), p, bad, cluster=cl,
                     names_exog=["x1", "const"], names_endog=["price"],
                     names_z=["z1", "z2", "z3bad"])
        assert got.hansen_p < 0.01

    def test_under_identification_raises(self, rng):
        from econ.iv import iv2sls
        n, cl, p, x1, y, Z = self.setup_data(rng)
        with pytest.raises(ValueError, match="under-identified"):
            iv2sls(y, np.column_stack([x1, np.ones(n)]),
                   np.column_stack([p, x1 * 2.0]), Z[:, :1], cluster=cl)
