"""Ridge uses (X'X + z T I)^-1 X'y; ridgeless is the SVD minimum-norm solution."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.models import RidgeRegressor, RidgeSVD, ridge_beta


def _data(T, P, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P))
    y = X @ rng.standard_normal(P) * 0.1 + rng.standard_normal(T)
    return X, y


@pytest.mark.parametrize("T,P", [(60, 20), (60, 150)])
@pytest.mark.parametrize("z", [0.1, 1.0, 25.0])
def test_matches_sklearn_after_penalty_conversion(T, P, z):
    X, y = _data(T, P)
    ours = ridge_beta(X, y, z)
    # sklearn minimises ||y-Xb||^2 + alpha ||b||^2  =>  alpha = z * T for the course scaling
    ref = Ridge(alpha=z * T, fit_intercept=False, solver="svd").fit(X, y).coef_
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10)


def test_matches_normal_equations_when_well_posed():
    X, y = _data(80, 10)
    z, T = 0.5, X.shape[0]
    direct = np.linalg.solve(X.T @ X + z * T * np.eye(10), X.T @ y)
    np.testing.assert_allclose(ridge_beta(X, y, z), direct, rtol=1e-9)


def test_wrong_scaling_would_be_detected():
    X, y = _data(60, 20)
    wrong = Ridge(alpha=1.0, fit_intercept=False, solver="svd").fit(X, y).coef_    # penalty z, not z*T
    assert not np.allclose(ridge_beta(X, y, 1.0), wrong, rtol=1e-3)


def test_ridgeless_interpolates_and_is_minimum_norm_when_P_gt_T():
    X, y = _data(40, 120)
    beta = ridge_beta(X, y, 0.0)
    assert np.linalg.norm(X @ beta - y) < 1e-9                          # interpolation
    np.testing.assert_allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-9)
    np.testing.assert_allclose(beta, np.linalg.pinv(X) @ y, atol=1e-9)
    # any other interpolator = beta + null-space vector has a larger norm
    null = np.linalg.svd(X)[2][-1]
    assert np.linalg.norm(beta + 0.7 * null) > np.linalg.norm(beta)
    assert abs(beta @ null) < 1e-9


def test_ridgeless_equals_ols_when_P_lt_T():
    X, y = _data(100, 12)
    np.testing.assert_allclose(ridge_beta(X, y, 0.0), np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-10)


def test_stable_near_interpolation_threshold():
    X, y = _data(100, 100)                                           # square: c_q = 1, ill-conditioned
    beta = ridge_beta(X, y, 0.0)
    assert np.isfinite(beta).all()
    np.testing.assert_allclose(beta, np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-6, atol=1e-8)


def test_predict_many_matches_per_z_beta_and_norms():
    X, y = _data(50, 90)
    Xn = np.random.default_rng(5).standard_normal((30, 90))
    svd = RidgeSVD.fit(X, y)
    zs = [0.0, 0.5, 5.0]
    yhat, norms = svd.predict_many(Xn, zs)
    for j, z in enumerate(zs):
        b = svd.beta(z)
        np.testing.assert_allclose(yhat[:, j], Xn @ b, atol=1e-10)
        assert norms[j] == pytest.approx(b @ b, rel=1e-9)


def test_ridge_regressor_with_intercept_matches_sklearn():
    X, y = _data(70, 15)
    y = y + 3.0
    z, T = 2.0, X.shape[0]
    ours = RidgeRegressor(z=z, fit_intercept=True).fit(X, y)
    ref = Ridge(alpha=z * T, fit_intercept=True, solver="svd").fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)
