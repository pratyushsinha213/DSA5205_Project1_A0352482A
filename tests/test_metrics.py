"""Metric identities and conventions (paper R^2, paper Sharpe vs variance Sharpe)."""
import numpy as np
import pytest

from src import metrics as M


def test_r2_identity_matches_direct_formula():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.standard_normal(500) * rng.uniform(0.5, 3)
        yhat = 0.3 * y + rng.standard_normal(500)
        direct = 1 - np.mean((y - yhat) ** 2) / np.mean(y ** 2)
        identity = (2 * np.mean(yhat * y) - np.mean(yhat ** 2)) / np.mean(y ** 2)
        assert direct == pytest.approx(identity, abs=1e-12)
        assert M.r2_paper(y, yhat) == pytest.approx(direct, abs=1e-12)
        assert M.r2_paper_identity(y, yhat) == pytest.approx(identity, abs=1e-12)


def test_r2_uses_second_moment_not_variance():
    y = np.array([1.0, 1.0, 1.0, 1.0])           # variance 0, second moment 1
    assert M.r2_paper(y, np.zeros(4)) == pytest.approx(0.0)
    assert M.r2_paper(y, y) == pytest.approx(1.0)


def test_paper_sharpe_denominator_is_uncentred_second_moment():
    y, yhat = np.array([2.0, 0.0]), np.array([1.0, 1.0])     # timing returns [2, 0]
    assert M.expected_timing_return(y, yhat) == pytest.approx(1.0)
    assert M.sharpe_paper(y, yhat) == pytest.approx(1.0 / np.sqrt(2.0))   # sqrt(E[x^2]) = sqrt(2)
    assert M.sharpe_var(y, yhat) == pytest.approx(1.0 / 1.0)               # sd = 1
    assert M.sharpe_paper(y, yhat) != pytest.approx(M.sharpe_var(y, yhat))


def test_never_trading_strategy_has_zero_sharpe_not_nan():
    y = np.array([1.0, -2.0, 0.5])
    assert M.sharpe_paper(y, np.zeros(3)) == 0.0
    assert M.sharpe_var(y, np.zeros(3)) == 0.0


def test_compute_metrics_keys_and_shape_check():
    y = np.arange(5.0)
    out = M.compute_metrics(y, y * 0.5, beta_norm2=2.0)
    assert set(out) == {"r2_paper", "timing_return", "sharpe_paper", "sharpe_var", "beta_norm_sq"}
    with pytest.raises(AssertionError):
        M.r2_paper(y, y[:-1])
