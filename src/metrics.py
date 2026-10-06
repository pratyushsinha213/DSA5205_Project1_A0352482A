"""Evaluation metrics using the project's (KMZ) zero-mean / second-moment conventions.

* ``R2_paper = 1 - mean[(R - Rhat)^2] / mean[R^2]``  (denominator is NOT a variance)
* timing return ``R^pi = pi_t R_{t+1}`` with ``pi_t = Rhat_{t+1}``
* paper Sharpe ratio ``E[R^pi] / sqrt(E[(R^pi)^2])``  (uncentred second moment)
* secondary ``SR_var = E[R^pi] / sqrt(Var(R^pi))``
"""
from __future__ import annotations

import numpy as np


def _check(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    assert y.shape == yhat.shape and y.ndim == 1, "y and yhat must be 1-D arrays of equal length"
    return y, yhat


def r2_paper(y: np.ndarray, yhat: np.ndarray) -> float:
    """Direct finite-sample paper-style out-of-sample R^2."""
    y, yhat = _check(y, yhat)
    return float(1.0 - np.mean((y - yhat) ** 2) / np.mean(y ** 2))


def r2_paper_identity(y: np.ndarray, yhat: np.ndarray) -> float:
    """Equivalent diagnostic form ``(2 E[Rhat R] - E[Rhat^2]) / E[R^2]``."""
    y, yhat = _check(y, yhat)
    return float((2.0 * np.mean(yhat * y) - np.mean(yhat ** 2)) / np.mean(y ** 2))


def timing_returns(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Timing-strategy returns ``pi_t * R_{t+1}`` with ``pi_t = yhat``."""
    y, yhat = _check(y, yhat)
    return yhat * y


def expected_timing_return(y: np.ndarray, yhat: np.ndarray) -> float:
    """``E[R^pi] ~= mean(Rhat * R)``."""
    return float(np.mean(timing_returns(y, yhat)))


def sharpe_paper(y: np.ndarray, yhat: np.ndarray) -> float:
    """Paper SR: mean timing return over the root *uncentred* second moment."""
    rp = timing_returns(y, yhat)
    denom = np.sqrt(np.mean(rp ** 2))
    # Convention: a strategy that never takes a position (all-zero predictions,
    # e.g. fully shrunk Lasso) has timing returns identically 0, so SR := 0 rather than 0/0.
    return float(rp.mean() / denom) if denom > 0 else 0.0


def sharpe_var(y: np.ndarray, yhat: np.ndarray) -> float:
    """Secondary variance-based Sharpe ratio (kept separate from ``sharpe_paper``)."""
    rp = timing_returns(y, yhat)
    sd = rp.std(ddof=0)
    return float(rp.mean() / sd) if sd > 0 else 0.0     # same never-trades convention as sharpe_paper


def beta_norm_sq(beta: np.ndarray) -> float:
    """Squared Euclidean norm of the estimated coefficient vector."""
    beta = np.asarray(beta, dtype=float)
    return float(beta @ beta)


def compute_metrics(y: np.ndarray, yhat: np.ndarray, beta_norm2: float | None = None) -> dict[str, float]:
    """All reported metrics for one (replication, c_q, model-setting)."""
    out = {
        "r2_paper": r2_paper(y, yhat),
        "timing_return": expected_timing_return(y, yhat),
        "sharpe_paper": sharpe_paper(y, yhat),
        "sharpe_var": sharpe_var(y, yhat),
    }
    if beta_norm2 is not None:
        out["beta_norm_sq"] = float(beta_norm2)
    return out
