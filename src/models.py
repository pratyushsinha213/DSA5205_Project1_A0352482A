"""Linear estimators with the project's exact ridge scaling.

    beta_hat(z) = (X'X + z * T_train * I)^{-1} X'y

Everything goes through a thin SVD ``X = U diag(s) V'`` so that
``beta_hat(z) = V diag(s / (s^2 + z T)) U'y``.  For ``z = 0`` this is the
minimum-norm least-squares (Moore-Penrose) solution, which is numerically
stable at the interpolation threshold; ``X'X`` is never formed or inverted.
When ``P > T`` the thin SVD is the dual/primal shortcut: the cost is
``O(T^2 P)`` and no ``P x P`` matrix is ever built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class RidgeSVD:
    """Thin-SVD factorisation of a training design supporting many ``z`` cheaply."""

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    uty: np.ndarray        # U'y
    n_train: int           # T in the penalty z*T

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "RidgeSVD":
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return cls(U=U, s=s, Vt=Vt, uty=U.T @ y, n_train=X.shape[0])

    def _spectral_filter(self, z: float) -> np.ndarray:
        """Diagonal ``d_i`` with ``beta = V (d * U'y)``."""
        if z == 0:
            # Moore-Penrose cut-off, identical to numpy.linalg.lstsq / pinv default.
            tol = np.finfo(float).eps * max(self.U.shape[0], self.Vt.shape[1]) * (self.s[0] if self.s.size else 0.0)
            d = np.zeros_like(self.s)
            keep = self.s > tol
            d[keep] = 1.0 / self.s[keep]
            return d
        return self.s / (self.s ** 2 + z * self.n_train)

    def coef_in_basis(self, z: float) -> np.ndarray:
        """Coefficients in the right-singular basis (``||beta||_2 = ||coef||_2``)."""
        return self._spectral_filter(z) * self.uty

    def beta(self, z: float) -> np.ndarray:
        return self.Vt.T @ self.coef_in_basis(z)

    def predict_many(self, X_new: np.ndarray, z_grid: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictions and squared coefficient norms for every ``z`` in one pass.

        Returns ``(yhat, norms)`` with shapes ``(n_new, n_z)`` and ``(n_z,)``.
        """
        proj = X_new @ self.Vt.T                                   # (n_new, k)
        coefs = np.stack([self.coef_in_basis(float(z)) for z in z_grid], axis=1)   # (k, n_z)
        return proj @ coefs, np.sum(coefs ** 2, axis=0)


def ridge_beta(X: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
    """Convenience wrapper: ``beta_hat(z)`` with the course scaling."""
    return RidgeSVD.fit(X, y).beta(z)


class RidgeRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn compatible ridge using the ``z * T`` penalty scaling (Task 3).

    ``fit_intercept=True`` centres X and y on the training data only.
    """

    def __init__(self, z: float = 1.0, fit_intercept: bool = True):
        self.z = z
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        self.y_mean_ = float(y.mean()) if self.fit_intercept else 0.0
        self.coef_ = ridge_beta(X - self.x_mean_, y - self.y_mean_, self.z)
        return self

    def predict(self, X):
        return (np.asarray(X, dtype=float) - self.x_mean_) @ self.coef_ + self.y_mean_


# ---------------------------------------------------------------------------
# Task 3 candidate factory
# ---------------------------------------------------------------------------
def build_estimator(family: str, params: dict, fit_intercept: bool = True, max_iter: int = 10000):
    """Return an unfitted sklearn ``Pipeline`` for a Task 3 candidate.

    Every pipeline starts with a ``StandardScaler`` that is fitted on the training
    rows of whichever fold it is fitted on, so no validation/test statistics leak.
    Ridge uses the course scaling ``(X'X + z T I)``; Lasso/ElasticNet use sklearn's
    ``1/(2n)`` loss scaling with penalty ``alpha``.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNet, Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = ("scale", StandardScaler(with_mean=fit_intercept))
    if family == "ridge":
        return Pipeline([scaler, ("model", RidgeRegressor(z=params["z"], fit_intercept=fit_intercept))])
    if family == "lasso":
        return Pipeline([scaler, ("model", Lasso(alpha=params["alpha"], fit_intercept=fit_intercept, max_iter=max_iter))])
    if family == "elastic_net":
        return Pipeline([scaler, ("model", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                                                      fit_intercept=fit_intercept, max_iter=max_iter))])
    if family == "pca_ridge":
        return Pipeline([scaler, ("pca", PCA(n_components=params["n_components"], svd_solver="full")),
                         ("model", RidgeRegressor(z=params["z"], fit_intercept=fit_intercept))])
    raise ValueError(f"unknown model family {family!r}")
