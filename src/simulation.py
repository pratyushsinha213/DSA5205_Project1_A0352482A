"""Task 1/2 data-generating process with partial observability.

DGP:   R_{t+1} = S_t' beta* + eps_{t+1},   S_t ~ N(0, I_P),  eps ~ N(0, noise_std^2),
       beta* dense isotropic, rescaled so that ||beta*||^2 = b*,  P = c * T_train.

Observation protocol (per replication): ONE random column permutation is drawn
and kept fixed; the model with target complexity ``c_q`` sees the first
``P1 = round(c_q * T_train)`` columns of that permutation, so observed feature
sets are nested across ``c_q``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import STREAM_SIMULATION, child_rng


@dataclass
class SimData:
    """One Monte Carlo replication."""

    rep: int
    S_train: np.ndarray
    y_train: np.ndarray
    S_test: np.ndarray
    y_test: np.ndarray
    beta_star: np.ndarray
    perm: np.ndarray            # fixed column permutation for this replication

    def observed_columns(self, p1: int) -> np.ndarray:
        """First ``p1`` columns of the replication's permutation (nested sets)."""
        assert 1 <= p1 <= self.perm.size, "P1 must lie in [1, P]"
        return self.perm[:p1]

    def fingerprint(self) -> str:
        """SHA-256 over every array; Task 2 asserts it matches Task 1's value."""
        h = hashlib.sha256()
        for arr in (self.S_train, self.y_train, self.S_test, self.y_test, self.beta_star, self.perm):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]


def make_beta_star(P: int, b_star: float, rng: np.random.Generator) -> np.ndarray:
    """Dense isotropic coefficient vector normalised to ``||beta||^2 = b*``."""
    g = rng.standard_normal(P)
    return g * np.sqrt(b_star) / np.linalg.norm(g)


def n_features(cfg: dict[str, Any]) -> int:
    sim = cfg["simulation"]
    return int(round(sim["c_true"] * sim["T_train"]))


def n_observed(cq: float, t_train: int, p_total: int) -> int:
    """``P1 = round(c_q * T_train)``, capped at ``P``."""
    return int(min(p_total, max(1, round(cq * t_train))))


def simulate_replication(cfg: dict[str, Any], rep: int) -> SimData:
    """Deterministically generate replication ``rep`` from the master seed."""
    sim = cfg["simulation"]
    T_tr, T_te = int(sim["T_train"]), int(sim["T_test"])
    P = n_features(cfg)
    rng = child_rng(int(cfg["seed"]), STREAM_SIMULATION, rep)
    # Draw order is part of the reproducibility contract - do not reorder.
    beta = make_beta_star(P, float(sim["b_star"]), rng)
    S = rng.standard_normal((T_tr + T_te, P))
    eps = rng.standard_normal(T_tr + T_te) * float(sim["noise_std"])
    R = S @ beta + eps
    perm = rng.permutation(P)
    return SimData(rep=rep, S_train=S[:T_tr], y_train=R[:T_tr], S_test=S[T_tr:], y_test=R[T_tr:],
                   beta_star=beta, perm=perm)
