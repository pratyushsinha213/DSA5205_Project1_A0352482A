"""Chronological (blocked) validation utilities - training data only, no leakage."""
from __future__ import annotations

from typing import Iterator

import numpy as np


def holdout_split(n: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Single chronological split: first ``(1-f)n`` rows fit, last ``f n`` rows validate.

    Shared by Task 1 (validated ridge) and Task 2 (Lasso/EN) so both pick their
    hyper-parameters from exactly the same rows.
    """
    n_val = max(1, int(round(n * val_fraction)))
    n_fit = n - n_val
    assert n_fit >= 2, "not enough rows for a fit/validation split"
    idx = np.arange(n)
    return idx[:n_fit], idx[n_fit:]


def forward_splits(n: int, n_folds: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window forward validation.

    Rows are cut into ``n_folds + 1`` consecutive blocks; fold ``k`` trains on
    blocks ``0..k`` and validates on block ``k+1``.  Validation is therefore
    always strictly later than training.
    """
    assert n_folds >= 1 and n >= 2 * (n_folds + 1), "series too short for requested folds"
    edges = np.linspace(0, n, n_folds + 2).astype(int)
    for k in range(n_folds):
        yield np.arange(0, edges[k + 1]), np.arange(edges[k + 1], edges[k + 2])
