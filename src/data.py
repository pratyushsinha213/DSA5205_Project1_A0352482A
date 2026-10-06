"""Robust CSV loading, feature alignment and prediction-file I/O (Task 3).

Training schema:  ``t, <features...>, return``
Public test:      ``t, <features...>``

The loader fails loudly (``DataValidationError``) on missing/extra/renamed
features, duplicate or non-increasing ``t``, NaNs or non-numeric values, and
returns test features re-ordered to the training feature order while keeping
the test ``t`` strings byte-for-byte as given.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "return"
TIME_COL = "t"


class DataValidationError(ValueError):
    """Raised when a train/test/prediction file violates the expected schema."""


@dataclass
class TaskData:
    name: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    t_train: list[str]
    X_test: np.ndarray
    t_test: list[str]          # raw strings, original order preserved


def _check_time(t: pd.Series, label: str) -> None:
    """``t`` must be unique and strictly increasing (numeric, else datetime)."""
    if t.isna().any():
        raise DataValidationError(f"{label}: NaN in 't'")
    if t.duplicated().any():
        raise DataValidationError(f"{label}: duplicate 't' values")
    key = pd.to_numeric(t, errors="coerce")
    if key.isna().any():
        key = pd.to_datetime(t, errors="coerce")
    if key.isna().any():
        raise DataValidationError(f"{label}: 't' is neither numeric nor a parsable date")
    if not key.is_monotonic_increasing:
        raise DataValidationError(f"{label}: 't' is not strictly increasing")


def _numeric_block(df: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
    block = df[cols].apply(pd.to_numeric, errors="coerce")
    arr = block.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise DataValidationError(f"{label}: NaN/inf/non-numeric values in {[c for c in cols if not np.isfinite(block[c]).all()][:5]}")
    return arr


def align_train_test(train: pd.DataFrame, test: pd.DataFrame, name: str = "dataset") -> TaskData:
    """Validate and align already-loaded frames (``t`` column must be string dtype)."""
    for col in (TIME_COL, TARGET_COL):
        if col not in train.columns:
            raise DataValidationError(f"{name}: training file lacks column '{col}'")
    if TIME_COL not in test.columns:
        raise DataValidationError(f"{name}: test file lacks column '{TIME_COL}'")
    if train.columns.duplicated().any() or test.columns.duplicated().any():
        raise DataValidationError(f"{name}: duplicate column names")

    features = [c for c in train.columns if c not in (TIME_COL, TARGET_COL)]   # ordered from training
    if not features:
        raise DataValidationError(f"{name}: no feature columns in training file")
    test_features = [c for c in test.columns if c != TIME_COL]
    missing = [c for c in features if c not in test_features]
    extra = [c for c in test_features if c not in features]
    if missing:
        raise DataValidationError(f"{name}: test is missing features {missing[:5]}")
    if extra:
        raise DataValidationError(f"{name}: test has unexpected extra columns {extra[:5]}")

    _check_time(train[TIME_COL], f"{name} train")
    _check_time(test[TIME_COL], f"{name} test")
    X_tr = _numeric_block(train, features, f"{name} train")
    y_tr = _numeric_block(train, [TARGET_COL], f"{name} train")[:, 0]
    X_te = _numeric_block(test, features, f"{name} test")            # selecting by name re-orders to training order
    return TaskData(name=name, feature_names=features, X_train=X_tr, y_train=y_tr,
                    t_train=train[TIME_COL].tolist(), X_test=X_te, t_test=test[TIME_COL].tolist())


def load_task_data(raw_dir: Path, name: str) -> TaskData:
    """Load ``train_<name>.csv`` / ``test_<name>.csv`` from ``raw_dir``."""
    tr_path, te_path = Path(raw_dir) / f"train_{name}.csv", Path(raw_dir) / f"test_{name}.csv"
    for p in (tr_path, te_path):
        if not p.exists():
            raise FileNotFoundError(p)
    train = pd.read_csv(tr_path, dtype={TIME_COL: str}, encoding="utf-8")
    test = pd.read_csv(te_path, dtype={TIME_COL: str}, encoding="utf-8")
    return align_train_test(train, test, name)


def prediction_path(pred_dir: Path, student_id: str, name: str) -> Path:
    return Path(pred_dir) / f"{student_id}_predictions_{name}.csv"


def write_predictions(path: Path, t_test: list[str], yhat: np.ndarray, expected_t: list[str]) -> None:
    """Write exactly ``t,yhat`` (UTF-8, no index) after strict assertions."""
    yhat = np.asarray(yhat, dtype=float)
    if len(yhat) != len(expected_t):
        raise DataValidationError(f"prediction count {len(yhat)} != test rows {len(expected_t)}")
    if not np.isfinite(yhat).all():
        raise DataValidationError("non-finite predictions")
    if list(t_test) != list(expected_t):
        raise DataValidationError("output 't' differs from the test file's 't'")
    df = pd.DataFrame({"t": list(t_test), "yhat": yhat})
    assert list(df.columns) == ["t", "yhat"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def validate_prediction_file(path: Path, expected_t: list[str]) -> None:
    """Re-read a written file and check schema, order and finiteness."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path}: not UTF-8") from exc
    if text.splitlines()[0].strip() != "t,yhat":
        raise DataValidationError(f"{path}: header must be exactly 't,yhat'")
    df = pd.read_csv(path, dtype={"t": str})
    if list(df.columns) != ["t", "yhat"]:
        raise DataValidationError(f"{path}: columns must be exactly ['t', 'yhat'], got {list(df.columns)}")
    if df["t"].tolist() != list(expected_t):
        raise DataValidationError(f"{path}: 't' sequence differs from the test file")
    yhat = pd.to_numeric(df["yhat"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(yhat).all():
        raise DataValidationError(f"{path}: non-finite yhat")
