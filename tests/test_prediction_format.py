"""Prediction files: exactly t,yhat, UTF-8, no index, exact t order, finite values."""
import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, prediction_path, validate_prediction_file, write_predictions

T = ["10", "11", "12", "13"]


def test_written_file_has_exact_schema(tmp_path):
    path = prediction_path(tmp_path, "A0123456X", "A")
    assert path.name == "A0123456X_predictions_A.csv"
    write_predictions(path, T, np.array([0.1, -0.2, 0.0, 1.5]), expected_t=T)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "t,yhat" and len(lines) == len(T) + 1
    df = pd.read_csv(path, dtype={"t": str})
    assert list(df.columns) == ["t", "yhat"] and df["t"].tolist() == T
    assert np.isfinite(df["yhat"]).all()
    validate_prediction_file(path, T)


def test_write_rejects_bad_predictions(tmp_path):
    p = tmp_path / "x.csv"
    with pytest.raises(DataValidationError, match="count"):
        write_predictions(p, T, np.zeros(3), expected_t=T)
    with pytest.raises(DataValidationError, match="finite"):
        write_predictions(p, T, np.array([0, np.nan, 0, 0.0]), expected_t=T)
    with pytest.raises(DataValidationError, match="finite"):
        write_predictions(p, T, np.array([0, np.inf, 0, 0.0]), expected_t=T)
    with pytest.raises(DataValidationError, match="'t'"):
        write_predictions(p, T[::-1], np.zeros(4), expected_t=T)


def test_validator_catches_bad_files(tmp_path):
    good = pd.DataFrame({"t": T, "yhat": [0.1, 0.2, 0.3, 0.4]})
    cases = {
        "index": lambda p: good.to_csv(p, index=True),                         # saved dataframe index
        "extra_col": lambda p: good.assign(z=1).to_csv(p, index=False),
        "renamed": lambda p: good.rename(columns={"yhat": "pred"}).to_csv(p, index=False),
        "reordered": lambda p: good.iloc[::-1].to_csv(p, index=False),
        "truncated": lambda p: good.iloc[:3].to_csv(p, index=False),
        "nan": lambda p: good.assign(yhat=[0.1, np.nan, 0.3, 0.4]).to_csv(p, index=False),
    }
    for name, writer in cases.items():
        p = tmp_path / f"{name}.csv"
        writer(p)
        with pytest.raises(DataValidationError):
            validate_prediction_file(p, T)


def test_validator_rejects_non_utf8(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_bytes("t,yhat\n10,1.0\n".encode("utf-16"))
    with pytest.raises(DataValidationError):
        validate_prediction_file(p, ["10"])
