"""End-to-end smoke test on the tiny config: every stage runs and writes its artefacts."""
from unicodedata import name

import numpy as np
import pandas as pd
import pytest

from src import run_all, task1_ridge, task2_alternative, task3_hidden_prediction, task4_nn_dynamics


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke_outputs")
    raw = tmp_path_factory.mktemp("synthetic_raw")          # synthetic stand-in, NOT course data
    rng = np.random.default_rng(1)
    for name, (T, P) in {"A": (120, 6), "B": (150, 25)}.items():
        X = rng.standard_normal((T + 30, P))
        y = X[:, 0] * 0.4 + rng.standard_normal(T + 30)
        cols = [f"feature{i + 1}" for i in range(P)]
        tr = pd.DataFrame(X[:T], columns=cols); tr.insert(0, "t", range(T)); tr["return"] = y[:T]
        te = pd.DataFrame(X[T:], columns=cols); te.insert(0, "t", range(T, T + 30))
        tr.to_csv(raw / f"pair{name}_train.csv", index=False)
        te[["t"] + cols[::-1]].to_csv(raw / f"pair{name}_test_features.csv", index=False)     # shuffled column order
    run_all.main(["--config", "configs/smoke.yaml", "--output-dir", str(out), "--raw-dir", str(raw),
                  "--student-id", "TESTID"])
    return out


def test_expected_artifacts_exist(smoke_run):
    for f in ["tables/task1_results.csv", "tables/task1_summary.csv", "tables/task2_results.csv",
              "tables/task3_validation_A.csv", "tables/task3_validation_B.csv",
              "predictions/TESTID_predictions_A.csv", "predictions/TESTID_predictions_B.csv",
              "figures/task1_r2_vs_cq.png", "figures/task1_timing_return_vs_cq.png",
              "figures/task1_sharpe_vs_cq.png", "figures/task1_beta_norm_vs_cq.png",
              "figures/task2_ridge_vs_alternative_sharpe.png", "figures/task4_training_dynamics.png",
              "logs/config_used.yaml", "logs/environment.yaml"]:
        assert (smoke_run / f).exists(), f
    assert not (smoke_run / "predictions" / "TESTID_predictions_C.csv").exists()   # C was not provided


def test_task1_grid_and_sanity_checks(smoke_run):
    res = pd.read_csv(smoke_run / "tables/task1_results.csv")
    assert set(res["cq"]) == {0.5, 0.9, 1.1, 2, 5, 10} and res["rep"].nunique() == 3
    assert res["r2_identity_gap"].max() < 1e-8
    assert pd.read_csv(smoke_run / "tables/task1_sanity.csv")["passed"].all()


def test_task2_uses_identical_data(smoke_run):
    t1 = pd.read_csv(smoke_run / "tables/task1_data_fingerprints.csv", dtype=str)
    assert len(t1) == 3
    t2 = pd.read_csv(smoke_run / "tables/task2_results.csv")
    assert set(t2["cq"]) == {0.5, 0.9, 1.1, 2, 5, 10} and t2["rep"].nunique() == 2


def test_task2_refuses_mismatched_data(smoke_run, tmp_path):
    from src.config import load_config
    cfg = load_config("configs/smoke.yaml")
    cfg["seed"] = 999                                         # different data than Task 1 saved
    paths = task1_ridge.resolve_paths(cfg, smoke_run)
    with pytest.raises(RuntimeError, match="differ from Task 1"):
        task2_alternative.run(cfg, paths)


def test_prediction_files_validated(smoke_run):
    from src.data import validate_prediction_file
    for name in ("A", "B"):
        df = pd.read_csv(smoke_run / f"predictions/TESTID_predictions_{name}.csv", dtype={"t": str})
        assert list(df.columns) == ["t", "yhat"] and len(df) == 30 and np.isfinite(df["yhat"]).all()
        validate_prediction_file(smoke_run / f"predictions/TESTID_predictions_{name}.csv", df["t"].tolist())


def test_task3_skips_missing_data_without_crashing(tmp_path):
    task3_hidden_prediction.main(["--config", "configs/smoke.yaml", "--output-dir", str(tmp_path),
                                  "--raw-dir", str(tmp_path / "nothing"), "--student-id", "X"])
    assert not list((tmp_path / "predictions").glob("*.csv"))


def test_task4_curves_record_all_quantities(smoke_run):
    c = pd.read_csv(smoke_run / "tables/task4_training_curves.csv")
    for col in ("train_mse", "val_mse", "test_mse", "test_r2_paper", "test_sharpe_paper", "weight_norm"):
        assert col in c.columns and np.isfinite(c[col]).all()
