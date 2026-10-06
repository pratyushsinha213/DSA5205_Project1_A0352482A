"""Task 3 - hidden-test prediction for datasets A, B, C.

Run:  python -m src.task3_hidden_prediction --config configs/default.yaml --student-id YOUR_ID

Drop the course files ``train_{A,B,C}.csv`` / ``test_{A,B,C}.csv`` into ``data/raw/``.
No code edits are needed.  Procedure per dataset (training labels only):

 1. strict schema / feature-alignment validation;
 2. diagnostics on the training sample;
 3. expanding-window *forward* validation of Ridge, Lasso, Elastic Net, PCA+Ridge
    (scalers, PCA and every hyper-parameter fitted inside the training part of each fold);
 4. selection by the validation metric named in ``task3.selection_metric``;
 5. refit the selected model on ALL training rows, predict the public test features;
 6. write and re-validate ``<student-id>_predictions_<X>.csv`` with exactly ``t,yhat``.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from . import metrics as M
from .config import Paths, base_parser, load_config, resolve_paths, save_run_metadata, setup_logging
from .data import (DataValidationError, TaskData, load_task_data, prediction_path, validate_prediction_file,
                   write_predictions)
from .models import build_estimator
from .validation import forward_splits

log = logging.getLogger("dsa5205")
SELECTABLE = {"r2_paper", "sharpe_paper", "timing_return"}
METRICS = ["r2_paper", "timing_return", "sharpe_paper", "sharpe_var"]


def diagnostics(d: TaskData) -> dict[str, float]:
    """Summary statistics computed from the TRAINING sample only."""
    y, X = d.y_train, d.X_train
    T, P = X.shape
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    corr = Xs.T @ ((y - y.mean()) / (y.std() + 1e-12)) / T
    sv = np.linalg.svd(Xs, compute_uv=False)
    return {
        "n_train": T, "n_test": len(d.t_test), "n_features": P, "c_P_over_T": P / T,
        "y_mean": float(y.mean()), "y_std": float(y.std()), "y_second_moment": float(np.mean(y ** 2)),
        "y_skew": float(stats.skew(y)), "y_excess_kurtosis": float(stats.kurtosis(y)),
        "y_lag1_autocorr": float(np.corrcoef(y[:-1], y[1:])[0, 1]),
        "mean_abs_feature_mean": float(np.mean(np.abs(X.mean(0)))), "mean_feature_std": float(X.std(0).mean()),
        "max_abs_corr_with_y": float(np.max(np.abs(corr))),
        "share_corr_beyond_2se": float(np.mean(np.abs(corr) > 2 / np.sqrt(T))),   # ~0.05 expected under pure noise
        "effective_rank": float(np.sum(sv ** 2) ** 2 / np.sum(sv ** 4)),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
    }


def candidates(cfg3: dict[str, Any], n_feat: int, min_train: int) -> list[tuple[str, dict]]:
    """Explicit, small candidate grid (no uncontrolled model zoo)."""
    out: list[tuple[str, dict]] = [("ridge", {"z": float(z)}) for z in cfg3["z_grid"]]
    out += [("lasso", {"alpha": float(a)}) for a in cfg3["alpha_grid"]]
    out += [("elastic_net", {"alpha": float(a), "l1_ratio": float(l)})
            for a in cfg3["alpha_grid"] for l in cfg3["l1_ratio_grid"]]
    kmax = min(n_feat, min_train) - 1
    out += [("pca_ridge", {"n_components": int(k), "z": float(z)})
            for k in cfg3["pca_components"] if k <= kmax for z in cfg3["z_grid"]]
    return out


def evaluate_candidates(X: np.ndarray, y: np.ndarray, cfg3: dict[str, Any]) -> pd.DataFrame:
    """Forward-validate every candidate; one row per candidate with fold-mean/std metrics."""
    splits = list(forward_splits(len(y), int(cfg3["n_folds"])))
    min_train = min(len(tr) for tr, _ in splits)
    rows = []
    for family, params in candidates(cfg3, X.shape[1], min_train):
        fold_metrics = []
        for tr, va in splits:
            est = build_estimator(family, params, bool(cfg3["fit_intercept"]), int(cfg3["max_iter"]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                est.fit(X[tr], y[tr])
            fold_metrics.append(M.compute_metrics(y[va], est.predict(X[va])))
        fm = pd.DataFrame(fold_metrics)
        row = {"family": family, **{k: params.get(k, np.nan) for k in ("z", "alpha", "l1_ratio", "n_components")}}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = float(fm[m].mean()), float(fm[m].std(ddof=1))
        row["n_folds"] = len(splits)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best(table: pd.DataFrame, metric: str) -> int:
    """Row index of the best candidate by fold-mean ``metric`` (ties -> first listed, i.e. simpler/ridge-first)."""
    assert metric in SELECTABLE, f"selection_metric must be one of {sorted(SELECTABLE)}"
    return int(table[f"{metric}_mean"].to_numpy().argmax())


def process_dataset(name: str, cfg: dict[str, Any], paths: Paths, student_id: str) -> dict[str, Any]:
    cfg3 = cfg["task3"]
    d = load_task_data(paths.raw_data, name)
    log.info("[%s] loaded: %d train rows, %d test rows, %d features (alignment OK)",
             name, len(d.y_train), len(d.t_test), len(d.feature_names))
    diag = diagnostics(d)
    log.info("[%s] diagnostics: c=P/T=%.3f, y std=%.4g, excess kurtosis=%.2f, lag-1 autocorr=%.3f, "
             "eff. rank=%.1f, share |corr|>2se=%.3f", name, diag["c_P_over_T"], diag["y_std"],
             diag["y_excess_kurtosis"], diag["y_lag1_autocorr"], diag["effective_rank"], diag["share_corr_beyond_2se"])

    table = evaluate_candidates(d.X_train, d.y_train, cfg3)
    best = select_best(table, cfg3["selection_metric"])
    table["selected"] = False
    table.loc[best, "selected"] = True
    table["rank_by_selection_metric"] = table[f"{cfg3['selection_metric']}_mean"].rank(ascending=False, method="first").astype(int)
    table = table.sort_values("rank_by_selection_metric")
    table.to_csv(paths.tables / f"task3_validation_{name}.csv", index=False)
    sel = table[table["selected"]].iloc[0]
    log.info("[%s] selected %s (z=%s alpha=%s l1=%s k=%s): val %s = %.4g  (zero-predictor baseline: 0)",
             name, sel["family"], sel["z"], sel["alpha"], sel["l1_ratio"], sel["n_components"],
             cfg3["selection_metric"], sel[f"{cfg3['selection_metric']}_mean"])
    if sel["r2_paper_mean"] <= 0:
        log.warning("[%s] no candidate beats the zero predictor on validation R2 - expect a weak/no signal", name)

    params = {k: sel[k] for k in ("z", "alpha", "l1_ratio", "n_components") if pd.notna(sel[k])}
    if "n_components" in params:
        params["n_components"] = int(params["n_components"])
    final = build_estimator(sel["family"], params, bool(cfg3["fit_intercept"]), int(cfg3["max_iter"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final.fit(d.X_train, d.y_train)                       # all training rows
    yhat = final.predict(d.X_test)

    out = prediction_path(paths.predictions, student_id, name)
    write_predictions(out, d.t_test, yhat, expected_t=d.t_test)
    validate_prediction_file(out, d.t_test)
    log.info("[%s] wrote %s (%d rows) and re-validated its schema", name, out.name, len(yhat))
    return {"dataset": name, **diag, "family": sel["family"], **{f"hp_{k}": v for k, v in params.items()},
            **{f"val_{m}": sel[f"{m}_mean"] for m in METRICS}, "selection_metric": cfg3["selection_metric"],
            "prediction_file": out.name}


def run(cfg: dict[str, Any], paths: Paths, student_id: str, datasets: list[str], strict: bool = False) -> None:
    if student_id in ("YOUR_ID", ""):
        log.warning("No real --student-id given: files will be named YOUR_ID_predictions_*.csv")
    done, missing = [], []
    for name in datasets:
        try:
            done.append(process_dataset(name, cfg, paths, student_id))
        except FileNotFoundError as exc:
            missing.append(name)
            log.warning("[%s] skipped: %s not found (put the course CSVs in %s)", name, Path(str(exc)).name,
                        paths.rel(paths.raw_data))
    if strict and missing:
        raise FileNotFoundError(f"datasets missing: {missing}")
    if done:
        pd.DataFrame(done).to_csv(paths.tables / "task3_selected_models.csv", index=False)
    if missing and not done:
        log.warning("Task 3 produced no predictions: no dataset files found in %s", paths.rel(paths.raw_data))


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--student-id", default=None, help="used in <student-id>_predictions_X.csv")
    ap.add_argument("--raw-dir", default=None, help="folder with train_X.csv / test_X.csv (default data/raw)")
    ap.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--strict", action="store_true", help="fail if any dataset file is missing")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.output_dir, args.raw_dir)
    setup_logging(paths.logs, "task3")
    save_run_metadata(cfg, paths)
    run(cfg, paths, args.student_id or str(cfg["task3"].get("student_id", "YOUR_ID")), args.datasets, args.strict)


if __name__ == "__main__":
    main()
