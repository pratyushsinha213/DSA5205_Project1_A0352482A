"""Task 1 - misspecified simulation with ridge / ridgeless regression.

Run:  python -m src.task1_ridge --config configs/default.yaml

Pipeline: simulate -> sweep (c_q, z) -> save raw tables -> aggregate -> plots.
Tables are written before any figure so every figure is reproducible from CSV.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import metrics as M
from .config import Paths, base_parser, load_config, resolve_paths, save_run_metadata, setup_logging
from .models import RidgeSVD
from .simulation import SimData, n_features, n_observed, simulate_replication
from .validation import holdout_split

log = logging.getLogger("dsa5205")

METRIC_COLS = ["r2_paper", "timing_return", "sharpe_paper", "sharpe_var", "beta_norm_sq"]


def evaluate_replication(cfg: dict[str, Any], rep: int, limit_threads: bool = False) -> dict[str, Any]:
    """Run the full (c_q x z) sweep for one replication.

    Returns raw result rows, the validation-selected ridge rows (z picked on a
    chronological hold-out of the *training* sample only) and the data fingerprint.
    """
    def _run() -> dict[str, Any]:
        data = simulate_replication(cfg, rep)
        T_tr = data.S_train.shape[0]
        P = n_features(cfg)
        z_grid = [float(z) for z in cfg["task1"]["z_grid"]]
        fit_idx, val_idx = holdout_split(T_tr, float(cfg["task1"]["validation_fraction"]))
        rows, val_rows = [], []
        for cq in cfg["task1"]["cq_grid"]:
            p1 = n_observed(cq, T_tr, P)
            cols = data.observed_columns(p1)                     # first P1 columns of the fixed permutation
            Xtr, Xte = data.S_train[:, cols], data.S_test[:, cols]
            svd = RidgeSVD.fit(Xtr, data.y_train)
            yhat, norms = svd.predict_many(Xte, z_grid)
            for j, z in enumerate(z_grid):
                m = M.compute_metrics(data.y_test, yhat[:, j], norms[j])
                m["r2_identity_gap"] = abs(m["r2_paper"] - M.r2_paper_identity(data.y_test, yhat[:, j]))
                rows.append({"rep": rep, "cq": cq, "p1": p1, "z": z, **m})
            # Validation-selected shrinkage: fit on early rows, score on later rows, training data only.
            # Penalty uses the fit-sample size so that z keeps its per-observation meaning.
            svd_fit = RidgeSVD.fit(Xtr[fit_idx], data.y_train[fit_idx])
            yv, _ = svd_fit.predict_many(Xtr[val_idx], z_grid)
            val_mse = np.mean((data.y_train[val_idx][:, None] - yv) ** 2, axis=0)
            val_rows.append({"rep": rep, "cq": cq, "p1": p1, "z_selected": z_grid[int(np.argmin(val_mse))],
                             "val_mse_min": float(val_mse.min())})
        extra = {"rep": rep, "mean_y_test_sq": float(np.mean(data.y_test ** 2)),
                 "beta_star_norm_sq": float(np.sum(data.beta_star ** 2))}
        return {"rows": rows, "val_rows": val_rows, "fingerprint": data.fingerprint(), "extra": extra}

    if limit_threads:
        with threadpool_limits(limits=1):
            return _run()
    return _run()


def attach_validated_ridge(results: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Metrics of ridge evaluated at the validation-selected z for each (rep, c_q)."""
    merged = val.merge(results, left_on=["rep", "cq", "p1", "z_selected"], right_on=["rep", "cq", "p1", "z"],
                       how="left", validate="one_to_one")
    assert merged[METRIC_COLS].notna().all().all(), "validated z not found in the z grid"
    return merged.drop(columns=["z"])


def summarise(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Monte Carlo mean, std (ddof=1) and standard error of the mean."""
    g = df.groupby(keys)[cols]
    mean, std, n = g.mean(), g.std(ddof=1), g.count().iloc[:, 0]
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    for c in cols:
        out[f"{c}_se"] = out[f"{c}_std"] / np.sqrt(n)
    out["n_reps"] = n
    return out.reset_index()


def run_sanity_checks(cfg: dict[str, Any], results: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Numerical sanity checks written to ``task1_sanity.csv`` (computed, not asserted by hand)."""
    b, s2 = float(cfg["simulation"]["b_star"]), float(cfg["simulation"]["noise_std"]) ** 2
    checks = []
    checks.append(("max |R2_direct - R2_identity|", float(results["r2_identity_gap"].max()),
                   bool(results["r2_identity_gap"].max() < 1e-8)))
    gap = abs(extras["mean_y_test_sq"].mean() - (s2 + b))
    # Var(R^2) = 2 (s2 + b)^2 for Gaussian R; tolerance = 4 standard errors of the pooled mean.
    tol = 4.0 * np.sqrt(2.0) * (s2 + b) / np.sqrt(cfg["simulation"]["T_test"] * len(extras))
    checks.append(("|mean(R^2) - (noise^2 + b*)| (Monte Carlo)", float(gap), bool(gap < tol)))
    checks.append(("max |  ||beta*||^2 - b* |", float((extras["beta_star_norm_sq"] - b).abs().max()),
                   bool((extras["beta_star_norm_sq"] - b).abs().max() < 1e-12)))
    # SVD solver vs np.linalg.lstsq / direct solve on replication 0
    data: SimData = simulate_replication(cfg, 0)
    T_tr, P = data.S_train.shape[0], n_features(cfg)
    worst0, worstz = 0.0, 0.0
    for cq in cfg["task1"]["cq_grid"]:
        cols = data.observed_columns(n_observed(cq, T_tr, P))
        X = data.S_train[:, cols]
        svd = RidgeSVD.fit(X, data.y_train)
        ref = np.linalg.lstsq(X, data.y_train, rcond=None)[0]
        worst0 = max(worst0, float(np.linalg.norm(svd.beta(0.0) - ref) / max(np.linalg.norm(ref), 1e-12)))
        z = 1.0
        dual = X.T @ np.linalg.solve(X @ X.T + z * T_tr * np.eye(T_tr), data.y_train)   # kernel (dual) form
        worstz = max(worstz, float(np.linalg.norm(svd.beta(z) - dual) / np.linalg.norm(dual)))
    checks.append(("ridgeless SVD vs np.linalg.lstsq (max rel. diff, rep 0)", worst0, bool(worst0 < 1e-6)))
    checks.append(("ridge(z=1) SVD vs dual solve (max rel. diff, rep 0)", worstz, bool(worstz < 1e-8)))
    return pd.DataFrame(checks, columns=["check", "value", "passed"])


def run(cfg: dict[str, Any], paths: Paths) -> None:
    n_reps = int(cfg["simulation"]["n_replications"])
    n_jobs = int(cfg.get("n_jobs", 1))
    log.info("Task 1: P=%d, T_train=%d, %d replications, %d c_q x %d z",
             n_features(cfg), cfg["simulation"]["T_train"], n_reps,
             len(cfg["task1"]["cq_grid"]), len(cfg["task1"]["z_grid"]))
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(evaluate_replication)(cfg, r, n_jobs != 1) for r in range(n_reps))
    outs = list(tqdm(gen, total=n_reps, desc="Task 1 replications"))

    results = pd.DataFrame([r for o in outs for r in o["rows"]])
    val = pd.DataFrame([r for o in outs for r in o["val_rows"]])
    extras = pd.DataFrame([o["extra"] for o in outs])
    fp = pd.DataFrame({"rep": range(n_reps), "fingerprint": [o["fingerprint"] for o in outs]})

    results.to_csv(paths.tables / "task1_results.csv", index=False)
    validated = attach_validated_ridge(results, val)
    validated.to_csv(paths.tables / "task1_validated_ridge.csv", index=False)
    fp.to_csv(paths.tables / "task1_data_fingerprints.csv", index=False)

    summary = summarise(results, ["cq", "z"], METRIC_COLS)
    summary.insert(2, "p1", summary["cq"].map(dict(zip(results["cq"], results["p1"]))))
    summary.to_csv(paths.tables / "task1_summary.csv", index=False)
    summarise(validated, ["cq"], METRIC_COLS).to_csv(paths.tables / "task1_validated_summary.csv", index=False)

    sanity = run_sanity_checks(cfg, results, extras)
    sanity.to_csv(paths.tables / "task1_sanity.csv", index=False)
    for _, r in sanity.iterrows():
        log.info("sanity | %-62s %.3e  %s", r["check"], r["value"], "PASS" if r["passed"] else "FAIL")

    from . import plotting  # imported late so table generation never depends on matplotlib
    plotting.make_task1_figures(paths)
    log.info("Task 1 finished; tables in %s, figures in %s", paths.rel(paths.tables), paths.rel(paths.figures))


def main(argv: list[str] | None = None) -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.output_dir)
    setup_logging(paths.logs, "task1")
    save_run_metadata(cfg, paths)
    run(cfg, paths)


if __name__ == "__main__":
    main()
