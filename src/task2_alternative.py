"""Task 2 - Lasso / Elastic Net on EXACTLY the same simulated data as Task 1.

Run:  python -m src.task2_alternative --config configs/default.yaml   (after Task 1)

Fair-comparison guarantees
* data are regenerated from the same (seed, replication) streams and the SHA-256
  fingerprint of every replication is asserted equal to the one saved by Task 1;
* same train/test split, same fixed column permutation, same c_q grid, same metrics;
* hyper-parameters are tuned on a chronological hold-out of the TRAINING sample
  (identical rows to Task 1's validated ridge); test labels are never used for tuning;
* features are divided by their training standard deviation (no centring and no
  intercept, because the DGP is zero-mean) and coefficients are mapped back to
  original feature units before ``||beta||^2`` and predictions are computed.

sklearn objective:  1/(2n) ||y - Xw||^2 + alpha (l1 |w|_1 + (1-l1)/2 |w|_2^2).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import enet_path
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import warnings

from . import metrics as M
from .config import Paths, base_parser, load_config, resolve_paths, save_run_metadata, setup_logging
from .simulation import n_features, n_observed, simulate_replication
from .task1_ridge import summarise
from .validation import holdout_split

log = logging.getLogger("dsa5205")
METRIC_COLS = ["r2_paper", "timing_return", "sharpe_paper", "sharpe_var", "beta_norm_sq", "n_nonzero"]


def _train_scale(X: np.ndarray, enabled: bool) -> np.ndarray:
    """Per-column training standard deviation about zero (guarded against zeros)."""
    if not enabled:
        return np.ones(X.shape[1])
    s = np.sqrt(np.mean(X ** 2, axis=0))
    s[s == 0] = 1.0
    return s


def select_hyperparameters(val_mse: dict[float, np.ndarray], alphas: np.ndarray) -> dict[str, dict[str, float]]:
    """Choose (alpha, l1_ratio) for 'elastic_net' and alpha for 'lasso' (l1 = 1) from validation MSEs."""
    best: dict[str, dict[str, float]] = {}
    for l1, mse in val_mse.items():
        j = int(np.argmin(mse))
        cand = {"alpha": float(alphas[j]), "l1_ratio": float(l1), "val_mse": float(mse[j])}
        if "elastic_net" not in best or cand["val_mse"] < best["elastic_net"]["val_mse"]:
            best["elastic_net"] = cand
        if float(l1) == 1.0:
            best["lasso"] = cand
    return best


def path_over_alphas(X: np.ndarray, y: np.ndarray, l1: float, alphas: np.ndarray, t2: dict[str, Any]) -> np.ndarray:
    """Warm-started coefficient path (features x alphas) for descending ``alphas``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, coefs, _ = enet_path(X, y, l1_ratio=float(l1), alphas=alphas, max_iter=int(t2["max_iter"]),
                                tol=float(t2["tol"]))
    return coefs


def evaluate_cq(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                t2: dict[str, Any]) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Fixed-alpha sweep + validated choice for one (replication, c_q).

    1. Tuning (training data only): paths on the early training rows, scored on the later rows.
    2. Final fits: paths on the full training sample; test metrics for EVERY (l1, alpha) are
       recorded (fixed-hyper-parameter sweep), and the validated choice is read off that table.
    """
    fit_idx, val_idx = holdout_split(Xtr.shape[0], float(t2["validation_fraction"]))
    scale_fit = _train_scale(Xtr[fit_idx], t2["scale_features"])
    scale_full = _train_scale(Xtr, t2["scale_features"])
    alphas = np.sort(np.asarray(t2["alpha_grid"], dtype=float))[::-1]     # descending for warm starts
    Xf, Xv = Xtr[fit_idx] / scale_fit, Xtr[val_idx] / scale_fit
    val_mse, rows = {}, []
    for l1 in t2["l1_ratio_grid"]:
        c_fit = path_over_alphas(Xf, ytr[fit_idx], l1, alphas, t2)
        val_mse[float(l1)] = np.mean((ytr[val_idx][:, None] - Xv @ c_fit) ** 2, axis=0)
        beta = path_over_alphas(Xtr / scale_full, ytr, l1, alphas, t2) / scale_full[:, None]   # original units
        yhat = Xte @ beta
        for j, a in enumerate(alphas):
            m = M.compute_metrics(yte, yhat[:, j], M.beta_norm_sq(beta[:, j]))
            rows.append({"l1_ratio": float(l1), "alpha": float(a), "n_nonzero": int(np.count_nonzero(beta[:, j])), **m})
    return rows, select_hyperparameters(val_mse, alphas)


def evaluate_replication(cfg: dict[str, Any], rep: int, limit_threads: bool = False) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        t2 = cfg["task2"]
        data = simulate_replication(cfg, rep)
        T_tr, P = data.S_train.shape[0], n_features(cfg)
        fixed, chosen_rows = [], []
        for cq in cfg["task1"]["cq_grid"]:
            p1 = n_observed(cq, T_tr, P)
            cols = data.observed_columns(p1)                 # same nested permutation prefix as Task 1
            rows, chosen = evaluate_cq(data.S_train[:, cols], data.y_train, data.S_test[:, cols], data.y_test, t2)
            fixed += [{"rep": rep, "cq": cq, "p1": p1, **r} for r in rows]
            by_key = {(r["l1_ratio"], r["alpha"]): r for r in rows}
            for model, hp in chosen.items():
                r = by_key[(hp["l1_ratio"], hp["alpha"])]
                chosen_rows.append({"rep": rep, "cq": cq, "p1": p1, "model": model, "alpha": hp["alpha"],
                                    "l1_ratio": hp["l1_ratio"], "val_mse": hp["val_mse"],
                                    **{k: r[k] for k in METRIC_COLS}})
        return {"fixed": fixed, "chosen": chosen_rows, "fingerprint": data.fingerprint()}

    if limit_threads:
        with threadpool_limits(limits=1):
            return _run()
    return _run()


def run(cfg: dict[str, Any], paths: Paths) -> None:
    fp_path = paths.tables / "task1_data_fingerprints.csv"
    if not fp_path.exists():
        raise FileNotFoundError("Run Task 1 first (task1_data_fingerprints.csv missing): "
                                "python -m src.task1_ridge --config <same config>")
    t1_fp = pd.read_csv(fp_path, dtype={"fingerprint": str}).set_index("rep")["fingerprint"]
    n_reps = int(cfg["task2"]["n_replications"])
    assert n_reps <= len(t1_fp), "Task 2 asks for more replications than Task 1 produced"
    n_jobs = int(cfg.get("n_jobs", 1))
    log.info("Task 2: %d replications (first %d of Task 1), grid %d alphas x %d l1 ratios",
             n_reps, n_reps, len(cfg["task2"]["alpha_grid"]), len(cfg["task2"]["l1_ratio_grid"]))
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(evaluate_replication)(cfg, r, n_jobs != 1) for r in range(n_reps))
    outs = list(tqdm(gen, total=n_reps, desc="Task 2 replications"))

    for r, o in enumerate(outs):       # identical-data assertion
        if o["fingerprint"] != t1_fp.loc[r]:
            raise RuntimeError(f"Replication {r}: Task 2 data differ from Task 1 data "
                               f"(config/seed changed since Task 1 was run?)")
    log.info("Data fingerprints of all %d replications match Task 1.", n_reps)

    fixed = pd.DataFrame([r for o in outs for r in o["fixed"]])
    fixed.to_csv(paths.tables / "task2_fixed_alpha_results.csv", index=False)
    summarise(fixed, ["cq", "l1_ratio", "alpha"], METRIC_COLS).to_csv(
        paths.tables / "task2_fixed_alpha_summary.csv", index=False)
    results = pd.DataFrame([r for o in outs for r in o["chosen"]])
    results.to_csv(paths.tables / "task2_results.csv", index=False)
    summary = summarise(results, ["cq", "model"], METRIC_COLS)
    summary.insert(2, "p1", summary["cq"].map(dict(zip(results["cq"], results["p1"]))))
    summary.to_csv(paths.tables / "task2_summary.csv", index=False)
    hp = (results.groupby(["cq", "model"])
          .agg(alpha_median=("alpha", "median"), l1_ratio_median=("l1_ratio", "median"),
               alpha_min=("alpha", "min"), alpha_max=("alpha", "max")).reset_index())
    hp.to_csv(paths.tables / "task2_selected_hyperparameters.csv", index=False)

    from . import plotting
    plotting.make_task2_figures(paths)
    log.info("Task 2 finished.")


def main(argv: list[str] | None = None) -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.output_dir)
    setup_logging(paths.logs, "task2")
    save_run_metadata(cfg, paths)
    run(cfg, paths)


if __name__ == "__main__":
    main()
