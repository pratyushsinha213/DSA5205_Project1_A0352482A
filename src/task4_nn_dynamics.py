"""Task 4 - training dynamics of a small two-layer network on noisy single-index returns.

Run:  python -m src.task4_nn_dynamics --config configs/default.yaml

Motivation: Montanari & Urbani (NeurIPS 2025) describe a separation of timescales in large
two-layer networks - useful low-dimensional structure is learned first, while overfitting
(complexity growth, feature unlearning) happens later.  This script is a *focused finance
illustration*, not a reproduction of their dynamical mean-field theory:

    R_{t+1} = tanh(s * u'X_t) + eps_{t+1},   X_t ~ N(0, I_d),  ||u|| = 1,  eps ~ N(0, sigma^2)

A width-m network f(x) = (1/m) sum_j a_j tanh(w_j'x / sqrt(d)) is trained by full-batch
gradient descent on the squared loss in the mean-field time scaling (step size multiplied by
m, so each neuron moves at an O(1) rate).  The network is implemented in NumPy so the
experiment is deterministic and needs no deep-learning framework.

Recorded every ``log_every`` epochs (mean over seeds): train/validation/test MSE, test
R2_paper, test paper-Sharpe of the timing strategy pi = fhat(x), a weight-norm complexity
proxy, and the weight mass along the signal direction u versus orthogonal directions.
Early stopping epoch = argmin of the *validation* MSE (test labels are never used to stop).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .config import STREAM_TASK4, Paths, base_parser, child_rng, load_config, resolve_paths, save_run_metadata, setup_logging

log = logging.getLogger("dsa5205")


class TwoLayerNet:
    """f(x) = mean_j a_j tanh(w_j . x / sqrt(d)); mean-field parametrisation, NumPy gradients."""

    def __init__(self, d: int, m: int, rng: np.random.Generator):
        self.d, self.m = d, m
        self.W = rng.standard_normal((m, d))      # rows w_j
        self.a = rng.standard_normal(m)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.tanh(X @ self.W.T / np.sqrt(self.d))      # (n, m)
        return H @ self.a / self.m, H

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One full-batch GD step on 0.5*mean((y-f)^2); returns the pre-step training MSE."""
        n = X.shape[0]
        f, H = self.forward(X)
        res = y - f
        grad_a = -(H.T @ res) / n                                       # per-neuron rate (already x m)
        dH = (1.0 - H ** 2) * (res[:, None] * self.a[None, :])         # (n, m)
        grad_W = -(dH.T @ X) / (n * np.sqrt(self.d))
        self.a -= lr * grad_a
        self.W -= lr * grad_W
        return float(np.mean(res ** 2))

    def complexity(self) -> float:
        """Weight-norm proxy: (||a||^2 + ||W||_F^2) / m."""
        return float((self.a @ self.a + np.sum(self.W ** 2)) / self.m)


def make_data(cfg4: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    d = int(cfg4["d"])
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    out = {"u": u}
    for split, n in (("train", cfg4["n_train"]), ("val", cfg4["n_val"]), ("test", cfg4["n_test"])):
        X = rng.standard_normal((int(n), d))
        out[f"X_{split}"] = X
        out[f"y_{split}"] = np.tanh(float(cfg4["signal_scale"]) * (X @ u)) + float(cfg4["noise_std"]) * rng.standard_normal(int(n))
    return out


def run_seed(cfg4: dict[str, Any], seed_master: int, k: int) -> pd.DataFrame:
    rng = child_rng(seed_master, STREAM_TASK4, k)
    data = make_data(cfg4, rng)
    net = TwoLayerNet(int(cfg4["d"]), int(cfg4["width"]), rng)
    u = data["u"]
    lr = float(cfg4["lr"])
    rows = []
    for epoch in range(int(cfg4["epochs"]) + 1):
        if epoch % int(cfg4["log_every"]) == 0:
            f_tr, f_va, f_te = (net.predict(data[f"X_{s}"]) for s in ("train", "val", "test"))
            proj = net.W @ u
            rows.append({
                "seed": k, "epoch": epoch,
                "train_mse": float(np.mean((data["y_train"] - f_tr) ** 2)),
                "val_mse": float(np.mean((data["y_val"] - f_va) ** 2)),
                "test_mse": float(np.mean((data["y_test"] - f_te) ** 2)),
                "test_r2_paper": M.r2_paper(data["y_test"], f_te),
                "test_timing_return": M.expected_timing_return(data["y_test"], f_te),
                "test_sharpe_paper": M.sharpe_paper(data["y_test"], f_te),
                "weight_norm": net.complexity(),
                "signal_weight": float(np.mean(proj ** 2)),                                  # mean_j (w_j.u)^2
                "noise_weight": float((np.sum(net.W ** 2) / net.m - np.mean(proj ** 2)) / (net.d - 1)),
            })
        if epoch < int(cfg4["epochs"]):
            net.step(data["X_train"], data["y_train"], lr)
    df = pd.DataFrame(rows)
    df["early_stop_epoch"] = int(df.loc[df["val_mse"].idxmin(), "epoch"])        # chosen on validation only
    return df


def early_stopping_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Per-seed comparison: stop at the validation-MSE minimum vs train to the last epoch."""
    rows = []
    for seed, g in curves.groupby("seed"):
        g = g.set_index("epoch")
        es, last = int(g["early_stop_epoch"].iloc[0]), int(g.index.max())
        for label, ep in (("early_stop", es), ("final_epoch", last)):
            r = g.loc[ep]
            rows.append({"seed": seed, "stopping": label, "epoch": ep, "train_mse": r["train_mse"],
                         "test_mse": r["test_mse"], "test_r2_paper": r["test_r2_paper"],
                         "test_sharpe_paper": r["test_sharpe_paper"], "weight_norm": r["weight_norm"]})
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any], paths: Paths) -> None:
    cfg4 = cfg["task4"]
    if not cfg4.get("enabled", True):
        log.info("Task 4 disabled in config - skipping.")
        return
    log.info("Task 4: width=%d, d=%d, n_train=%d, noise=%.2f, %d epochs x %d seeds", cfg4["width"], cfg4["d"],
             cfg4["n_train"], cfg4["noise_std"], cfg4["epochs"], cfg4["n_seeds"])
    curves = pd.concat([run_seed(cfg4, int(cfg["seed"]), k) for k in range(int(cfg4["n_seeds"]))], ignore_index=True)
    curves.to_csv(paths.tables / "task4_training_curves.csv", index=False)
    es = early_stopping_table(curves)
    es.to_csv(paths.tables / "task4_early_stopping.csv", index=False)
    summ = es.groupby("stopping")[["epoch", "train_mse", "test_mse", "test_r2_paper", "test_sharpe_paper", "weight_norm"]] \
             .agg(["mean", "sem"])
    summ.to_csv(paths.tables / "task4_early_stopping_summary.csv")
    log.info("Early stopping vs final epoch (mean over seeds):\n%s",
             es.groupby("stopping")[["epoch", "train_mse", "test_mse", "test_r2_paper", "test_sharpe_paper"]].mean().round(4))
    from . import plotting
    plotting.plot_training_dynamics(curves, paths.figures / "task4_training_dynamics.png")
    log.info("Task 4 finished.")


def main(argv: list[str] | None = None) -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.output_dir)
    setup_logging(paths.logs, "task4")
    save_run_metadata(cfg, paths)
    run(cfg, paths)


if __name__ == "__main__":
    main()
