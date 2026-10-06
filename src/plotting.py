"""Publication-quality figures.  Every function reads a saved CSV table, never live results.

Figure -> producing function (cross-referenced in report/README_REPORT_OUTLINE.md):
    task1_r2_vs_cq.png, task1_timing_return_vs_cq.png, task1_sharpe_vs_cq.png,
    task1_beta_norm_vs_cq.png                    -> plot_metric_vs_cq
    task1_heatmap_{sharpe,r2}.png                -> plot_heatmap
    task1_best_z_envelope.png                    -> plot_best_z_envelope
    task1_zoom_near_interpolation.png            -> plot_zoom
    task1_ridgeless_vs_validated.png             -> plot_ridgeless_vs_validated
    task2_ridge_vs_alternative_*.png             -> plot_ridge_vs_alternative
    task4_training_dynamics.png                  -> plot_training_dynamics
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

from .config import Paths  # noqa: E402

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
DPI = 200
METRIC_LABELS = {
    "r2_paper": r"Out-of-sample $R^2_{paper}$",
    "timing_return": r"Expected timing return $E[R^\pi]$",
    "sharpe_paper": r"Sharpe ratio $E[R^\pi]/\sqrt{E[(R^\pi)^2]}$",
    "sharpe_var": r"Variance-based Sharpe $E[R^\pi]/\mathrm{sd}(R^\pi)$",
    "beta_norm_sq": r"$\|\hat\beta\|_2^2$",
    "n_nonzero": "Number of non-zero coefficients",
}
MODEL_STYLE = {
    "ridge_z0": ("Ridgeless (z=0)", "#d62728", "-"),
    "ridge_validated": ("Ridge, validated z", "#1f77b4", "-"),
    "ridge_oracle": ("Ridge, best fixed z (oracle, test-selected)", "#7f7f7f", ":"),
    "lasso": ("Lasso (validated)", "#2ca02c", "--"),
    "elastic_net": ("Elastic net (validated)", "#ff7f0e", "--"),
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _cq_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"Observed complexity $c_q = P_1 / T_{train}$ (log scale)")


def _yscale(ax: plt.Axes, metric: str) -> None:
    if metric == "beta_norm_sq":
        ax.set_yscale("log")
    elif metric == "n_nonzero":
        ax.set_yscale("symlog", linthresh=1)
    elif metric == "r2_paper":
        ax.set_yscale("symlog", linthresh=0.05)   # ridgeless R^2 near c_q = 1 is hugely negative


def _tidy_r2_axis(ax: plt.Axes, top_data: float) -> None:
    """Readable symlog R^2 axis: explicit ticks, top limit just above the best curve."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_ylim(top=max(0.02, 1.6 * top_data))
    lo = ax.get_ylim()[0]
    ticks = [t for t in (-1000, -100, -10, -1, -0.1, 0.0, 0.02, 0.05) if lo <= t <= ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def plot_metric_vs_cq(summary: pd.DataFrame, metric: str, path: Path, band: str = "se",
                      group_col: str = "z", title: str = "Ridge") -> None:
    """One line per hyper-parameter value (default: shrinkage z); band is +/- 1 Monte Carlo s.e."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    levels = sorted(summary[group_col].unique())
    colors = sns.color_palette("viridis", len(levels))
    for color, g in zip(colors, levels):
        d = summary[summary[group_col] == g].sort_values("cq")
        m, e = d[f"{metric}_mean"].to_numpy(), d[f"{metric}_{band}"].to_numpy()
        ax.plot(d["cq"], m, marker="o", ms=3, lw=1.4, color=color, label=f"{group_col}={g:g}")
        ax.fill_between(d["cq"], m - e, m + e, color=color, alpha=0.15, lw=0)
    _cq_axis(ax)
    _yscale(ax, metric)
    if metric == "r2_paper":
        _tidy_r2_axis(ax, float(summary["r2_paper_mean"].max()))
    ax.set_ylabel(METRIC_LABELS[metric])
    n = int(summary["n_reps"].max())
    ax.set_title(f"{title}, c = P/T = 10, mean over {n} replications (band: ±1 s.e.)")
    ax.legend(title="penalty", ncol=2, fontsize=8, loc="best")
    _save(fig, path)


def plot_heatmap(summary: pd.DataFrame, metric: str, path: Path) -> None:
    piv = summary.pivot(index="z", columns="cq", values=f"{metric}_mean").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    kw = {}
    if metric == "r2_paper":
        kw["norm"] = SymLogNorm(linthresh=0.05, vmin=min(piv.min().min(), -0.05), vmax=max(piv.max().max(), 0.05))
    sns.heatmap(piv, ax=ax, cmap="RdBu", center=0 if metric != "r2_paper" else None, annot=len(piv.columns) <= 15,
                fmt=".2f", annot_kws={"size": 6}, cbar_kws={"label": METRIC_LABELS[metric]}, **kw)
    ax.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=45)
    ax.set_xlabel(r"$c_q$")
    ax.set_ylabel("z")
    _save(fig, path)


def plot_best_z_envelope(summary: pd.DataFrame, path: Path) -> None:
    """Best fixed z per c_q by mean Sharpe (selected on the Monte Carlo test means: an oracle envelope)."""
    idx = summary.groupby("cq")["sharpe_paper_mean"].idxmax()
    best = summary.loc[idx].sort_values("cq")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(best["cq"], best["sharpe_paper_mean"], "o-", color="#1f77b4")
    axes[0].set_ylabel("Best-z Sharpe (oracle envelope)")
    axes[1].step(best["cq"], best["z"], where="mid", color="#d62728", marker="o")
    axes[1].set_ylabel("Sharpe-maximising z")
    axes[1].set_yscale("symlog", linthresh=0.1)
    for ax in axes:
        _cq_axis(ax)
    _save(fig, path)


def plot_zoom(summary: pd.DataFrame, path: Path, lo: float = 0.75, hi: float = 1.5) -> None:
    """Four-panel zoom around the interpolation threshold c_q = 1."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    zs = sorted(summary["z"].unique())
    colors = sns.color_palette("viridis", len(zs))
    sub = summary[(summary["cq"] >= lo) & (summary["cq"] <= hi)]
    for ax, metric in zip(axes.ravel(), ["r2_paper", "timing_return", "sharpe_paper", "beta_norm_sq"]):
        for color, z in zip(colors, zs):
            d = sub[sub["z"] == z].sort_values("cq")
            ax.plot(d["cq"], d[f"{metric}_mean"], "o-", ms=3, color=color, label=f"z={z:g}")
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        _yscale(ax, metric)
        ax.set_xlabel(r"$c_q$")
        ax.set_ylabel(METRIC_LABELS[metric])
    axes[0, 0].legend(ncol=2, fontsize=7)
    fig.suptitle(rf"Zoom around interpolation, $c_q\in[{lo},{hi}]$")
    _save(fig, path)


def plot_ridgeless_vs_validated(summary: pd.DataFrame, validated: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    z0 = summary[summary["z"] == 0].sort_values("cq")
    val = validated.sort_values("cq")
    for ax, metric in zip(axes, ["r2_paper", "sharpe_paper", "beta_norm_sq"]):
        for df, key in ((z0, "ridge_z0"), (val, "ridge_validated")):
            lab, col, ls = MODEL_STYLE[key]
            ax.plot(df["cq"], df[f"{metric}_mean"], marker="o", ms=3, color=col, ls=ls, label=lab)
            ax.fill_between(df["cq"], df[f"{metric}_mean"] - df[f"{metric}_se"],
                            df[f"{metric}_mean"] + df[f"{metric}_se"], color=col, alpha=0.15, lw=0)
        _cq_axis(ax)
        _yscale(ax, metric)
        ax.set_ylabel(METRIC_LABELS[metric])
    axes[0].legend(fontsize=8)
    _save(fig, path)


def make_task1_figures(paths: Paths) -> None:
    """Rebuild every Task 1 figure from ``outputs/tables``."""
    summary = pd.read_csv(paths.tables / "task1_summary.csv")
    validated = pd.read_csv(paths.tables / "task1_validated_summary.csv")
    for metric, name in [("r2_paper", "r2"), ("timing_return", "timing_return"),
                         ("sharpe_paper", "sharpe"), ("beta_norm_sq", "beta_norm")]:
        plot_metric_vs_cq(summary, metric, paths.figures / f"task1_{name}_vs_cq.png")
    plot_metric_vs_cq(summary, "sharpe_var", paths.figures / "task1_sharpe_var_vs_cq.png")
    plot_heatmap(summary, "sharpe_paper", paths.figures / "task1_heatmap_sharpe.png")
    plot_heatmap(summary, "r2_paper", paths.figures / "task1_heatmap_r2.png")
    plot_best_z_envelope(summary, paths.figures / "task1_best_z_envelope.png")
    plot_zoom(summary, paths.figures / "task1_zoom_near_interpolation.png")
    plot_ridgeless_vs_validated(summary, validated, paths.figures / "task1_ridgeless_vs_validated.png")


def plot_ridge_vs_alternative(task1_summary: pd.DataFrame, task1_validated: pd.DataFrame,
                              task2_summary: pd.DataFrame, metric: str, path: Path) -> None:
    """Ridge benchmarks vs validated Lasso / elastic net on identical replications."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    curves = {
        "ridge_z0": task1_summary[task1_summary["z"] == 0],
        "ridge_validated": task1_validated,
    }
    for model in task2_summary["model"].unique():
        curves[model] = task2_summary[task2_summary["model"] == model]
    for key, df in curves.items():
        lab, col, ls = MODEL_STYLE.get(key, (key, None, "-"))
        df = df.sort_values("cq")
        m = df[f"{metric}_mean"].to_numpy()
        e = df[f"{metric}_se"].to_numpy()
        ax.plot(df["cq"], m, marker="o", ms=3, color=col, ls=ls, label=lab)
        ax.fill_between(df["cq"], m - e, m + e, color=col, alpha=0.15, lw=0)
    _cq_axis(ax)
    _yscale(ax, metric)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_sparsity(task2_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in task2_summary["model"].unique():
        d = task2_summary[task2_summary["model"] == model].sort_values("cq")
        lab, col, ls = MODEL_STYLE.get(model, (model, None, "-"))
        ax.plot(d["cq"], d["n_nonzero_mean"], marker="o", ms=3, color=col, ls=ls, label=lab)
        ax.plot(d["cq"], d["p1"], color="k", lw=0.6, ls=":")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Non-zero coefficients (dotted: P1 observed)")
    _cq_axis(ax)
    ax.legend(fontsize=8)
    _save(fig, path)


def make_task2_figures(paths: Paths) -> None:
    t1 = pd.read_csv(paths.tables / "task1_summary.csv")
    t1v = pd.read_csv(paths.tables / "task1_validated_summary.csv")
    t2 = pd.read_csv(paths.tables / "task2_summary.csv")
    for metric, name in [("r2_paper", "r2"), ("timing_return", "timing_return"),
                         ("sharpe_paper", "sharpe"), ("beta_norm_sq", "beta_norm")]:
        plot_ridge_vs_alternative(t1, t1v, t2, metric, paths.figures / f"task2_ridge_vs_alternative_{name}.png")
    plot_sparsity(t2, paths.figures / "task2_sparsity_vs_cq.png")
    fixed = pd.read_csv(paths.tables / "task2_fixed_alpha_summary.csv")
    lasso = fixed[fixed["l1_ratio"] == 1.0]
    for metric, name in [("r2_paper", "r2"), ("sharpe_paper", "sharpe"), ("beta_norm_sq", "beta_norm"),
                         ("n_nonzero", "n_nonzero")]:
        plot_metric_vs_cq(lasso, metric, paths.figures / f"task2_lasso_fixed_alpha_{name}_vs_cq.png",
                          group_col="alpha", title="Lasso, fixed alpha")


def plot_training_dynamics(curves: pd.DataFrame, path: Path) -> None:
    """Eight-panel training-dynamics figure (mean over seeds, band = +/-1 s.e.)."""
    panels = [
        ("train_mse", "Train MSE"), ("val_mse", "Validation MSE"), ("test_mse", "Test MSE"),
        ("test_r2_paper", r"Test $R^2_{paper}$"), ("test_sharpe_paper", "Test paper Sharpe"),
        ("weight_norm", r"Complexity proxy $(\|a\|^2+\|W\|_F^2)/m$"),
        ("signal_weight", r"Weight mass along signal $u$: mean$_j (w_j\cdot u)^2$"),
        ("noise_weight", "Weight mass per orthogonal direction"),
    ]
    g = curves.groupby("epoch")
    mean, se = g.mean(numeric_only=True), g.sem(numeric_only=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        ax.plot(mean.index, mean[col], lw=1.6, color="#1f77b4")
        ax.fill_between(mean.index, mean[col] - se[col], mean[col] + se[col], color="#1f77b4", alpha=0.2, lw=0)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab, fontsize=9)
    if "early_stop_epoch" in curves.columns:
        es = float(curves.groupby("seed")["early_stop_epoch"].first().median())
        for ax in axes.ravel():
            ax.axvline(es, color="#d62728", ls="--", lw=1, label="median early-stop epoch (validation MSE)")
        axes[0, 0].legend(fontsize=7)
    n = curves["seed"].nunique()
    fig.suptitle(f"Two-layer network on noisy single-index returns: training dynamics (mean of {n} seeds, band ±1 s.e.)")
    _save(fig, path)
