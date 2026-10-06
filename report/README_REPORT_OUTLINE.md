# Report outline (5–15 pages) and figure → code traceability

State in the report the **final parameter values actually used** (copy from `outputs/logs/config_used.yaml`),
the master seed (5205), and the number of Monte Carlo replications. Say explicitly whether error bands are
standard deviations or standard errors (the pipeline's shaded bands are **±1 standard error**; both are in
`task1_summary.csv`).

## 1. Introduction
Research question (does complexity help return prediction?), why `c = P/T` matters, KMZ in one paragraph.

## 2. Task 1 — Ridge simulation
DGP and notation (`src/simulation.py`), partial observability and the nested fixed-permutation protocol,
ridge scaling `(X'X + zT I)` and SVD ridgeless solver (`src/models.py`), metrics (`src/metrics.py`),
grid, then results from `task1_summary.csv`. Discuss only what the tables/figures show: instability and
large `‖β̂‖²` near `c_q ≈ 1`, double-descent-like `R²`, stabilising effect of `z`, and whether timing
performance improves with `c_q`. Compare ridgeless vs validated ridge.

## 3. Task 2 — Lasso / Elastic Net
Model choice (dense `β*` vs L1 sparsity), tuning on training-only chronological hold-out, identical data
(fingerprint check), comparison with ridge, sparsity (`n_nonzero`), behaviour near interpolation,
departures from the KMZ ridge benchmark and why. The fixed-α sweep shows behaviour that validation hides.

## 4. Task 3 — Hidden prediction
Diagnostics on training data (`task3_selected_models.csv`), forward-validation design, candidate
families and grids, selected models/hyper-parameters for A/B/C and why, file validation. **No hidden-test
performance claims.**

## 5. Task 4 — Neural-network generalisation
Montanari & Urbani: timescale separation; feature learning vs unlearning; early stopping as regularisation.
Finance question and the experiment (`task4_nn_dynamics.py`, figure below), limitations (two-layer,
Gaussian single-index, gradient-flow theory vs non-stationary, heavy-tailed, serially dependent, costly real
markets). Note what the run does and does not show: in the saved run the late-phase degradation coincides with
growth of weights in orthogonal (noise) directions and of the complexity proxy; check `signal_weight` in the
table before claiming literal "feature unlearning". Develop **your own** finance implication.

## 6. Conclusion

---

## Figure → producing function → source table

| Figure (`outputs/figures/`) | Function | Source table |
|---|---|---|
| `task1_r2_vs_cq.png` | `src.plotting.plot_metric_vs_cq(summary, "r2_paper", …)` | `tables/task1_summary.csv` |
| `task1_timing_return_vs_cq.png` | `plot_metric_vs_cq(…, "timing_return", …)` | `task1_summary.csv` |
| `task1_sharpe_vs_cq.png` | `plot_metric_vs_cq(…, "sharpe_paper", …)` | `task1_summary.csv` |
| `task1_beta_norm_vs_cq.png` | `plot_metric_vs_cq(…, "beta_norm_sq", …)` | `task1_summary.csv` |
| `task1_sharpe_var_vs_cq.png` | `plot_metric_vs_cq(…, "sharpe_var", …)` | `task1_summary.csv` |
| `task1_heatmap_{sharpe,r2}.png` | `plot_heatmap` | `task1_summary.csv` |
| `task1_best_z_envelope.png` (oracle) | `plot_best_z_envelope` | `task1_summary.csv` |
| `task1_zoom_near_interpolation.png` | `plot_zoom` | `task1_summary.csv` |
| `task1_ridgeless_vs_validated.png` | `plot_ridgeless_vs_validated` | `task1_summary.csv`, `task1_validated_summary.csv` |
| `task2_ridge_vs_alternative_{r2,timing_return,sharpe,beta_norm}.png` | `plot_ridge_vs_alternative` | `task1_summary.csv`, `task1_validated_summary.csv`, `task2_summary.csv` |
| `task2_sparsity_vs_cq.png` | `plot_sparsity` | `task2_summary.csv` |
| `task2_lasso_fixed_alpha_*_vs_cq.png` | `plot_metric_vs_cq(…, group_col="alpha")` | `task2_fixed_alpha_summary.csv` |
| `task4_training_dynamics.png` | `plot_training_dynamics` | `task4_training_curves.csv` |

Simulation/estimation entry points: `src.task1_ridge.evaluate_replication`, `src.task2_alternative.evaluate_cq`,
`src.task3_hidden_prediction.evaluate_candidates` / `process_dataset`, `src.task4_nn_dynamics.run_seed`.
Whole-figure rebuild from saved tables: `src.plotting.make_task1_figures(paths)`, `make_task2_figures(paths)`.
