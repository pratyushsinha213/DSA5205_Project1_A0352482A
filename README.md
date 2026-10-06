# DSA5205 Project 1 — The Virtue of Complexity in Return Prediction

Reproduction of the key finite-sample phenomena in Kelly, Malamud & Zhou (2024), *The Virtue of
Complexity in Return Prediction*, plus a second model (Lasso / Elastic Net), a hidden-test prediction
pipeline for datasets A/B/C, and a small finance illustration of the Montanari & Urbani (NeurIPS 2025)
learning-vs-overfitting timescale separation.

Everything is Python 3.11+, deterministic (fixed seeds), uses relative paths only, and is runnable
from the repository root.

| Task | What it does | Entry point |
|---|---|---|
| 1 | Misspecified simulation (`c = P/T = 10`, partial observability), ridge / ridgeless sweep over `(c_q, z)` | `python3 -m src.task1_ridge` |
| 2 | Lasso / Elastic Net on **identical** simulated data, compared with ridge | `python3 -m src.task2_alternative` |
| 3 | Hidden-test prediction for A/B/C (forward validation, exact `t,yhat` files) | `python3 -m src.task3_hidden_prediction` |
| 4 | Two-layer network training dynamics on noisy single-index returns | `python3 -m src.task4_nn_dynamics` |
| 5 | Video: use the commands below (see "Video demo path") | `python3 -m src.run_all` |

The report PDF and the video are written by you; this repository produces every figure, table and
prediction file they need, and `report/README_REPORT_OUTLINE.md` maps each figure to the function
that produced it.

---

## 1. Quick start

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
                                     # Windows cmd.exe:     .venv\Scripts\activate.bat
pip install -r requirements.txt

pytest -q                            # unit + end-to-end smoke tests (~1 min)

python3 -m src.task1_ridge            --config configs/default.yaml
python3 -m src.task2_alternative      --config configs/default.yaml     # needs Task 1 first
python3 -m src.task3_hidden_prediction --config configs/default.yaml --student-id YOUR_ID
python3 -m src.task4_nn_dynamics      --config configs/default.yaml

# or everything at once
python3 -m src.run_all --config configs/default.yaml --student-id YOUR_ID
```

Replace `YOUR_ID` with your student ID (it only affects the prediction file names). `--output-dir DIR`
redirects all outputs; `python3 -m src.run_all --skip task2 task4` leaves stages out.

**Quick sanity run (≈1 minute, tiny problem sizes):** `python3 -m src.run_all --config configs/smoke.yaml --output-dir outputs_smoke`

### Task 3 data

Copy the course files into `data/raw/` using exactly these names (no code edits needed):

```
data/raw/train_A.csv  test_A.csv   train_B.csv  test_B.csv   train_C.csv  test_C.csv
```

Schema: train `t,feature1,…,featureP,return`; test `t,feature1,…,featureP`. Datasets that are not
present are skipped with a warning (`--strict` makes that an error). No datasets are included or
fabricated in this repository.

### Runtime

Measured on one CPU core: Task 1 ≈ 1.4 s per replication (≈ 1–2 min for 50), Task 2 ≈ 20 s per
replication (≈ 17 min for 50 on one core), Task 4 ≈ 30 s, Task 3 seconds to minutes depending on the
data size. Replications are independent, so `n_jobs: -1` (the default in `configs/default.yaml`)
spreads them over all cores. Set `n_jobs: 1` for serial execution. Lower
`simulation.n_replications` / `task2.n_replications` for a faster development run (the spec suggests
20); Task 2 always uses the *first* `task2.n_replications` replications of Task 1.

---

## 2. Repository layout

```
├── README.md  requirements.txt  pytest.ini  .gitignore
├── configs/
│   ├── default.yaml          # final experiment parameters (all seeds / grids live here)
│   └── smoke.yaml            # tiny configuration for tests and quick checks
├── data/raw/                 # put train_/test_ A,B,C CSVs here (not versioned)
├── src/
│   ├── config.py             # YAML loading, paths, child RNGs, logging, run metadata
│   ├── simulation.py         # DGP, dense beta*, fixed nested column permutation, data fingerprints
│   ├── models.py             # SVD ridge / ridgeless with the course scaling; Task 3 model factory
│   ├── metrics.py            # R2_paper, timing return, paper SR, variance SR, ||beta||^2
│   ├── validation.py         # chronological hold-out and expanding-window forward splits
│   ├── data.py               # strict CSV loader, feature alignment, prediction writer/validator
│   ├── plotting.py           # all figures; every function reads a saved CSV table
│   ├── task1_ridge.py  task2_alternative.py  task3_hidden_prediction.py  task4_nn_dynamics.py
│   └── run_all.py
├── tests/                    # metrics, ridge scaling, simulation protocol, feature alignment,
│                             # prediction format, validation splits, end-to-end smoke test
├── notebooks/                # thin analysis notebooks; they only call src/ and read saved tables
├── outputs/{figures,tables,predictions,logs}/
└── report/README_REPORT_OUTLINE.md   # report skeleton + figure → code cross-reference
```

---

## 3. Outputs

| Path | Produced by |
|---|---|
| `outputs/tables/task1_results.csv` | one row per (replication, c_q, z): all metrics |
| `outputs/tables/task1_summary.csv` | per (c_q, z): Monte Carlo mean, **std (ddof=1)** and **standard error** (= std/√n) of every metric |
| `outputs/tables/task1_validated_ridge.csv`, `task1_validated_summary.csv` | ridge with `z` chosen on a training-only chronological hold-out |
| `outputs/tables/task1_sanity.csv`, `task1_data_fingerprints.csv` | numerical checks; per-replication SHA-256 data fingerprints |
| `outputs/tables/task2_results.csv`, `task2_summary.csv` | validated Lasso and Elastic Net (selected hyper-parameters stored per row) |
| `outputs/tables/task2_fixed_alpha_results.csv`, `…_summary.csv` | every (l1_ratio, alpha) on the grid, no tuning (analogue of the fixed-z ridge sweep) |
| `outputs/tables/task2_selected_hyperparameters.csv` | median / range of selected alpha, l1_ratio per c_q |
| `outputs/tables/task3_validation_{A,B,C}.csv`, `task3_selected_models.csv` | candidate comparison (all four metrics, fold std, selected row flagged); one row per dataset with training-set diagnostics, selected model and its validation metrics |
| `outputs/predictions/<student-id>_predictions_{A,B,C}.csv` | exactly `t,yhat` |
| `outputs/tables/task4_training_curves.csv`, `task4_early_stopping*.csv` | per-seed, per-epoch dynamics; early-stop vs final epoch |
| `outputs/figures/*.png` | all figures (names listed in `src/plotting.py` and the report outline) |
| `outputs/logs/` | `config_used.yaml` (exact parameters of the run), `environment.yaml` (package versions), per-task logs |

Tables are always written **before** any figure is drawn, and figures are redrawn from the saved CSVs
(`src.plotting.make_task1_figures(paths)` / `make_task2_figures(paths)`), so nothing is plotted from
live, unsaved results and no generated CSV is ever edited by hand.

---

## 4. Reproducibility

* **Master seed** `seed: 5205` (`configs/default.yaml`). Child generators are
  `numpy.random.SeedSequence(entropy=seed, spawn_key=(stream, index))` (`src.config.child_rng`):
  stream 0 = Task 1/2 simulation (index = replication), stream 2 = Task 4 (index = seed number).
  Replication *i* is therefore identical no matter how many replications are requested, and Task 2
  regenerates exactly Task 1's data instead of caching hundreds of MB.
* **Proof of identical data:** Task 1 stores a SHA-256 fingerprint of every replication's
  `(S_train, y_train, S_test, y_test, β*, permutation)`; Task 2 recomputes and **aborts** if any differ.
* **Deterministic Task 3 / Task 4:** no random components (full SVD/PCA, cyclic coordinate descent,
  full-batch gradient descent, seeded initialisation).
* **Relative paths only**; run from the repository root; outputs go to deterministic locations.
* The exact parameters and package versions of every run are saved in `outputs/logs/`.
  Results can differ in the last floating-point digits across BLAS builds.

---

## 5. Implementation details and conventions

**Notation.** `R_{t+1} = S_tᵀβ* + ε_{t+1}`, `S_t ~ N(0, I_P)`, `ε ~ N(0,1)`, `β*` dense Gaussian rescaled
to `‖β*‖² = b* = 0.2` (so `E[R²] = 1 + b*`), `T_train = 300`, `T_test = 2000`, `P = c·T_train = 3000`
(`c = 10`), `P1 = round(c_q · T_train)`.

**Observation protocol (Task 1/2).** One column permutation per replication, fixed for all `c_q`; the
model with complexity `c_q` sees the **first `P1` columns of that permutation** (nested feature sets;
`SimData.observed_columns`, tested in `tests/test_simulation.py`). Features are never reselected per `c_q`.

**Ridge scaling and numerics (`src/models.py`).** `β̂(z) = (XᵀX + z·T_train·I)⁻¹Xᵀy`, always evaluated
through a thin SVD `X = UΣVᵀ` as `V·diag(σ/(σ²+zT))·Uᵀy`. For `z = 0` the same code applies the
Moore–Penrose cut-off (identical to `numpy.linalg.lstsq`), i.e. the minimum-norm least-squares solution.
`XᵀX` is never formed or inverted and no `P×P` matrix is built (cost `O(T²P1)`). Tests compare against
`sklearn.linear_model.Ridge(alpha = z·T)`, the normal equations, the kernel/dual form,
`np.linalg.lstsq` and `np.linalg.pinv`.

**Metrics (`src/metrics.py`).**
`R²_paper = 1 − mean[(R−R̂)²]/mean[R²]` (second moment, not variance) and the equivalent
`(2E[R̂R] − E[R̂²])/E[R²]` (the gap is recorded as `r2_identity_gap`; max ≈ 1e-14);
timing return `R^π = R̂·R`; **paper SR** `= E[R^π]/√E[(R^π)²]` (uncentred); separate, clearly labelled
`SR_var = E[R^π]/sd(R^π)`; `‖β̂‖²`.
*Convention:* a strategy that never takes a position (all-zero predictions, e.g. a fully shrunk Lasso)
has identically zero timing returns; both Sharpe ratios are defined as 0 instead of 0/0.

**Validated ridge (Task 1 extra).** `z` is chosen from the same grid on a chronological hold-out of the
*training* sample only (first 75 % fit, last 25 % validate; penalty `z·T_fit` so `z` keeps its
per-observation meaning), then the model is refitted on all training rows. The "best fixed z" envelope
plot (`task1_best_z_envelope.png`) is selected on Monte Carlo test means and is therefore an **oracle**
diagnostic, labelled as such; it is not a tradable choice.

**Task 2 (Lasso / Elastic Net).** Same simulations, splits, permutation, `c_q` grid and metrics as Task 1.
Hyper-parameters (`alpha` on a log-spaced grid; `l1_ratio ∈ {0.25, 0.5, 0.75, 1}`, where 1 is the pure
Lasso) are tuned on the same training-only chronological hold-out as the validated ridge, using warm-started
coefficient paths. Features are divided by their training standard deviation (**no centring and no
intercept**, matching the zero-mean DGP and the ridge benchmark); coefficients are mapped back to
original units before computing `‖β̂‖²` and predictions. sklearn's objective is
`1/(2n)‖y−Xw‖² + α(ρ‖w‖₁ + (1−ρ)/2‖w‖²)`; because `α` is tuned, no conversion to the ridge `z`
scale is needed (and none is claimed). In addition to the validated choice, the **fixed-hyper-parameter
sweep** over the whole `(l1_ratio, alpha)` grid is saved, which shows the Lasso's behaviour around
`c_q ≈ 1` even when validation shrinks it to zero.

**Task 3.** `src/data.py` validates and aligns features (rejects missing, extra and renamed features,
duplicate or non-increasing `t`, NaN/non-numeric values; re-orders test features to the training order;
keeps test `t` strings byte-for-byte). Candidates: Ridge (course `z·T` scaling), Lasso, Elastic Net,
PCA+Ridge — a deliberately small grid defined in `configs/default.yaml`. Selection uses **expanding-window
forward validation** (`n_folds` blocks, validation always strictly later than training) with the fold-mean of
`task3.selection_metric` (default `r2_paper`; `sharpe_paper` / `timing_return` also available); all four
metrics are reported for every candidate in `task3_validation_X.csv`. Scalers, PCA and every hyper-parameter
are fitted inside the training part of each fold (no leakage); the selected model is refitted on all
training rows. Before writing, the predictions are asserted to have the right count, be finite, and
match the test `t` exactly; afterwards the written file is re-read and re-validated (header exactly `t,yhat`,
UTF-8, no index). A warning is logged if no candidate beats the zero predictor on validation `R²`.

**Task 4 (`src/task4_nn_dynamics.py`).** `R_{t+1} = tanh(s·uᵀX_t) + ε` with a unit direction `u`, a width-`m`
network `f(x) = (1/m)Σ_j a_j tanh(w_jᵀx/√d)` trained by full-batch gradient descent in the mean-field time
scaling (step size × m), implemented in NumPy (no deep-learning framework needed → deterministic and
light). It records train/validation/test MSE, test `R²_paper`, test paper SR, a weight-norm complexity proxy
and the weight mass along `u` versus orthogonal directions every `log_every` epochs; early stopping uses
**validation MSE only**. *Disclosure:* the parameters in `configs/default.yaml` (`noise_std 1.5`, `n_train 600`,
`d 20`, `m 100`, `lr 0.5`) were chosen after a small 2-seed pilot scan because the first guess (`noise 3`,
`n 400`) was too noisy to learn anything before overfitting; a pilot with `lr 0.1` showed much milder
late overfitting. Whether and how strongly late overfitting appears therefore depends on step size, noise and
sample size — state this in the report, and do not present the experiment as proof about real markets.

---

## 6. Tests

```bash
pytest -q
```

Covers: R² identity; second-moment denominator; paper vs variance Sharpe; ridge vs
`sklearn.Ridge(alpha=zT)` and the normal equations; ridgeless minimum-norm / interpolation vs
`lstsq` and `pinv`; stability at `c_q = 1`; DGP normalisation, determinism and nested permutation
protocol; feature alignment failures (missing, extra, renamed, duplicate `t`, NaN, unordered `t`);
prediction file schema, order, finiteness and no index; forward-validation splits; and an end-to-end
smoke run of every stage (Task 3 uses synthetic stand-in CSVs created inside the test's temp folder —
never in `data/raw`).

---

## 7. Video demo path (8–10 min)

1. Create/activate the environment, `pip install -r requirements.txt`, `pytest -q`.
2. Show `configs/default.yaml` and `outputs/logs/config_used.yaml`.
3. Run `python3 -m src.run_all --config configs/smoke.yaml --output-dir outputs_demo --student-id YOUR_ID`
   live (≈1 minute), or a small-replication Task 1.
4. Present the saved full-run figures in `outputs/figures/` (Task 1 four plots, zoom, ridgeless vs validated; Task 2
   comparison and sparsity; Task 4 dynamics) and `outputs/tables/task3_validation_*.csv`.
5. Show an exact prediction CSV (`head outputs/predictions/<id>_predictions_A.csv`).

Figures and files shown must come from the same run you submit.

---

## 8. Not included / to do by you

* Datasets A/B/C (provided by the instructor) and therefore the actual prediction CSVs.
* Your student ID (pass `--student-id`).
* The report PDF, its interpretation of the results and the independent finance insight for Task 4, and the video.
  Do not claim hidden-test performance; only validation results exist.
