# PHASE_4_ML_MODELING.md
## ML Modeling & Evaluation

Status: **Pipeline implemented and unit-tested against synthetic fixtures. Model implementation is complete, but empirical model performance has not yet been established because real Mininet-generated data is unavailable.** Nothing in this document should be read as reporting real-world accuracy — none exists yet.

---

## 1. Inspected State Before Starting (Important Discrepancy)

The Phase 4 brief asked to inspect and reuse "Phase 3" components (feature engineering, leakage audit, target generation, temporal/group-aware splitting, dataset validation). Before writing any code, `src/`, `scripts/`, `tests/`, `data/`, `reports/`, and `docs/` were inspected directly: **no Phase 3 exists anywhere in this repository** — no `docs/PHASE_3*.md`, no `src/features`, no `reports/`, no combined dataset. `data/raw/` contains only a `.gitkeep` placeholder; no real Mininet experiment has ever been executed (this remains blocked on the WSL2/Mininet environment dependency documented in `ENVIRONMENT_SETUP.md`).

What *does* already exist and was reused as-is, without duplication:
- **Target generation & leakage-safe shift**: `network.targets.add_next_interval_target()` — inspected, no bug found, zero changes made.
- **Row/leakage validation**: `network.validation.validate_rows()` and `validate_no_leakage()` (the latter added in Phase 2) — reused directly inside the new `ml.dataset.load_dataset()`.
- **Schema**: `network.schema.FEATURE_COLUMNS` / `TARGET_COLUMN` / `CSV_COLUMNS` — the single source of truth for what a "feature" and "target" are; Phase 4 never redefines these.

What did **not** exist and had to be built as new Phase 4 plumbing (documented here rather than silently invented as if it were a reused "Phase 3"):
- Combining multiple per-experiment CSVs into one modeling table (`ml.dataset`).
- Group-aware train/test splitting (`ml.split`) — PROJECT_PLAN.md Section 17 describes this requirement, but no code implementing it existed until now.
- No additional feature engineering (lag/rolling features, described as a *future* idea in PROJECT_PLAN.md Section 11) was added — the model trains directly on Phase 1/2's raw per-interval measurement columns (`FEATURE_COLUMNS`). This keeps the pipeline honest about what actually exists versus what was aspirational in the original plan.

## 2. ML Formulation (unchanged from PROJECT_PLAN.md)

- **Type**: regression. Target remains `next_interval_packet_loss_pct` (`TARGET_COLUMN` in code) — never replaced with classification as the primary task.
- **Research question**: can current/previous network conditions predict packet loss in the next interval?
- **Optional derived layer**: LOW/MODERATE/HIGH risk labels computed *from* the regression output (Section 13) — not a replacement for it.

## 3. Target and Features

Unchanged from Phase 1/2 (`network/schema.py`):
- **Target** (`target_next_packet_loss_pct`): `packet_loss_pct` measured one interval ahead (`horizon=1` by default), within the same `experiment_id`, added by `add_next_interval_target()`.
- **Features** (`FEATURE_COLUMNS`, 15 columns): 5 configured-condition columns (`bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts`, `traffic_type`, `active_connections`) known before the run starts, plus 10 measured current-interval columns (`current_rtt_ms`, `current_jitter_ms`, `throughput_mbps`, `bandwidth_utilization_pct`, `packet_rate_pps`, `queue_length`, `retransmissions`, `packets_sent`, `packets_received`, `packet_loss_pct`). `traffic_type` is the only categorical feature; the other 14 are numeric.

## 4. Split Strategy (`src/ml/split.py`)

Two group-aware strategies, both operating on **whole `experiment_id` groups**, never individual rows:

- **`chronological_split`** (default): orders experiments by their earliest timestamp; the latest `test_fraction` of experiments become the test set. Simulates "train on past data, predict on future data."
- **`random_group_split`**: randomly holds out whole experiment groups with a fixed seed (default 42), for reproducibility. Tests generalization across configurations rather than time.

A plain row-level `train_test_split(..., random_state=...)` is never used — consecutive intervals within one experiment are highly autocorrelated (same configured conditions, evolving queue state), and splitting at the row level would put adjacent rows from the same run's trajectory on both sides of the boundary.

**Defense in depth**: `assert_no_group_overlap()` is called by both split functions internally, and is unit-tested with a deliberately corrupted split (`test_assert_no_group_overlap_catches_a_deliberately_broken_split`) to confirm it actually detects overlap rather than trivially passing.

## 5. Baseline

**`NaivePersistenceBaseline`** (`src/ml/baselines.py`): predicts `packet_loss_next = packet_loss_current`, i.e. "loss doesn't change from now." This is the meaningful reference point the brief asked for — a real ML model that can't beat "no change" isn't learning network dynamics, regardless of how it compares to the other candidate models. It participates in the same `train_and_evaluate()` loop and comparison table as every other model, not as a separate, differently-evaluated afterthought.

## 6. Candidate Models (`src/ml/models.py`)

| Model | Preprocessing | Notes |
|---|---|---|
| Linear Regression | Imputation + **StandardScaler** (numeric), one-hot (categorical) | Baseline ML reference; scaling matters here since features span very different magnitudes (e.g. `bottleneck_bw_mbps` ~1-5 vs `current_rtt_ms` ~tens of ms) |
| Decision Tree Regressor | Imputation only (no scaling), `max_depth=8` | Simple, interpretable, visualizable |
| Random Forest Regressor | Imputation only, `n_estimators=200` | Primary candidate per PROJECT_PLAN.md |
| Gradient Boosting Regressor | Imputation only | scikit-learn's `GradientBoostingRegressor` |

All use `RANDOM_SEED = 42` (`src/ml/models.py`) for reproducibility. **XGBoost was not added** — the brief says to consider it only if existing results justify it, and with no real data trained yet (Section 1), there is nothing to justify it against; `GradientBoostingRegressor` already covers the boosted-trees family for now.

Every model factory in `MODEL_REGISTRY` is a zero-argument callable returning a **fresh, unfitted** `sklearn.Pipeline` — verified by `test_factories_produce_independent_unfitted_instances`, so no fitted state leaks between test runs or repeated calls.

## 7. Preprocessing (`src/ml/preprocessing.py`)

A single `ColumnTransformer` (`build_preprocessor(scale_numeric)`):
- Numeric features: median imputation, **+ StandardScaler only when `scale_numeric=True`** (Linear Regression only — "do not blindly scale everything"; tree-based models are invariant to monotonic rescaling).
- Categorical feature (`traffic_type`): most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`, `sparse_output=False`).

**Leakage prevention**: this is an `sklearn.Pipeline`/`ColumnTransformer`, always used as `pipeline.fit(X_train, y_train)` then `pipeline.predict(X_test)`. sklearn's own contract guarantees imputation medians, scaler mean/std, and one-hot categories are learned from `X_train` alone and only ever *applied* (never re-fit) to `X_test` — verified directly by `test_imputer_statistics_come_from_train_only` (fits on a train set with a uniform value, transforms a test set with an entirely different/missing value, and confirms the imputed result reflects train's statistic) and `test_transform_never_refits_on_test_data`.

## 8. Training (`src/ml/train.py`, `scripts/train_models.py`)

`train_and_evaluate(train_df, test_df, ...)`:
1. Splits `train_df`/`test_df` into `X`/`y` using `FEATURE_COLUMNS`/`TARGET_COLUMN`.
2. Fits the naive baseline and every registered model **on train only**.
3. Predicts on **test only**.
4. Computes MAE/RMSE/R² per model.
5. Returns fitted models, predictions, and metrics together — nothing here reads `X_test`/`y_test` during any `.fit()` call.

`scripts/train_models.py --input data/raw`:
1. `discover_csv_files()` — if none found, **prints a clear message and exits 1, writing nothing**. Verified on this machine (Section 16).
2. `load_dataset()` — combines + validates (reuses Phase 2's `validate_rows`/`validate_no_leakage`), drops end-of-experiment rows with no target.
3. `chronological_split()` (default) or `random_group_split()`.
4. `train_and_evaluate()`, prints and saves the comparison table.
5. Plots + feature importance + risk distribution + model/metadata artifacts for the lowest-MAE model (Section 12 explains why this is a starting point, not the final say).

## 9. Evaluation (`src/ml/evaluate.py`)

- `compute_regression_metrics()`: MAE, RMSE, R² — unit-tested against known values (e.g. all-correct predictions → MAE=RMSE=0, R²=1).
- `comparison_table()`: sorted by **MAE**, not R² — "do not select a model using only R²."
- `residual_stats()`: mean/std residual, median/max absolute error, and `pct_within_1pt`/`pct_within_5pt` (percentage of predictions within 1 or 5 percentage points of actual) — added specifically because packet loss is often near-zero, so aggregate metrics alone can hide whether errors are uniformly small or mostly-tiny-with-occasional-large-misses.

## 10. Feature Importance (`src/ml/evaluate.py`)

Two methods, both implemented:
- **`random_forest_feature_importance()`**: native impurity-based importance, correctly labeled in *post-transform* space via the preprocessor's own `get_feature_names_out()` (so one-hot-expanded `traffic_type` columns are labeled correctly rather than misattributed).
- **`permutation_feature_importance()`**: `sklearn.inspection.permutation_importance` run **only on already-held-out `X_test`/`y_test`** (from the group-aware split), operating in raw `FEATURE_COLUMNS` space. No future information is used — the test set was already constructed to exclude any group present in training.

## 11. Prediction Quality Visualizations (`src/ml/visualize.py`)

`plot_actual_vs_predicted`, `plot_residual_distribution`, `plot_feature_importance`, `plot_prediction_timeline` (optional, one experiment at a time). All use matplotlib's non-interactive `Agg` backend. **None of these functions know whether their input is real or fixture data** — that responsibility sits with the caller: `scripts/train_models.py` only ever points them at `reports/modeling/` when running against a real, loaded dataset; every test that exercises these functions writes to `tmp_path` instead (see `test_full_pipeline_runs_end_to_end_on_fixture_data`). `reports/modeling/` currently contains only a `.gitkeep` — no chart in this repository was generated from real data, because none exists yet.

## 12. Model Selection Criteria

**Not automated as a single "winner."** `scripts/train_models.py` reports the lowest-MAE model as a starting point (explicitly labeled as such in its own output), but final selection should weigh, per the brief: MAE/RMSE/R², stability across repeated runs/folds, generalization to unseen experiments (this is exactly what the group-aware test split is for), interpretability (Decision Tree > Random Forest > Gradient Boosting > Linear Regression, roughly, for a non-ML audience explaining results in a viva), and computational cost (Linear Regression and Decision Tree are near-instant; Random Forest and Gradient Boosting cost more, though trivially so at this dataset's expected scale). **This selection cannot actually be made yet** — there is no real comparison table to weigh these criteria against (Section 16).

## 13. Risk Classification (`src/ml/risk.py`)

Derived from the regression prediction, never replacing it. `RiskThresholds(low_max=0.1, moderate_max=1.0)` (percent) by default:
- **LOW**: predicted loss ≤ 0.1%
- **MODERATE**: 0.1% < predicted loss ≤ 1.0%
- **HIGH**: predicted loss > 1.0%

**Justification**: these follow widely used real-time-traffic quality guidance (ITU-T-style voice/video quality thresholds) — loss under ~0.1% is generally imperceptible for interactive traffic, 0.1–1% causes noticeable but often tolerable degradation, and above ~1% commonly degrades VoIP/video-conferencing quality enough to be considered a real risk. These are round, defensible, **fixed a priori domain thresholds** — they are not fit or tuned against any dataset (train or test), which would let the "risk" boundary quietly leak information about a specific dataset's distribution back into what is supposed to be a project design decision.

## 14. Model Artifacts (`src/ml/artifacts.py`)

- `save_model()` / `load_model()`: joblib serialization of the fitted `Pipeline` (preprocessing + estimator together, so a reloaded model needs no separate preprocessing step).
- `save_metadata()` / `load_metadata()`: a `ModelMetadata` JSON sidecar with exactly `model_name`, `feature_columns`, `target_column`, `training_config`, `metrics` — verified by `test_saved_metadata_contains_no_unexpected_keys` that nothing else (e.g. an environment variable or secret) sneaks in.
- Round-trip correctness verified directly: `test_save_and_load_model_predictions_match` asserts a reloaded model produces **identical** predictions to the original, not just "loads without error."

`models/` currently contains only `.gitkeep` — no real trained model exists yet.

## 15. Testing

44 new tests, 110 total in the repo (66 from Phase 1/2, unchanged and still passing), **zero requiring Mininet or real data**:

| File | Covers |
|---|---|
| `tests/ml_fixtures.py` | Synthetic TEST FIXTURE generator (explicitly labeled, never used to claim real performance) — provides both the raw (Phase-1-CSV-shaped) and pre-labeled (null-target rows dropped) variants |
| `tests/test_dataset.py` | CSV discovery/combination/validation, including a deliberately leakage-corrupted fixture that must be rejected |
| `tests/test_split.py` | Group-aware split correctness, zero-overlap guarantee (including a deliberately broken split to prove the guard works), determinism, chronological ordering |
| `tests/test_preprocessing.py` | Fit-on-train-only leakage prevention, scaler on/off, unknown category handling |
| `tests/test_models.py` | Every registered model fits/predicts on fixture data, factories are independent, reproducibility given fixed seed |
| `tests/test_evaluate.py` | Metric math against known values, comparison table sorting, both feature-importance methods |
| `tests/test_risk.py` | Threshold boundaries, custom thresholds, invalid threshold rejection |
| `tests/test_artifacts.py` | Save/load round-trips for model and metadata, no unexpected metadata keys |
| `tests/test_train_pipeline.py` | Full integration: dataset → split → train → evaluate → importance → risk → plots, on fixture data only, asserting **only structural correctness** (shapes, types, no NaNs, files exist) — never a performance threshold |

**Three real bugs were found and fixed while writing these tests** (Section 17) — none were pre-existing Phase 1/2 bugs; all were in the new Phase 4 code being tested for the first time.

Executed: `py -3 -m pytest tests/ -v` → **110 passed, 0 failed**.

## 16. Real Data / Real Performance

**No real dataset exists.** `data/raw/` contains only `.gitkeep`. Running the actual training command on this machine:

```
$ python3 scripts/train_models.py
No CSV files found under .../data/raw.
No real Mininet-generated dataset exists yet -- refusing to train on nothing.
Run scripts/generate_dataset.py against a real Mininet environment first
(see docs/PHASE_2_DATA_GENERATION.md and docs/ENVIRONMENT_SETUP.md).
```
— confirmed on this machine, exit code 1, and `models/` / `reports/modeling/` verified unchanged (still only `.gitkeep`) afterward. **No numbers are reported here because none exist.** This is expected and directly required by the Data Rule at the top of the Phase 4 brief.

## 17. Bugs Found During Implementation (reported per instructions, not silently fixed)

All three were caught by running the test suite, not by inspection alone — all in **new Phase 4 code**, none in the reused Phase 1/2 modules:

1. **Fixture/loader shape mismatch**: the first version of `ml_fixtures.make_fixture_dataframe()` pre-dropped end-of-experiment (null-target) rows, which doesn't match the shape of a real Phase 1 CSV. This made `validate_no_leakage()` (correctly) flag the second-to-last row of each fixture experiment as anomalous, since its legitimate target appeared to point at an interval that no longer existed. Fixed by making the fixture mirror the real CSV shape, and adding a second `make_labeled_fixture_dataframe()` helper for tests that need pre-cleaned, ready-to-train data.
2. **NaN vs `None` after a pandas round-trip**: `ml.dataset.load_dataset()`'s original `combined.where(pd.notnull(combined), None)` silently failed to convert empty CSV cells to `None` for numeric (float64) columns — pandas coerces `None` right back to `NaN` in a float column, and `validate_no_leakage()` checks `is None`, not `pd.isna()`. Fixed by forcing `astype(object)` before the substitution, so `None` can actually be stored.
3. Two test-only bugs (not implementation bugs): a `dict()` call on `ColumnTransformer.transformers` (a list of 3-tuples, not 2-tuples), and a `.index`-based overlap check that didn't account for `_apply_group_split()` deliberately resetting both frames' positional index. Both fixed in the tests themselves.

## 18. Limitations

- No real dataset, so no real performance numbers, no real model selection, no real feature-importance ranking, no real risk-distribution — all of Section 9–13's machinery is verified structurally, not empirically.
- No lag/rolling feature engineering (PROJECT_PLAN.md Section 11's "engineered temporal features" were aspirational, never built in any phase) — the model trains on raw per-interval measurements only.
- Permutation importance and Random Forest importance may disagree once real data exists (they measure different things); both are provided rather than picking one, but interpreting a disagreement is future work.
- Model selection (Section 12) genuinely cannot be finalized without real data — this document deliberately does not pretend otherwise.

---

*This document will be updated with a real model comparison table, real feature importance rankings, real risk distribution, and an actual final model selection once WSL2 + Mininet are confirmed working and `scripts/generate_dataset.py` has produced a real dataset.*
