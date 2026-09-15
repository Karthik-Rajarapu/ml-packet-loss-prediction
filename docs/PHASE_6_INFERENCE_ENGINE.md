# PHASE_6_INFERENCE_ENGINE.md
## Real-Time Packet-Loss Prediction / Inference Engine

**No production inference results are claimed because Phase 5 did not produce a real network dataset or a production-trained model.** Everything demonstrated in this phase uses an explicitly labeled TEST FIXTURE model trained on synthetic data. This document describes a working, thoroughly tested inference *engine*, not empirical prediction accuracy.

---

## 1. Purpose

Build the layer that sits between "a trained model" (Phase 4/5) and any future consumer of predictions (a CLI now, a Streamlit dashboard later, per Section 14) — turning a snapshot of current network metrics into a structured next-interval packet-loss prediction and derived risk level, without that consumer needing to know anything about preprocessing, pipelines, or feature ordering.

## 2. Prediction Definition (unchanged project-wide)

"Predict packet-loss percentage for the NEXT network interval using information available before/during the prediction cutoff." At cutoff `T`: inputs describe interval `T` (or earlier); the prediction is for interval `T+1`. This is identical to the definition used since Phase 1 (`network.targets.add_next_interval_target`) and Phase 4 (`network.schema.TARGET_COLUMN`) — Phase 6 does not redefine it, only consumes it at inference time.

## 3. Input Feature Contract

Inference input is a `dict[str, Any]` keyed by exactly `network.schema.FEATURE_COLUMNS` (15 fields: 5 configured-condition fields + 10 measured-current-interval fields — see `docs/PHASE_4_ML_MODELING.md` Section 3). `src/ml/inference.py::validate_feature_input()` enforces:
- every required feature present (no missing),
- no extra/unexpected fields (see Section 4 for why this is strict, not permissive),
- correct type per feature (`traffic_type` a string in `{"udp","tcp"}`, everything else numeric, not `bool`),
- no `NaN`, no infinite values,
- no negative values (physically impossible for every one of these measurements),
- `packet_loss_pct` within `[0, 100]` (a percentage).

**Feature ordering**: `build_feature_frame()` always constructs a single-row `pandas.DataFrame` with columns in canonical `FEATURE_COLUMNS` order. This is actually not strictly load-bearing for correctness today — Phase 4's `ColumnTransformer` (`ml.preprocessing.build_preprocessor`) selects columns **by name**, not position, so a `DataFrame` with the right column names in a different order would still transform correctly. `build_feature_frame()` builds canonical order anyway, for two reasons: (1) it keeps the contract self-evident and easy to eyeball/debug, and (2) it protects against a future refactor that switches to positional column selection, which would silently break without this discipline. Verified directly: `test_build_feature_frame_order_independent_of_input_dict_order` confirms two logically-identical inputs supplied with keys in different insertion order produce byte-identical output frames.

## 4. Feature Preprocessing (fully reused from Phase 4, unmodified)

`InferenceEngine.predict()` calls `model_handle.pipeline.predict(frame)` — the loaded `sklearn.Pipeline` **is** Phase 4's `preprocess` + `model` pipeline, saved whole via `ml.artifacts.save_model` (joblib). Inference therefore applies **the exact same fitted imputation/scaling/one-hot-encoding** the model was trained with — there is no separate, hand-written inference-time preprocessing step to accidentally drift out of sync with training. This directly satisfies Step 3's "do not duplicate feature transformations manually" instruction: there is nothing to duplicate, because nothing was rewritten.

## 5. Target Timing / Leakage Safety

The inference engine structurally cannot receive target-interval (`T+1`) information as a *feature*, because `FEATURE_COLUMNS` never contained any (verified: `test_target_column_never_in_feature_columns`, `test_feature_columns_contain_no_leakage_suspect_names`). The remaining risk is a **caller** mistakenly supplying future-looking information as an "extra" field (e.g. accidentally forwarding `target_next_packet_loss_pct` from a logging pipeline). `validate_feature_input()` handles this explicitly:
- Any extra field is rejected outright (Section 3).
- If an extra field's name contains `target`, `next_interval`, `next_packet_loss`, or `future` (case-insensitive), the error message specifically calls out that it looks like target/future information and refuses it, rather than a generic "unexpected feature" message — see `tests/test_inference_leakage.py::test_various_future_looking_field_names_are_flagged_specifically` (parametrized over 5 such names) and `test_unrelated_extra_field_gets_generic_message_not_leakage_message` (confirms the specific wording is reserved for genuinely suspicious names, not applied indiscriminately).

`current_rtt_ms`, `current_jitter_ms`, and `packet_loss_pct` are legitimately CURRENT-interval measurements (safe, per Phase 1's leakage documentation) — the only loss-related input feature is `packet_loss_pct`, verified by `test_current_packet_loss_pct_is_the_only_loss_related_feature`.

## 6. Leakage Prevention — Summary of Tests

`tests/test_inference_leakage.py` (a dedicated file, per Step 5's explicit instruction) covers: target column absent from the feature contract; target column supplied as input rejected with the specific leakage message; five different future-looking field-name variants each specifically flagged; an unrelated typo'd field gets the *generic* (not leakage-specific) message, so the specific wording stays meaningful; a full end-to-end check that a clean prediction succeeds while a contaminated one (same input + a smuggled `target_next_packet_loss_pct` field) is rejected outright, not silently ignored.

## 7. Model Loading (`src/ml/inference.py`)

`ModelHandle` bundles the fitted `pipeline`, its `ModelMetadata` (Phase 4, extended this phase — Section 15), the artifact's `source_path`, and an `is_test_fixture` flag.

- **`load_production_model(models_dir)`**: looks for exactly one `*.joblib` in `models_dir` (normally `models/`). Zero found → raises `ModelArtifactError` with the **exact** message specified in the Phase 6 brief: *"No production model is available. Train a model using a validated real network dataset first..."* More than one found → raises, asking the caller to load one explicitly (ambiguity is an error, not a silent "pick the first one"). Found, but its metadata says `is_test_fixture=True` → **still raises** (the exact wording differs, but it never returns a fixture model from this function) — this is the structural enforcement of "never silently replace a missing production model with a test model."
- **`load_test_fixture_model(joblib_path)`**: the mirror-image guard — refuses to load anything whose metadata does **not** say `is_test_fixture=True`, so a real trained model can't accidentally be used through the "this is just a fixture" code path either.
- **`_validate_feature_compatibility()`**: every load (production or fixture) checks the artifact's `feature_columns`/`target_column` against the CURRENT `network.schema` values — an artifact trained against a stale schema is rejected with a clear message rather than silently mispredicting with misaligned columns.

All of this is exercised in `tests/test_inference_model_loading.py` (13 tests) — every "production model" constructed there lives under `tmp_path`, never the real repo `models/` directory, and is only ever used to test loader *mechanics*.

## 8. Prediction API

```python
from ml.inference import InferenceEngine, load_production_model, predict_naive_baseline

# ML model path (raises ModelArtifactError if none exists -- see Section 7):
handle = load_production_model(Path("models"))
engine = InferenceEngine(handle)
result = engine.predict(current_network_metrics)   # -> PredictionResult

# Naive baseline path (Step 7 -- no trained artifact needed at all):
result = predict_naive_baseline(current_network_metrics)
```

`PredictionResult` (frozen dataclass): `predicted_packet_loss_pct`, `risk_level`, `model_name`, `model_kind` (`"ml_model"` or `"naive_baseline"` — the two paths are never conflated), `feature_names`, `predicted_at` (ISO timestamp), `is_test_fixture`, `model_metadata`. **Deliberately no confidence/uncertainty field** — Step 9: this model family provides no calibrated uncertainty, so none is invented (`test_prediction_result_has_no_confidence_field` asserts no field name even contains "confidence" or "uncertainty").

## 9. Risk Classification (fully reused from Phase 4, unmodified)

`ml.risk.classify_risk()` and `DEFAULT_RISK_THRESHOLDS` (`low_max=0.1`, `moderate_max=1.0`, documented in `PHASE_4_ML_MODELING.md` Section 13) are called directly, unchanged — no new thresholds were invented for Phase 6. Risk is always derived from the **predicted** value (`classify_risk([predicted])`), never from any actual/future value, which the engine never has access to in the first place (Section 5).

## 10. CLI Usage (`scripts/predict_packet_loss.py`)

```bash
# Naive baseline, built-in demo metrics, no trained model needed:
python3 scripts/predict_packet_loss.py --demo --baseline

# TEST FIXTURE demonstration (always labeled as such in the output):
python3 scripts/predict_packet_loss.py --demo --use-test-fixture

# Production model (fails clearly if models/ has no real trained model):
python3 scripts/predict_packet_loss.py --metrics-json my_metrics.json
python3 scripts/predict_packet_loss.py --metrics '{"current_rtt_ms": 12.3, ...}'
```
Exactly one of `--demo` / `--metrics-json` / `--metrics` is required (argparse mutually-exclusive group). `--models-dir` overrides the production model location (default `models/`). Verified running on this machine (Section 12).

## 11. Error Handling

Every failure path prints a clear message and returns a non-zero exit code — never a raw traceback for an expected condition (bad input, missing model), never a silent fallback:

| Situation | Behavior |
|---|---|
| Invalid/malformed metrics (any `validate_feature_input` violation) | Prints `"Invalid input metrics: <specific reason>"`, exit 1 |
| Unparseable `--metrics`/`--metrics-json` | Prints `"Failed to read input metrics: <reason>"`, exit 1 |
| No production model in `models/` | Prints the exact required message, exit 1 — **never** falls back to a test fixture |
| Only a fixture-marked artifact present where a production one was expected | Same as above — the fixture is not silently used |

## 12. Test-Fixture Usage

`src/ml/demo.py` builds a small `DecisionTreeRegressor` on **explicitly synthetic, hand-written rows** (not `tests/ml_fixtures.py` — `src/` never imports from `tests/`, so this is a small, deliberate, separately-maintained duplication) and saves it with `ModelMetadata(is_test_fixture=True, ...)`. `TEST_FIXTURE_MODEL_NAME = "TEST_FIXTURE_decision_tree"` makes this unmistakable even by filename. `--use-test-fixture` builds this fresh into a temporary directory (`tempfile.TemporaryDirectory`) for each CLI invocation — it is never written into `models/`.

Verified this phase (real command, real output, both reproduced below):
```
$ python3 scripts/predict_packet_loss.py --demo --use-test-fixture
======================================================================
TEST FIXTURE DEMONSTRATION -- NOT REAL NETWORK PERFORMANCE
======================================================================
*** TEST FIXTURE MODEL -- NOT TRAINED ON REAL NETWORK DATA ***
Predicted next-interval packet loss: 10.0000%
Risk: HIGH
Model: TEST_FIXTURE_decision_tree (ml_model)
Predicted at: 2026-09-15T07:20:25.415730+00:00

$ python3 scripts/predict_packet_loss.py --demo --baseline
======================================================================
NAIVE PERSISTENCE BASELINE (not the primary ML path)
======================================================================
Predicted next-interval packet loss: 8.5000%
Risk: HIGH
Model: naive_persistence (naive_baseline)
Predicted at: 2026-09-15T07:20:20.477388+00:00
```
**Neither of these is a project result.** Both use hand-written/synthetic input and a model trained on synthetic data. They demonstrate the engine works, not that the project predicts real packet loss accurately.

## 13. Production-Model Requirements

A production model requires, at minimum: (1) a real Mininet-generated dataset (Phase 5, currently blocked), (2) `scripts/train_models.py` run against it (Phase 4, code-complete, never executed on real data), which saves `models/<best_name>.joblib` + `models/<best_name>_metadata.json` with `is_test_fixture=False`. Verified this phase: running `scripts/predict_packet_loss.py --demo` (no `--baseline`, no `--use-test-fixture`) against the real, currently-empty `models/` directory correctly fails:
```
$ python3 scripts/predict_packet_loss.py --demo
No production model is available. Train a model using a validated real network
dataset first (see docs/PHASE_5_REAL_DATA_VALIDATION.md and run
scripts/train_models.py against real Mininet-generated data).
```
Exit code 1. This is the honest, current state of the project — **no production model exists**.

## 14. Future Live Integration (design only — not implemented this phase)

```
Network Metrics Collector  (future: reads from a live Mininet run or a monitoring feed)
        |
        v
Feature Builder             (validate_feature_input + build_feature_frame -- already built, this phase)
        |
        v
Inference Engine            (InferenceEngine.predict -- already built, this phase)
        |
        v
Prediction Result           (PredictionResult -- already built, this phase)
        |
        v
Streamlit Dashboard         (NOT built -- explicitly out of scope this phase)
```
`InferenceEngine.predict()` takes a plain `dict` and returns a plain, typed `PredictionResult` — no Mininet-specific or CLI-specific code inside it, so a future metrics collector (or a future dashboard) can call it directly without needing to know anything about pipelines, preprocessing, or joblib. This is a design property already achieved by today's code, not a promise about unbuilt code.

## 15. Model Version / Metadata

`ModelMetadata` (Phase 4) was extended this phase with two genuinely-populatable fields: `is_test_fixture: bool` (set explicitly at save time, never left to inference to guess) and `training_timestamp: str | None` (real `datetime.now(timezone.utc).isoformat()` at the moment `scripts/train_models.py` saves an artifact — not fabricated). **Not added**: `model_version`, `training_dataset_id`, `feature_schema_version` — this project has no model-versioning or dataset-identity scheme yet, and adding fields that would always read `None` risked implying tracking that doesn't exist. This is reported honestly here rather than invented, per Step 15's explicit instruction.

`test_saved_metadata_contains_no_unexpected_keys` (Phase 4, updated this phase) confirms the metadata JSON contains exactly the seven documented fields and nothing else — no secrets, no environment variables, no untracked extras.

## 16. Limitations

- **No production model exists.** Every prediction demonstrated this phase used a test-fixture model on synthetic input.
- **No real-time integration exists.** The engine is called synchronously with a plain dict today; nothing reads live Mininet output yet (Section 14 is a design, not an implementation).
- **No uncertainty/confidence estimate** is provided, by design (Step 9) — the current model family (scikit-learn point-estimate regressors) doesn't produce calibrated uncertainty without additional work explicitly deferred as optional.
- **`traffic_type` is restricted to `{"udp", "tcp"}`** at the validation layer — matches `ExperimentConfig`'s own constraint (Phase 1), but would need updating if a future phase adds a new traffic type.
- Feature-name-based leakage detection (Section 5) is a heuristic (substring match on suspicious words) — it catches the field names a careless caller would plausibly use, not a formal proof that no leakage path exists. The structural guarantee (target never in `FEATURE_COLUMNS`, extra fields always rejected) is what actually does the safety work; the specific wording is a usability nicety on top of it.

---

*This document will be updated once a real production model exists (Phase 5 unblocked → Phase 4 run for real) — at that point Section 13's CLI output, and only that output, becomes a real project result rather than a fixture demonstration.*
