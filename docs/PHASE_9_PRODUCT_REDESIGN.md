# PHASE_9_PRODUCT_REDESIGN.md
## Dashboard Redesign: From Developer Console to Product Workflow

Supersedes the single-page dashboard described in `docs/PHASE_7_STREAMLIT_DASHBOARD.md` (kept for history, now marked superseded). **This redesign changes the UI and adds a dataset-driven training workflow; it does not change, weaken, duplicate, or bypass any existing ML, leakage-protection, splitting, or inference logic.** Every page calls directly into the same `src/network/`, `src/ml/{split,train,evaluate,artifacts,inference,risk}.py` modules used since Phase 4/6, unmodified.

---

## 1. What Changed

**Before**: a single scrolling page — sidebar mode toggle, then a 15-field manual-entry form as the *first* thing a user saw, framed around Mininet/feature-contract/inference-engine language.

**After**: a 6-page product workflow — **Home → Upload Data → Prepare Dataset → Train Model → Predict → History** — where a user can upload their *own* CSV, have its columns matched to the project's feature contract, build a leakage-safe target, train and compare the existing 5 models, explicitly promote one to "production," and predict against it. The original manual-entry form still exists, unchanged, relocated to Predict → "Manual Prediction."

## 2. New Components

| Module | Role |
|---|---|
| `src/ml/dataset_upload.py` | Safe CSV parsing (type/size/shape checks) + generic pre-mapping health stats |
| `src/ml/column_mapping.py` | Alias-based detection of the 15 canonical features in an arbitrary CSV, with an explicit confidence level (high/low/none) — never silently accepts an ambiguous match |
| `src/ml/dataset_adapter.py` | Turns a mapped upload into the canonical schema and constructs the target by calling `network.targets.add_next_interval_target` and `network.validation`'s validators **unmodified** |
| `src/ml/training_orchestration.py` | Sequences the **existing** `ml.split` / `ml.train.train_and_evaluate` / `ml.artifacts` functions; adds an explicit (never automatic) "save as production" step that archives, rather than overwrites, any prior artifact |
| `src/dashboard/session.py` | One `AppSession` dataclass in `st.session_state`, replacing ad hoc keys |
| `src/dashboard/pages/*.py` | Six thin page renderers, each calling only into the above and the pre-existing `src/dashboard/{feature_inputs,model_status,history,reports}.py` |

`src/dashboard/history.py` gained one field (`actual_packet_loss_pct`, optional, default `None`) and one method (`record_actual()`) — see Section 5.

## 3. What Was Reused, Unchanged

`network/schema.py` (the one feature contract), `network/targets.py`, `network/validation.py`, `ml/split.py`, `ml/models.py`, `ml/train.py`, `ml/evaluate.py`, `ml/artifacts.py`, `ml/inference.py`, `ml/risk.py`, `ml/demo.py`, `src/dashboard/{feature_inputs,model_status,reports}.py`. None of these files were modified. The manual-entry form in Predict → "Manual Prediction" is the exact same widget code that was previously the entire app.

## 4. Production vs. Demonstration — Unchanged Guarantee, New Entry Point

The structural separation from Phase 6/7 is untouched: `ml.inference.load_production_model()` never returns a fixture-marked artifact, and `ml.demo.build_test_fixture_model()` is never used to produce anything saved into `models/`. What's new is that a **real** (non-fixture) production model can now be created *from the app itself* — a user uploads their own data, trains, and clicks "Save as Production Model." That save path (`training_orchestration.save_as_production_model`) always writes `is_test_fixture=False` and never imports `ml.demo` (verified structurally: `tests/test_training_orchestration.py::test_save_as_production_model_never_imports_ml_demo`). Any prior production artifact is archived to `models/archive/<timestamp>_...`, never deleted or silently overwritten.

**On this development machine, `models/` still contains only `.gitkeep`.** No dataset has been uploaded and trained through this workflow for real; every verification this phase used an explicitly labeled TEST FIXTURE CSV, and no artifact from that verification was saved to the real `models/` directory (checked directly after every test run).

## 5. Actual-vs-Predicted (History Page)

Phase 7 explicitly omitted "actual" outcome tracking because nothing in a manual-input dashboard could observe a true future value. This phase adds an honest mechanism for it: a user can, after the fact, type in the real value they later observed for a specific past prediction (`PredictionHistory.record_actual(index, value)`). This is the **only** way `actual_packet_loss_pct` is ever set — it defaults to `None` and nothing in the software infers or estimates it. Once at least one actual value is recorded, an actual-vs-predicted scatter plot appears; until then, the History page says so plainly and shows nothing.

## 6. Testing

54 new tests this phase, all passing:

| File | Covers |
|---|---|
| `test_dataset_upload.py` (13) | Untrusted-CSV handling: type/size/empty/malformed rejection, generic health stats |
| `test_column_mapping.py` (12) | Alias detection confidence levels, constant-value eligibility, mapping application |
| `test_dataset_adapter.py` (8) | Canonical-shape construction, **leakage-safety of the adapter itself** (a dedicated test tries to make a target cross an experiment boundary through this new glue code) |
| `test_training_orchestration.py` (6) | Reuses `train_and_evaluate` correctly, production-save always `is_test_fixture=False`, archiving instead of overwriting, structural non-dependency on `ml.demo` |
| `test_dashboard_session.py` (2) | No shared-mutable-default bug |
| `test_dashboard_app.py` (6, rewritten) | The new nav-shell: imports cleanly, cached helpers work, `main()` stays behind its `__main__` guard |
| `test_dashboard_pages_predict.py` (6) | The safety-critical routing logic that replaced the old `app.run_prediction` — production/demo/baseline separation, invalid-input handling |
| `test_dashboard_pages_render.py` (11) | Every page renders without a Python exception across multiple real session states (empty, populated) |
| `test_dashboard_workflow_integration.py` (1) | The **entire** new workflow chained together — upload (realistic aliased column names) → mapping → prepare → train → predict → record an actual outcome → history — using one clearly labeled TEST FIXTURE CSV, verifying no artifact leaks into the real `models/` directory |

Full suite: **280 passed, 2 skipped** (the pre-existing live-Mininet tests, unrelated to this phase). Nine tests in the old `test_dashboard_app.py` were intentionally rewritten (not deleted-and-forgotten) because they asserted against `app.py` internals (`run_prediction`, `DEMO_CURRENT_METRICS`, etc.) that intentionally moved into page modules — an obsolescence directly caused by the requested redesign, not an unrelated deletion.

## 7. Live Verification

`streamlit run app.py --server.headless true` started a real server (HTTP 200 confirmed) this phase, and `python3 -c "import app; app.main()"` — Streamlit's documented bare-execution mode — ran the full Home-page render path with zero Python tracebacks. Every individual page's render function was additionally exercised directly (not just the default Home path) via `tests/test_dashboard_pages_render.py`, which is a stronger check than the bare-mode `main()` call alone since it reaches every page, not just whichever one loads by default.

**Not verified**: real click-through interaction in a JS-capable browser (unchanged limitation from Phase 7/8).

## 8. Limitations Introduced or Carried Forward

- Uploading a dataset whose columns don't resemble any known alias for a required feature still blocks training until the user manually maps it or supplies a constant (for link-configuration fields only) — this is by design (never silently guess), not a bug.
- A dataset with only one experiment/group cannot be group-split and will not train (same underlying constraint as the CLI pipeline since Phase 4).
- The "record actual outcome" feature is manual and per-row; there is still no live feed of ground truth.
- File upload size is capped at 50MB / 500,000 rows as a basic safety limit for untrusted input on a course-project-scale machine.
