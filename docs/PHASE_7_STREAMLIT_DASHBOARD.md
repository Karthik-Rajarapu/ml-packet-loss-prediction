# PHASE_7_STREAMLIT_DASHBOARD.md
## Streamlit Prediction Dashboard

> **SUPERSEDED by `docs/PHASE_9_PRODUCT_REDESIGN.md`.** The single-page developer-console UI described below was redesigned in Phase 9 into a multi-page product workflow (Home → Upload Data → Prepare Dataset → Train Model → Predict → History). The manual-entry form and every piece of underlying inference/risk logic described here are unchanged and still exist (relocated to Predict → "Manual Prediction") — this document is kept for history, not as the current UI description.

**Production predictions are unavailable until Phase 5 generates a real network dataset and a production model is trained.** Everything demonstrated in this phase uses an explicitly labeled TEST FIXTURE model trained on synthetic data, or the naive persistence baseline. This document describes a working, tested dashboard *shell*, not empirical prediction results.

---

## 1. Dashboard Architecture

```
Network Metrics (dict)
        |
        v
Feature Validation      -- ml.inference.validate_feature_input (Phase 6, unchanged)
        |
        v
Feature Frame Building  -- ml.inference.build_feature_frame (Phase 6, unchanged)
        |
        v
Inference Engine         -- ml.inference.InferenceEngine.predict / predict_naive_baseline (Phase 6, unchanged)
        |
        v
Risk Classification       -- ml.risk.classify_risk (Phase 4, unchanged, called from inside InferenceEngine)
        |
        v
PredictionResult            -- typed dataclass (Phase 6, unchanged)
        |
        v
app.py (Streamlit)           -- PRESENTATION ONLY: renders PredictionResult, never computes one itself
```

`app.py` and the new `src/dashboard/` package (`feature_inputs.py`, `model_status.py`, `history.py`, `reports.py`) contain **zero** model-loading, preprocessing, feature-engineering, leakage-checking, or risk-classification logic. Every one of those concerns is a direct, unmodified call into Phase 4/6 code — verified by inspection (Step 1) before any dashboard code was written, and enforced structurally: `src/dashboard/*.py` only imports from `ml.inference`, `ml.risk` (indirectly), and `network.schema`, never reimplementing any of it.

## 2. UI Sections

Matches the brief's suggested layout: **System Status** → **Current Network Conditions** (input form) → **Next-Interval Prediction** → **Prediction History** → **Model Information** → **Feature Importance** → **Actual vs Predicted** → **Model Evaluation Metrics**. Two-column responsive layout for the 15 input widgets; a sidebar holds the PRODUCTION/DEMONSTRATION mode switch and the baseline toggle, kept separate from the main content flow per "keep the UI professional... minimal clutter."

## 3. Production Mode

On startup, `dashboard.model_status.check_production_model(MODELS_DIR)` (`MODELS_DIR = <repo>/models`) tries `ml.inference.load_production_model()` and returns a `ModelStatus(available, handle, message)` — **never raises**, so `app.py` always has something to branch on. If unavailable, the exact required message is shown:

> **Production model unavailable.**
> Phase 5 real-network dataset generation has not yet been completed.
> The dashboard interface is available in demonstration mode. Production predictions require a validated model trained on real network experiments.

Confirmed this phase (current repo state — `models/` has only `.gitkeep`): PRODUCTION mode selected in the sidebar shows exactly this message, and the prediction form, if submitted in PRODUCTION mode with no model available, returns no result rather than silently using anything else (`test_run_prediction_production_mode_returns_none_when_unavailable`).

## 4. Demonstration Mode

Selecting **DEMONSTRATION** in the sidebar shows a persistent banner: **"DEMONSTRATION MODE — TEST FIXTURE — NOT REAL NETWORK PERFORMANCE."** Every prediction made in this mode is served by `ml.demo.build_test_fixture_model()` (Phase 6, unmodified) — a `DecisionTreeRegressor` trained on small, hand-written synthetic rows, cached per Streamlit process via `st.cache_resource` (so it's built once, not re-trained on every click) and written only to a throwaway `tempfile.mkdtemp()` directory, never `models/`. Every `PredictionResult` from this path carries `is_test_fixture=True`, which `render_prediction_result()` uses to show the same warning banner again directly above the prediction output — the label travels with the data, not just with the mode selector.

Because `mode` and `model_status` are two independent pieces of state, DEMONSTRATION mode is verified to use the test fixture **even when a production model happens to be available** (`test_run_prediction_demonstration_mode_always_uses_test_fixture`) — modes never bleed into each other in either direction.

## 5. Feature Inputs

`dashboard.feature_inputs.build_feature_input_specs()` builds one `FeatureInputSpec` per `network.schema.FEATURE_COLUMNS` entry, **in that exact order**, pulling `unit` and `description` directly from `network.schema.SCHEMA` (never inventing a feature name — verified: `test_feature_mapping_matches_schema`). Widget bounds (slider min/max/step) are a presentation convenience layered on top, explicitly documented as non-authoritative: `ml.inference.validate_feature_input` is still the only thing that decides what's actually accepted (e.g. `bandwidth_utilization_pct` isn't capped at 100 in the widget, since the engine doesn't cap it either — utilization can legitimately exceed 100% under above-bottleneck offered load). Default values come from `ml.demo.DEMO_CURRENT_METRICS` — the same explicitly-synthetic example Phase 6 already used for its CLI demo, reused here rather than inventing a second set of example numbers.

## 6. Prediction Workflow

Inputs are collected inside a single `st.form` (so widget interactions don't trigger a rerun/prediction on every keystroke) with a **"PREDICT NEXT INTERVAL"** submit button. On submit, `app.run_prediction(mode, model_status, use_baseline, metrics)` is the **one function** that calls into the inference layer:
1. `use_baseline=True` → `ml.inference.predict_naive_baseline()` (no model artifact needed at all).
2. `mode == "DEMONSTRATION"` → the cached test-fixture `InferenceEngine`.
3. `mode == "PRODUCTION"` → the real `InferenceEngine` from `model_status.handle`, or a clear error if unavailable.

No ML logic lives in the Streamlit button-click handler itself — `run_prediction()` is a plain, independently-testable function that `main()` calls (Step 6's explicit instruction). All three paths, plus the invalid-input and no-model-available cases, are covered by `tests/test_dashboard_app.py`.

## 7. Risk Display

`risk_level` comes straight from the `PredictionResult` (already classified by `ml.risk.classify_risk` inside `InferenceEngine`/`predict_naive_baseline` — Phase 4's thresholds, unmodified, no new ones invented). Displayed with a colored indicator (🟢 LOW / 🟡 MODERATE / 🔴 HIGH) directly under the prediction. **CURRENT OBSERVATION** (the input's own `packet_loss_pct`) and **PREDICTED NEXT INTERVAL** are shown side-by-side in two visually distinct columns with those exact labels — never just "Current Packet Loss" for the prediction, per Step 7's explicit instruction.

## 8. Prediction History

`dashboard.history.PredictionHistory`, one instance per browser session via `st.session_state`, **never written to disk**. Each entry: timestamp, model name/kind, `is_test_fixture` flag, predicted loss, risk. **Deliberately has no "actual packet loss" column** — a next-interval outcome is not known at prediction time in this manual-input dashboard (there is no live feed replaying ground truth yet), so displaying one would mean fabricating a network measurement (Strict Rule #1/#4). This is a considered deviation from the brief's suggested `Time | Actual | Predicted | Risk` layout, documented here rather than silently worked around: fabricating a plausible-looking "actual" value was rejected as a design option. `test_dataframe_never_contains_an_actual_loss_column` enforces this doesn't regress.

## 9. Model Information

Feature count and full feature list (from `network.schema.FEATURE_COLUMNS`, in an expander); in PRODUCTION mode with a model available, also model type, training timestamp, and training configuration from `ModelMetadata` (Phase 6's extended fields — Section 15 of `PHASE_6_INFERENCE_ENGINE.md`). Otherwise, the exact required message: *"Model evaluation metrics unavailable because no production model has been trained on real network data."*

## 10. Feature Importance

Reads (never recomputes) `reports/modeling/*_feature_importance.csv` and `*_permutation_importance.csv` via `dashboard.reports.find_feature_importance()` / `find_permutation_importance()` — the exact files `scripts/train_models.py` (Phase 4) would produce from a real training run. `reports/modeling/` currently contains only `.gitkeep`, so this section currently always shows: *"Feature importance unavailable: no production-trained model with saved feature importance exists yet..."*

## 11. Actual-vs-Predicted

Reads `reports/modeling/*_actual_vs_predicted.png` / `*_residuals.png` if present — real Phase 4 output, never regenerated by the dashboard. **No test-fixture version was built for this section**, even though the brief allows one if clearly labeled: constructing a meaningful actual-vs-predicted chart requires paired (true future value, predicted value) data, and this dashboard has no way to observe a true future outcome for any manually-entered input without inventing one — which would cross into fabricating a network measurement. This gap is a deliberate, documented design choice, not an oversight. Currently shows: *"Actual-vs-predicted visualization will be available after real network experiments and production-model training."*

## 12. Error Handling

`run_prediction()` catches exactly three cases and turns each into a concise `st.error()` message instead of a raw traceback: `FeatureValidationError` (bad input — message names the specific problem feature, from Phase 6), `ModelArtifactError` (no/incompatible model), and a last-resort broad `Exception` handler that shows a generic message plus a collapsed **"Technical details (for developers)"** expander containing the actual traceback — satisfying both "don't expose raw stack traces to normal users" and "keep useful debugging information available for developers" simultaneously.

## 13. How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
Verified this phase on this machine:
- `python3 -c "import app"` — imports cleanly, no Streamlit runtime errors, `main()` does **not** auto-execute (guarded by `if __name__ == "__main__"`).
- `streamlit run app.py --server.headless true --server.port 8765` — server started (`Uvicorn server started on :::8765`), `GET http://localhost:8765` returned **HTTP 200** with a valid Streamlit HTML shell.
- **Additionally** (since a plain HTTP GET only proves the static asset server works — Streamlit doesn't execute the script body until a browser opens a websocket, which wasn't available to verify against): `python3 -c "import app; app.main()"` was run directly. This exercises `main()`'s full logic — system-status check, all 15 input-widget calls, form handling, history/model-info/feature-importance/actual-vs-predicted/metrics rendering — in Streamlit's documented "bare mode" (no `ScriptRunContext`). Every `st.*` call safely no-ops with a logged warning in that mode (this is standard, documented Streamlit behavior, not an error condition); the full run produced **zero Python tracebacks** across 386 lines of output. This is the strongest available verification without a JS-capable browser — actual visual rendering, click interaction, and multi-rerun session-state persistence were **not** verified and are noted as a limitation below.

## 14. Current Limitations

- **No production model exists** — every prediction shown by this dashboard is either the naive baseline or an explicitly labeled TEST FIXTURE. This is the headline limitation, not a footnote.
- **No real browser/JS verification.** The startup smoke test (Section 13) proves the app imports, starts, serves, and runs `main()` without a Python exception in bare mode — it does not prove the actual rendered page looks right, that clicking "PREDICT NEXT INTERVAL" in a real browser works end-to-end, or that `st.session_state`-based history correctly persists across multiple real reruns in one browser session (bare-mode execution used for verification explicitly does not exercise Streamlit's real rerun/session machinery).
- **No live network monitoring integration** (Step 12/14) — the dashboard takes manual form input or the test fixture only; a future metrics collector would call `run_prediction()`'s same `metrics: dict` contract, but nothing reads live Mininet output yet. This is a design property of the current code, not a promise about unbuilt code.
- **No actual-vs-predicted fixture visualization** — deliberately omitted (Section 11) rather than built with an invented "actual" value.
- **Prediction history has no "actual" column** — deliberately omitted (Section 8), a considered deviation from the brief's suggested layout.

---

*This document will be updated with real screenshots/interaction notes once a JS-capable browser is available for verification, and with real production-mode behavior once Phase 5 produces real data and Phase 4 is run against it.*
