# RESULTS.md

This document has two sections, deliberately kept separate and never blended: **Section A** lists only what has actually been run and verified on this machine. **Section B** is the empirical network-performance section every ML project needs — and it is honestly empty, with the reason stated.

---

## SECTION A — VERIFIED ENGINEERING RESULTS

Everything below was actually executed on this development machine; none of it is estimated, projected, or assumed.

### A.1 Automated Test Suite
```
280 / 280 tests passed
2 tests skipped (live-Mininet integration tests -- environment unavailable, see A.3)
0 tests failed
```
Command: `py -3 -m pytest tests/ -v`. Covers: network experiment/config/validation logic, dataset generation and sweep configuration, leakage-safe target construction, group-aware splitting, preprocessing leakage prevention, all 4 ML models + naive baseline, evaluation metrics, feature importance (both methods), risk classification, model artifact save/load, the full inference engine (validation, feature contract, model loading, production/fixture separation), the Streamlit dashboard's support modules and app-level integration, a full end-to-end fixture pipeline (Section A.5), a system health check, and inference latency (Section A.6).

### A.2 System Health Check
`scripts/system_check.py` — real output, this machine:
- **All REQUIRED software checks: AVAILABLE** — Python 3.14.0, all 7 third-party packages, all 27 project module imports, `app.py` importability, all 7 expected project directories, the inference engine (via test fixture).
- Production model check: **NOT AVAILABLE** (correctly categorized INFORMATIONAL — does not fail the check).
- Network tools (`mn`, `iperf3`, `tc`, `ovs-vsctl`): **NOT AVAILABLE**; `ping`: AVAILABLE (correctly categorized REQUIRED FOR LIVE EXPERIMENTS — does not fail the check).
- **Overall: SOFTWARE HEALTH CHECK: PASS**, exit code 0.

### A.3 Live-Mininet Integration Tests
`tests/test_live_mininet.py` — 2 tests, both **SKIPPED** (not passed, not failed) via `pytest.mark.skipif`, because `mn`/`iperf3`/`tc`/`ovs-vsctl` are not on PATH on this machine. These tests are real (unmocked) integration tests that would build an actual topology and run real traffic the moment the environment becomes available — see `docs/ENVIRONMENT_SETUP.md` for the exact blocker.

### A.4 Feature Contract / Leakage / Temporal-Split Audits
All three re-verified in Phase 8 with **zero issues found**:
- Single authoritative feature list (`network/schema.py`), confirmed by repository-wide audit — no module independently redefines it.
- Zero lag/rolling/windowed feature engineering anywhere in the codebase (`grep -rn "rolling|\.shift\(|window" src/` → no matches).
- Target column structurally absent from the feature list; inference-time validator specifically rejects any input field that looks like leaked target/future information.
- Group-aware splits verified to produce zero `experiment_id` overlap between train and test, including against a deliberately-corrupted split constructed specifically to prove the overlap guard actually works.

### A.5 End-to-End Fixture Integration
`tests/test_end_to_end_integration.py` — a single test chains, using real project code throughout: synthetic CSV → load + validate → data-quality report → both split strategies → train all 5 models (baseline + 4 ML) → save a model artifact (explicitly marked `is_test_fixture=True`) → load it back through the test-fixture-only loader → run inference → cross-check risk classification independently → verify the result flows cleanly into the dashboard's prediction-history and feature-input contracts. **Passed.**

    TEST FIXTURE — NOT REAL NETWORK DATA. Structural/plumbing proof only.

### A.6 Inference Latency Benchmark

> **TEST FIXTURE MODEL — DEVELOPMENT MACHINE — SOFTWARE INFERENCE ONLY**

```
Sequential predictions: n=300
  mean:   12.995 ms
  median: 12.564 ms
  p95:    15.687 ms
  min:    10.721 ms
  max:    18.889 ms

Concurrent throughput (concurrency=4, n=300): 68.7 predictions/sec
```
Measured via `scripts/benchmark_inference.py` on Intel Core i5-7300U (2 physical/4 logical cores), 8GB RAM, Windows 10 Pro, Python 3.14.0, using the tiny synthetic-data `DecisionTreeRegressor` test fixture (`ml.demo.build_test_fixture_model`). **This is Python-process call-latency for one small model on one laptop.** It is explicitly **not**: network latency, packet-loss prediction accuracy, production performance, or a claim about real concurrent network-monitoring load.

### A.7 Dashboard Integration
`app.py` imports cleanly; `streamlit run app.py --server.headless true` starts a real server (`Uvicorn server started`) that returns real HTTP 200; `app.main()` executed directly in Streamlit's documented "bare mode" produced **zero Python tracebacks** across the full render/interaction code path (status check, all 15 input widgets, history, model info, feature importance, actual-vs-predicted, metrics sections). PRODUCTION/DEMONSTRATION mode separation verified with no silent fallback in either direction.

---

## SECTION B — EMPIRICAL NETWORK RESULTS

```
REAL MININET-GENERATED DATASET:        NOT AVAILABLE
PRODUCTION-TRAINED MODEL:              NOT AVAILABLE
REAL MODEL PERFORMANCE (MAE/RMSE/R²):  NOT AVAILABLE
REAL FEATURE IMPORTANCE:               NOT AVAILABLE
REAL ACTUAL-VS-PREDICTED CHARTS:       NOT AVAILABLE
REAL PACKET-LOSS DISTRIBUTION:         NOT AVAILABLE
REAL RISK-LEVEL DISTRIBUTION:          NOT AVAILABLE
REAL INFERENCE-ON-LIVE-TRAFFIC RESULT: NOT AVAILABLE
```

**Reason**: the WSL2/Mininet environment required to run real network experiments has not been available on the development machine throughout this project (`Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` Windows features remain disabled; `mn`, `iperf3`, `tc`, `ovs-vsctl` are not installed — see `docs/ENVIRONMENT_SETUP.md` and `docs/PHASE_5_REAL_DATA_VALIDATION.md` for the full troubleshooting history). Without a real Mininet experiment, there is no real `data/raw/*.csv`, and without real data there is nothing for `scripts/train_models.py` to train a production model on.

**No placeholder values are given for MAE, RMSE, or R² in this document, and none exist anywhere in this repository.** No chart under `reports/modeling/` was generated from real data — the directory contains only a `.gitkeep` placeholder. This is a direct, intentional consequence of the strict data-integrity rule followed throughout this project: an unavailable result is reported as unavailable, never approximated, estimated, or illustrated with fixture numbers relabeled as real ones.
