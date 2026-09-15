# Machine Learning-Based Prediction of Packet Loss in Computer Networks
## Final Project Report

---

## 1. Title

**Machine Learning-Based Prediction of Packet Loss in Computer Networks**

## 2. Abstract

This project investigates whether machine learning can predict next-interval packet loss in a computer network from currently observable network conditions. A Mininet-based dumbbell-topology experiment framework, a leakage-safe dataset generation pipeline, a regression-based ML modeling pipeline (naive baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting), a production-ready inference engine, and a Streamlit demonstration dashboard were designed, implemented, and thoroughly tested (223 automated tests passing). However, the WSL2/Mininet environment required to execute real network experiments was not available on the development machine throughout the project, so **no real network dataset, production-trained model, or empirical prediction-accuracy result exists**. This report documents a complete, tested, ready-to-run software system and is explicit, throughout, about the boundary between what was engineered and tested versus what remains empirically unvalidated.

## 3. Introduction

Packet loss degrades throughput, inflates latency (via retransmission), and directly harms real-time applications such as VoIP and video conferencing. Most operational responses to packet loss are reactive — detected after the fact via monitoring or user complaints. This project asks whether a proactive, ML-based approach is viable: given current network telemetry, can a model predict the packet loss that will occur in the *next* measurement interval, before it happens?

## 4. Problem Statement

**Research question**: Given current/previous network conditions (RTT, jitter, throughput, utilization, queue occupancy, retransmissions, current packet loss, and configured link parameters), can a machine learning regression model predict the packet-loss percentage in the next measurement interval, and does it outperform a naive "loss doesn't change" baseline?

## 5. Objectives

1. Generate labeled network-condition data from **real, controlled Mininet experiments** — not a downloaded or synthetic dataset — so packet loss is a genuine emergent property of queueing/congestion.
2. Build a leakage-safe regression pipeline predicting next-interval packet loss.
3. Compare multiple models against a meaningful temporal baseline using MAE/RMSE/R².
4. Rigorously prevent data leakage given the temporal nature of the task.
5. Provide a derived LOW/MODERATE/HIGH risk classification layer.
6. Build a reusable inference engine and a demonstration dashboard.
7. Be transparent, throughout, about which results are engineering-verified versus empirically validated on real data.

## 6. Background

Packet loss in a network is predominantly driven by queueing behavior: when a link's offered load exceeds its service capacity for long enough, a finite buffer fills and tail-drops packets. This is a nonlinear function of bandwidth, delay, queue depth, and traffic load — well suited to being learned from data rather than derived from a closed-form queueing-theory expression. Regression (rather than classification) is the natural formulation since packet loss is a continuous percentage; an optional risk-tier classification is layered on top for operational interpretability, without replacing the underlying regression target.

## 7. Proposed System

An end-to-end pipeline: Mininet network experiments → data collection → dataset validation → feature access (raw per-interval measurements) → temporal/group-aware train/test split → model training and evaluation against a naive baseline → a saved model artifact → an inference engine that enforces the exact training-time feature contract → a derived risk layer → a Streamlit dashboard for demonstration. See `docs/ARCHITECTURE.md` for the full diagram and per-layer interface description.

## 8. System Architecture

```
Mininet Network (Dumbbell Topology) -> Data Collector -> Raw Dataset -> Validation
-> Feature Engineering -> Temporal/Group Split -> ML Models (Naive, LR, DT, RF, GB)
-> Model Artifact -> Inference Engine -> Risk Classification -> Streamlit Dashboard
```
Full detail, per-layer responsibilities, and module references: `docs/ARCHITECTURE.md`.

## 9. Network Experiment Design

A small dumbbell topology (left hosts → shaped bottleneck → right hosts, capped at 4 hosts for the development machine's resource budget) built with Mininet's real Linux networking stack — chosen over a discrete-event simulator (ns-3) or raw `tc`/netem specifically because it produces **genuinely emergent** packet loss from real kernel queueing, not an injected label, while remaining tractable for a single laptop. Offered load is swept both below and above each run's configured bandwidth (as a *factor* of that bandwidth, e.g. 0.7×/1.3×) so the resulting dataset would span both healthy and congested conditions. Full comparison of alternatives and justification: `docs/PHASE_1_NETWORK_EXPERIMENT.md` Sections 1–3.

## 10. Dataset Generation

`scripts/generate_dataset.py` runs a configurable sweep of `ExperimentConfig`s (via `src/network/sweep.py`), each producing one leakage-safe CSV via `src/network/experiment.py`. A documented pilot sweep (`configs/pilot_sweep.json`, 12 experiments, ~4 minutes) was authored and verified via dry-run (no network touched) in Phase 5, ready to run the moment the environment allows it. **No real dataset was ever generated** — `data/raw/` contains only a `.gitkeep` placeholder.

## 11. Feature Engineering

15 features (`network/schema.py::FEATURE_COLUMNS`): 5 configured-condition columns (bandwidth, delay, queue depth, traffic type, active connections) and 10 measured current-interval columns (RTT, jitter, throughput, utilization, packet rate, queue length, retransmissions, packets sent/received, current packet loss). No lag, rolling, or windowed features are computed — a deliberate scope decision, confirmed absent by a project-wide code audit in Phase 8, since no real dataset has ever existed to justify or validate more elaborate engineered features against.

## 12. Target Construction

`target_next_packet_loss_pct` = `packet_loss_pct` measured exactly one interval (configurable) after the row's own interval, within the same experiment. Constructed once, centrally, by `network/targets.py::add_next_interval_target()`, grouping by experiment and ordering by interval index before shifting, so a target can never be pulled from the wrong row or a different experiment.

## 13. Data Leakage Prevention

Four independent, layered protections (detailed in `docs/METHODOLOGY.md` Section 10): (1) structural absence of the target column from the feature list; (2) leakage-safe target construction with strict group/order semantics; (3) an independent post-hoc validator (`validate_no_leakage()`) that re-derives every target from its true future value; (4) an inference-time input validator that specifically rejects any field name resembling leaked target/future information. All four were re-audited project-wide in Phase 8 with zero issues found.

## 14. ML Models

Naive persistence baseline, Linear Regression (with feature scaling), Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor — all fixed at `random_state=42` for reproducibility. XGBoost was deliberately not added, since the project brief specified adding it only if results justified it, and no real results have ever existed to justify it against.

## 15. Evaluation Methodology

Group-aware train/test splitting (chronological or random-group, never row-level) to prevent temporal autocorrelation from leaking across the split boundary. MAE, RMSE, and R² as primary metrics, explicitly never selecting a "winning" model by R² alone; residual-distribution statistics computed in addition, since packet loss is often near-zero and aggregate metrics alone can obscure how errors are actually distributed. Full detail: `docs/METHODOLOGY.md` Sections 13, 16.

## 16. Inference Engine

`src/ml/inference.py` provides a single, reusable `InferenceEngine.predict(metrics: dict) -> PredictionResult` API: strict feature-contract validation, canonical-order feature-frame construction, prediction via the exact saved training pipeline (so preprocessing cannot drift between training and inference), and derived risk classification. A structural guarantee — enforced by the loader, not just convention — prevents a test-fixture model from ever being used as, or mistaken for, a production one, and vice versa.

## 17. Risk Classification

LOW (≤0.1%), MODERATE (0.1–1.0%), HIGH (>1.0%) predicted packet loss, following widely-used real-time-traffic quality guidance. Fixed a priori; never fit or tuned against any dataset (which would risk quietly leaking a specific dataset's distribution into a project design decision).

## 18. Streamlit Dashboard

`app.py` + `src/dashboard/`: a presentation-only layer over the inference engine, with an explicit PRODUCTION vs. DEMONSTRATION mode toggle. If no production model exists (the current state), the dashboard says so plainly and never silently substitutes a test-fixture model. DEMONSTRATION mode is permanently and visibly labeled "TEST FIXTURE — NOT REAL NETWORK PERFORMANCE." Full detail: `docs/PHASE_7_STREAMLIT_DASHBOARD.md`.

## 19. Implementation

Python 3, `pandas`/`numpy`/`scikit-learn` for the ML pipeline, `matplotlib` for plotting, `joblib` for model persistence, `Streamlit` for the dashboard — all listed in `requirements.txt`. Mininet/`iperf3`/`tc`/Open vSwitch are system-level Linux tools (installed via WSL2 Ubuntu + `apt`, never `pip`) confined entirely to `src/network/` — verified by a project-wide audit that no shell-invoking code exists in `src/ml/`, `src/dashboard/`, or `app.py`. The codebase totals roughly 30 source modules across `src/network/`, `src/ml/`, and `src/dashboard/`, plus 9 CLI scripts and 29 test files.

## 20. Testing

**223 automated tests passing, 2 skipped** (the live-Mininet integration tests, correctly skipped rather than faked, since the environment is unavailable). Coverage includes unit tests for every module, leakage-specific adversarial tests (deliberately-corrupted fixtures designed to slip past a weaker check), a full end-to-end integration test chaining dataset → training → artifact → inference → risk → dashboard contract, a system health check, and an inference-latency benchmark. Full test-suite results and command: `docs/RESULTS.md` Section A.

## 21. Results

See `docs/RESULTS.md` for the complete, two-part breakdown. In summary: every engineering/integration result (test suite, health check, leakage audit, end-to-end fixture pipeline, dashboard startup, inference latency) **passed**. Every empirical network-performance result (real dataset, production model, MAE/RMSE/R², feature importance, actual-vs-predicted, risk distribution) is **NOT AVAILABLE**, because no real Mininet experiment has ever been executed.

## 22. Limitations

- **No real network data was collected**, due to a persistent WSL2 environment blocker on the development machine (`docs/ENVIRONMENT_SETUP.md`, `docs/PHASE_5_REAL_DATA_VALIDATION.md`).
- Consequently, no production model, no real MAE/RMSE/R², no real feature-importance ranking, and no real risk-level distribution exist.
- No lag/rolling feature engineering was implemented — the model (once trained on real data) will only see raw per-interval measurements.
- No browser-based (JS-rendered) verification of the dashboard was possible; startup/import/bare-mode execution were verified instead.
- The reported inference-latency benchmark is a single-laptop, single-process, test-fixture-model measurement — not a production or network-latency figure.

## 23. Future Work

1. Resolve the WSL2/Mininet environment blocker and execute the already-authored pilot sweep (`configs/pilot_sweep.json`).
2. Run `scripts/train_models.py` against the resulting real dataset and populate `docs/RESULTS.md` Section B with real numbers.
3. Cross-validate against an independent ns-3-based dataset.
4. Explore lag/rolling engineered features once real data exists to justify them.
5. Add calibrated prediction intervals (e.g. quantile regression) rather than a point estimate only.
6. Wire a live metrics collector into the existing `InferenceEngine.predict(metrics: dict)` contract for real-time dashboard predictions.

## 24. Conclusion

This project delivers a complete, tested, leakage-safe software system for network-conditions-to-packet-loss regression prediction, spanning experiment design, data generation, ML training/evaluation, inference, risk classification, and a demonstration dashboard — 223 passing automated tests attest to its internal correctness. It does **not** deliver, and does not claim to deliver, empirical evidence that the approach predicts real network packet loss accurately, because the real-network data required for that claim could not be collected in the available environment. The engineering is complete; the empirical validation is the well-defined, explicitly-scoped next step.

## 25. References

- Project planning document: `PROJECT_PLAN.md` (this repository).
- Mininet: Lantz, B., Heller, B., McKeown, N. "A Network in a Laptop: Rapid Prototyping for Software-Defined Networks." *Hotnets* 2010.
- iperf3: ESnet / Lawrence Berkeley National Laboratory, https://github.com/esnet/iperf.
- scikit-learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR* 12, 2011.
- Streamlit: Streamlit Inc., https://streamlit.io.
- ITU-T-style real-time-traffic quality guidance referenced for risk thresholds (`src/ml/risk.py`) — see that module's docstring; a specific ITU-T recommendation number should be added here if required by the submission venue (not fabricated in this draft).

*Additional academic citations (e.g. specific packet-loss-prediction prior work) were not added to avoid fabricating references not actually consulted during this project — see `docs/PAPER_DRAFT.md` Section "References" for the same note.*
