# METHODOLOGY.md

Terminology and definitions here are consistent with the code (`src/network/schema.py`, `src/ml/`) and with every prior phase document — nothing here redefines a term differently from how it's implemented.

## 1. Problem Definition

Given a snapshot of currently observable network conditions at time interval `T` (RTT, jitter, throughput, utilization, queue occupancy, retransmissions, current packet loss, and the configured link conditions), predict the packet-loss percentage that will be measured during the **next** interval, `T+1`. This is framed as a supervised **regression** problem, not a classification problem, with an optional derived LOW/MODERATE/HIGH risk label computed from the regression output.

## 2. Motivation

Packet loss results from a nonlinear interaction of queueing behavior, congestion, and traffic load. Machine learning can learn this relationship directly from observed telemetry without an explicit analytical queueing model, and a genuinely predictive model has practical value for proactive traffic engineering, admission control, or early-warning dashboards — see `PROJECT_PLAN.md` Section 2 for the original motivation, unchanged since planning.

## 3. Network Setup

A small, resource-conscious Mininet-emulated network (real Linux network namespaces, real TCP/IP stack, real kernel queueing — not a discrete-event simulation) is used so that packet loss is a genuine emergent property of the network, never an injected label. See `docs/PHASE_1_NETWORK_EXPERIMENT.md` Sections 1–3 for the full comparison of alternatives (ns-3, raw `tc`/netem, real LAN) and the justification for choosing Mininet.

## 4. Dumbbell Topology

```
    h1 --\                                  /-- h3
          s1 ====[ shaped bottleneck ]==== s2
    h2 --/                                  \-- h4
```

Left-side hosts connect to switch `s1` over fast, unshaped links; right-side hosts connect to `s2` the same way. The **only** shaped link is `s1<->s2` (bandwidth, delay, queue depth via `tc`/`TCLink`), so it is a genuine, isolatable bottleneck — the standard topology for demonstrating congestion in networking coursework. Host count is capped at 4 by `ExperimentConfig` to fit the development machine's RAM (`src/network/config.py`).

## 5. Network Parameters

`ExperimentConfig` (`src/network/config.py`): `bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts`, `traffic_type` (udp/tcp), `offered_load_mbps`, `active_connections`, `sample_interval_s`, `duration_s`, `target_horizon_intervals`. The Phase 2 sweep (`src/network/sweep.py`) expresses offered load as a **factor of bandwidth** (e.g. 0.7×, 1.3×) precisely so the dataset spans both below- and above-bottleneck conditions regardless of which bandwidth value is being tested — see Section 6 below for why this matters.

## 6. Data Collection Methodology

For each sampling interval: `ping -c 3` gives mean RTT and a jitter proxy (population std-dev of the 3 samples); `tc -s qdisc show` gives bottleneck queue backlog; a single long-running `iperf3 -J -i <interval>` client/server pair gives per-interval throughput and (for UDP) packet counts/loss directly from iperf3's own JSON report — chosen over TCP specifically because iperf3's UDP mode reports `packets`/`lost_packets`/`lost_percent` directly, while its TCP mode only reports bytes and retransmits (`docs/PHASE_1_NETWORK_EXPERIMENT.md` Section 3). Offered load is deliberately set **above** bottleneck bandwidth for a subset of runs so loss emerges from real queue overflow, not an injected `netem loss%` parameter (Section 10 explains why the latter would be a leakage risk).

## 7. Feature Definitions

15 features (`network/schema.py::FEATURE_COLUMNS`), all describing interval `T` or earlier:

| Feature | Unit | Meaning |
|---|---|---|
| `bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts` | Mbps, ms, packets | configured link conditions (known before the run) |
| `traffic_type` | categorical | udp/tcp |
| `active_connections` | count | configured concurrent flows |
| `current_rtt_ms`, `current_jitter_ms` | ms | measured this interval |
| `throughput_mbps`, `bandwidth_utilization_pct`, `packet_rate_pps` | Mbps, %, pkt/s | measured this interval |
| `queue_length` | packets | bottleneck backlog snapshot |
| `retransmissions`, `packets_sent`, `packets_received` | count | measured this interval |
| `packet_loss_pct` | % | **current**-interval loss — safe as an input (see Section 10) |

## 8. Target Definition

`target_next_packet_loss_pct` — `packet_loss_pct` measured `target_horizon_intervals` (default 1) intervals **after** the row's own interval, within the same `experiment_id`. Computed once, in one place, by `network/targets.py::add_next_interval_target()`.

## 9. Future-Interval Prediction Formulation

At prediction cutoff `T`: features = everything observable at or before `T`; target = packet loss during `[T+1]`. This is a one-step-ahead forecasting formulation, evaluated per-`experiment_id` group (each experiment is its own short time series under fixed configured conditions), not a single continuous global series.

## 10. Leakage Prevention

Three independent layers, all still in force, re-audited in Phase 8 with zero issues found:
1. **Structural**: the target column is never included in `FEATURE_COLUMNS` — there is no code path where it could be selected as an input.
2. **Construction-time**: `add_next_interval_target()` groups by `experiment_id` and orders by `interval_index` before shifting, so a target can never be pulled from a different experiment or the wrong row; the last row(s) of each experiment get `None` rather than a fabricated value.
3. **Independent verification**: `network/validation.py::validate_no_leakage()` re-derives, from the raw current-vs-target columns, that every non-null target matches its true future value and never crosses an experiment boundary — run automatically on every dataset load.
4. **Inference-time**: `ml/inference.py::validate_feature_input()` rejects any caller-supplied field whose name suggests target/future information, with a specific error message.

No lag/rolling/windowed features exist anywhere in the codebase (confirmed by repository grep audit, Phase 8), so there is no windowed-feature leakage surface to reason about beyond what's listed above.

## 11. Dataset Validation

`network/validation.py::validate_rows()`: required columns present, `interval_index` contiguous from 0 per experiment, timestamps non-decreasing, `packets_received <= packets_sent`, `packet_loss_pct` within `[0,100]`. `ml/data_quality.py` (Phase 5) additionally reports duplicate rows/pairs, missing-interval gaps, and zero-target-variance experiments as data-quality signals (not hard failures).

## 12. Feature Engineering

Limited, deliberately, to the raw measurements in Section 7 — no derived lag/rolling/interaction features are computed. This was an explicit scope decision (`docs/PHASE_4_ML_MODELING.md` Section 18) given no real dataset has existed at any point to justify or validate more elaborate engineered features against.

## 13. Temporal / Group-Aware Evaluation

`src/ml/split.py`: `chronological_split()` (train on earlier experiments, test on later ones — simulates deploying on past data, predicting on future data) and `random_group_split()` (fixed-seed random hold-out of whole experiment groups — tests generalization across configurations). Both split on **entire experiment groups**, never rows, with an internal `assert_no_group_overlap()` guarantee. A plain row-level `train_test_split(..., random_state=...)` is never used, since consecutive intervals within one experiment are highly autocorrelated.

## 14. ML Algorithms

Linear Regression (scaled features — the only model requiring it), Decision Tree Regressor (`max_depth=8`), Random Forest Regressor (`n_estimators=200`), Gradient Boosting Regressor (`sklearn` defaults) — all `RANDOM_SEED=42`. XGBoost was deliberately not added: the brief said to consider it only if results justified it, and with no real data ever trained, there has been nothing to justify it against.

## 15. Baseline

Naive persistence: `predicted_next_loss = current packet_loss_pct`. Evaluated in the exact same pipeline and comparison table as the ML models — the point of a baseline is to prove the ML models learn something beyond "loss doesn't change," which cannot be claimed without real data (see `docs/RESULTS.md`).

## 16. Evaluation Metrics

MAE, RMSE, R² (`ml/evaluate.py::compute_regression_metrics`), plus residual distribution statistics (`residual_stats`: mean/median/max absolute error, % of predictions within 1/5 percentage points) — added specifically because packet loss is often near-zero, so aggregate metrics alone can hide how errors are actually distributed.

## 17. Feature Importance

Two methods, both reusing the trained pipeline directly (never duplicated logic): native impurity-based importance for tree models (`random_forest_feature_importance`, correctly labeled in post-one-hot-transform space via the preprocessor's own `get_feature_names_out()`), and permutation importance on held-out test data only (`permutation_feature_importance`, operating in raw feature space). Both are described as "predictive importance," never causal — see `docs/VIVA_QA.md` Q27.

## 18. Risk Classification

`ml/risk.py`: LOW (≤0.1%), MODERATE (0.1–1.0%), HIGH (>1.0%), following widely-used real-time-traffic quality guidance (ITU-T-style voice/video thresholds). Fixed a priori, never fit or tuned against any dataset — derived from the **predicted** value only.

## 19. Inference Architecture

`ml/inference.py`: `validate_feature_input()` → `build_feature_frame()` → `InferenceEngine.predict()` (runs the exact saved training pipeline, so preprocessing can never drift out of sync between training and inference) → risk classification → `PredictionResult`. `load_production_model()`/`load_test_fixture_model()` enforce that a test-fixture artifact structurally cannot be mistaken for or substituted as a production one.

## 20. Dashboard

`app.py` + `src/dashboard/`: presentation-only Streamlit layer, PRODUCTION/DEMONSTRATION mode toggle with no silent fallback, manual feature-input form, in-session bounded prediction history (never persisted, never fabricates an "actual" outcome column), and read-only display of any real Phase 4 report artifacts — see `docs/PHASE_7_STREAMLIT_DASHBOARD.md` for full detail.
