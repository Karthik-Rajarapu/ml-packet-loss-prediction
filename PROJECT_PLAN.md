# PROJECT_PLAN.md
## Machine Learning-Based Prediction of Packet Loss in Computer Networks

Status: **Planning phase — no implementation yet.** This document is the design reference for the project. Implementation begins only after this plan is approved.

---

## 1. Problem Statement

Packet loss is one of the most disruptive impairments in a computer network: it degrades throughput, triggers TCP congestion-control back-off, increases retransmissions and latency, and directly harms real-time applications (VoIP, video conferencing, gaming). Network operators today mostly *react* to packet loss after it is observed (via SNMP alarms, active probing, or user complaints). This project investigates whether the packet loss expected in the **next** measurement interval can be **predicted in advance** from currently observable network conditions, enabling proactive traffic engineering, admission control, or QoS adjustment before loss actually occurs.

## 2. Motivation

- Packet loss is caused by a mix of queueing behavior, congestion, link errors, and traffic load — a nonlinear interaction that is hard to capture with static thresholds.
- Machine learning is well suited to learning nonlinear relationships between observable network telemetry (RTT, jitter, throughput, queue occupancy, retransmissions) and the resulting loss, without requiring an explicit analytical queueing model.
- A working predictive model has practical value: it could feed into adaptive bitrate control, SD-WAN path selection, or early-warning dashboards for network operators.
- This project deliberately grounds the ML work in real Computer Networks concepts (queueing, congestion control, traffic shaping) rather than treating it as a generic tabular ML exercise — this is explicitly required for a Computer Networks course project.

## 3. Project Objectives

1. Generate a labeled, time-ordered dataset of network condition measurements paired with subsequent packet loss, from controlled, reproducible network experiments (not a downloaded Kaggle dataset).
2. Build a regression pipeline that predicts the packet-loss percentage in the next measurement interval from current/past network telemetry.
3. Optionally layer a LOW/MODERATE/HIGH risk classifier on top of the regression output.
4. Rigorously avoid data leakage given the temporal nature of the prediction task.
5. Compare a meaningful set of models against a naive baseline, using appropriate regression metrics.
6. Present the result through a simple, explainable dashboard suitable for a live viva demonstration.
7. Produce a codebase and report where every component can be explained by each team member.

## 4. Research Question

> **Given the currently observable network conditions (RTT, jitter, throughput, utilization, queue occupancy, retransmissions, connection load, etc.), can a machine learning model predict the packet loss percentage in the next measurement interval more accurately than a naive persistence baseline?**

Secondary question: which observable features are most predictive of imminent packet loss, and does this align with networking theory (e.g., queue occupancy and utilization approaching link capacity should be strong predictors, consistent with queueing-theoretic tail-drop behavior)?

## 5. Proposed Solution

Build an end-to-end pipeline:

1. Emulate a small, controllable network topology on real Linux networking infrastructure (not a pure math simulation), so that packet loss is a *genuine emergent property* of queueing and congestion, not an artificially injected label.
2. Drive real traffic (TCP/UDP via `iperf3`, probes via `ping`) across the topology while systematically varying network conditions (bandwidth, base delay, jitter, number of concurrent flows, traffic load).
3. Sample network telemetry at fixed intervals during each experiment run and log it as time-series rows tagged with an `experiment_id`.
4. Construct the supervised learning target by shifting the packet-loss measurement forward by one interval **within each experiment run**.
5. Train and compare regression models with a temporally/group-aware train/test split.
6. Wrap the best model in a small prediction module and a Streamlit dashboard for demonstration.

## 6. Computer Networks Concepts Involved

- **Queueing theory & tail drop**: router/switch queues are finite; when arrival rate exceeds service rate for long enough, the queue fills and packets are dropped. This is the primary real-world source of loss we aim to emulate (not synthetic random loss).
- **Congestion control**: TCP reacts to loss/RTT increases by shrinking its congestion window (e.g., Reno/CUBIC via Linux's default stack), which affects throughput and retransmissions — a naturally emerging feature source.
- **Traffic shaping / policing**: `tc` (traffic control) with `netem` (delay, jitter, stochastic loss) and `tbf`/`htb` (bandwidth limiting) are used to emulate WAN-like links.
- **Latency, jitter, RTT**: measured via ICMP `ping` and UDP jitter reporting in `iperf3`.
- **Retransmissions**: TCP retransmit counts (from `ss -ti` or `iperf3` TCP reports) indicate the transport layer's own loss recovery activity — useful predictive signal, but must be handled carefully to avoid leakage (see Section 16).
- **Utilization / bottleneck links**: classic dumbbell topology (multiple senders sharing one bottleneck link to a receiver) is the standard way networking courses demonstrate congestion and is directly reused here.
- **Active flows / connection load**: number of concurrent flows sharing a link is a textbook congestion driver, and is a controllable experimental variable here (`active_connections`).

## 7. Network Experiment / Data-Generation Strategy

We generate data from **real emulated network runs**, not synthetic random sampling. General design:

- **Topology**: dumbbell topology — N sender hosts → shared bottleneck link (with configurable bandwidth/delay/jitter/queue size via `tc`) → 1 receiver host. This is the standard topology for demonstrating congestion and shared-link contention in networking courses.
- **Controlled variables per experiment run**: bottleneck bandwidth, base latency, jitter, queue size (`tc qdisc` limit), number of concurrent flows, traffic mix (TCP vs UDP), sending rate/load.
- **Traffic generation**: `iperf3` for throughput/loss/retransmit-bearing flows, `ping` run in parallel for independent RTT/jitter probing.
- **Sampling**: telemetry captured every fixed interval (e.g., 1–2 seconds) for the duration of each run (e.g., 60–120 seconds), producing a time series of rows per `experiment_id`.
- **Parameter sweep**: an experiment driver script iterates over a grid/random sample of the controlled variables above, running many short experiments automatically to build up dataset diversity and volume.
- **Important design choice — loss should emerge, not be dictated**: we avoid setting an explicit `netem loss%` as the sole loss mechanism, because that would make the "target" a near-deterministic function of a single injected knob (a form of leakage/triviality — the model would just learn to invert our own parameter instead of learning real congestion dynamics). Instead, the primary loss-generating mechanism is **queue overflow from real congestion** (bandwidth-limited bottleneck + multiple flows + finite queue). A smaller secondary subset of runs may add modest stochastic `netem` loss to model link-error-style loss and test generalization — this will be clearly labeled/separated in the dataset, not silently mixed in as the main signal.

## 8. Comparison of Simulation / Emulation Approaches

| Approach | Real traffic / real stack | Setup complexity | Viva demonstrability | Feature realism | Windows compatibility | Verdict |
|---|---|---|---|---|---|---|
| **ns-3 (discrete-event simulation)** | No (simulated stack) | High (C++/Python bindings, steep learning curve) | Low–Medium (harder to show "real" traffic live) | Medium (depends entirely on the accuracy of custom instrumentation) | Poor natively; needs Linux/WSL2 | Not recommended as primary; possible future extension |
| **Mininet (Linux netns + OVS emulation)** | Yes (real Linux TCP/IP stack, real queues) | Medium (needs Linux; Python topology API is approachable) | High (can run live demo: start topology, generate traffic, show loss happen, show prediction) | High (loss is a genuine emergent effect of real queueing) | Needs WSL2/Linux VM | **Recommended** |
| **Raw Linux `tc`/`netem` + network namespaces or veth pairs (no Mininet)** | Yes | Medium-low for a *simple* topology, but you re-implement what Mininet already provides for anything beyond 2 hosts | Medium-High | High | Needs WSL2/Linux | Good fallback if Mininet install proves troublesome |
| **Controlled client/server over a real LAN/Wi-Fi with `tc netem` shaping one hop** | Yes (fully real) | Low | Medium (less "textbook networking" framing, but very tangible) | High, but less controllable/reproducible (subject to real network noise) | Works from Windows if server is a separate Linux box/VM | Reasonable fallback, weaker reproducibility |
| **Pure Python synthetic simulation (no real stack)** | No | Low | Low (defeats the "not a generic Kaggle project" requirement) | Low–Medium | Works everywhere | Rejected |

## 9. Recommended Approach and Justification

**Recommended: Mininet, run inside WSL2 (Ubuntu), using a dumbbell topology with `tc`/`netem`/`tbf` for link shaping, `iperf3` + `ping` for traffic/probing.**

Why:
- Mininet runs actual Linux networking (real sockets, real TCP congestion control, real finite queues), so packet loss is a **genuine emergent phenomenon**, not a label we hard-code — this is essential for a defensible ML problem and satisfies the "not a generic Kaggle project" requirement.
- It is the most recognized network-emulation tool in academic Computer Networks courses (originally built for SDN research at Stanford), so it is easy to justify and explain in a viva, and graders are likely to recognize it.
- Its Python API makes topology and experiment scripting straightforward and fully reproducible (a script rebuilds the exact topology every run).
- It directly supports the required controllable variables (bandwidth, delay, jitter, queue size via link parameters that translate to `tc` under the hood; number of flows via multiple hosts).
- It is demonstrable live in a viva: start the topology, run traffic, show real loss occurring, then show the trained model predicting it.
- WSL2 is a standard, well-documented path to run Mininet from a Windows machine (Windows 10 supports WSL2), so the team is not blocked by not having native Linux machines.

Fallback if Mininet installation is problematic on a given machine: drop to raw `tc netem` + Linux network namespaces with a hand-rolled 2–3 host topology (Section 8, row 3) — conceptually the same data, less topology flexibility, but requires less tooling.

NS-3 is explicitly **not** the primary approach: it doesn't generate "real" traffic in the same demonstrable sense, has a much steeper setup/learning cost for a course-timeline project, and the marginal fidelity gain isn't worth the complexity for this scope. It's listed as a future enhancement (Section 24) for teams that want to cross-validate against a second, independent data-generation method.

## 10. Dataset Schema

Each row = one sampling interval within one experiment run.

| Column | Type | Description |
|---|---|---|
| `experiment_id` | string | Unique id per experiment run (topology/parameter configuration + run number) |
| `interval_index` | int | Sequence position of this row within the run (0, 1, 2, …) |
| `timestamp` | float/datetime | Wall-clock or relative time of the sample |
| `bandwidth_limit_mbps` | float | Configured bottleneck bandwidth for this run (experimental condition, known in advance — not leakage since it's a setup parameter, not a measurement) |
| `base_latency_ms` | float | Configured base one-way/RTT delay for this run |
| `configured_jitter_ms` | float | Configured jitter for this run |
| `num_flows` | int | Number of concurrent sender flows in this run (proxy for `active_connections`) |
| `traffic_type` | categorical | TCP / UDP / mixed |
| `rtt_ms` | float | Measured RTT at this interval (from `ping`) |
| `jitter_ms` | float | Measured jitter at this interval (from `iperf3`/ping variance) |
| `throughput_mbps` | float | Measured throughput at this interval |
| `utilization_pct` | float | throughput / bandwidth_limit, i.e. link load fraction |
| `queue_length` | float/int | Bottleneck qdisc queue occupancy at sampling time (from `tc -s qdisc`) |
| `packets_sent` | int | Packets sent in this interval |
| `packets_received` | int | Packets received in this interval |
| `retransmissions` | int | TCP retransmissions observed in this interval |
| `packet_loss_current` | float | Packet loss % **measured in this same interval** (valid input feature — see Section 16) |
| `packet_loss_next` | float | **Target**: packet loss % measured in the *next* interval of the same `experiment_id` |

This is a working draft; exact column list may be trimmed/extended once instrumentation is built and we see what's reliably measurable at 1–2 second granularity.

## 11. Feature Definitions

Primary model inputs (all describing the *current* interval or earlier history, never the target interval):

- **Direct telemetry**: `rtt_ms`, `jitter_ms`, `throughput_mbps`, `utilization_pct`, `queue_length`, `retransmissions`, `packets_sent`, `packet_loss_current`.
- **Configured conditions** (known a priori, not measurements): `bandwidth_limit_mbps`, `base_latency_ms`, `configured_jitter_ms`, `num_flows`, `traffic_type`.
- **Engineered temporal features** (computed only from *past* intervals of the same `experiment_id`):
  - Lag features: `rtt_ms_lag1`, `throughput_mbps_lag1`, `packet_loss_current_lag1`, etc.
  - Rolling statistics over a short trailing window (e.g., last 3–5 intervals): rolling mean/std of RTT, throughput, queue length — captures trend/volatility rather than a single noisy sample.
  - Rate of change: e.g., `delta_queue_length = queue_length_t - queue_length_(t-1)`, `delta_throughput`.
- **Derived ratios**: `utilization_pct` (throughput vs. configured bandwidth), `retransmission_rate = retransmissions / packets_sent`.

Feature selection will be finalized after EDA (correlation with target, redundancy checks, e.g. `throughput` vs `utilization_pct` are likely highly correlated).

## 12. Target Variable

`packet_loss_next`: packet loss percentage measured in the interval **immediately following** the row's own interval, within the same `experiment_id`. Computed as a forward shift (`groupby(experiment_id).shift(-1)`) of the measured per-interval loss. The last interval of each run has no valid target and is dropped.

Primary formulation: **regression** (continuous percentage, 0–100%).
Secondary formulation (optional layer): bucket the regression output into `LOW` / `MODERATE` / `HIGH` risk using thresholds derived from the training data's distribution (e.g., tertiles or domain-informed cutoffs like <1% / 1–5% / >5%), decided after seeing the real data distribution.

## 13. ML Formulation

- **Type**: supervised regression, framed as one-step-ahead time-series forecasting **within grouped runs** (not a single continuous global time series — each `experiment_id` is its own short trajectory under fixed configured conditions).
- **Unit of prediction**: one row (interval) → one scalar (`packet_loss_next`).
- **Optional secondary task**: multiclass classification (LOW/MODERATE/HIGH) derived from the regression target, evaluated independently with classification metrics — kept clearly secondary so the project doesn't drift from the required regression formulation.

## 14. Candidate Models

| Model | Role | Justification |
|---|---|---|
| **Naive persistence baseline** (`packet_loss_next_pred = packet_loss_current`) | Baseline | Standard baseline for one-step-ahead forecasting; any real model must beat "loss doesn't change" to be meaningful |
| **Linear Regression** (+ optionally Ridge) | Baseline ML model | Simple, interpretable, tests whether relationships are largely linear |
| **Decision Tree Regressor** | Interpretable nonlinear model | Easy to visualize/explain in viva; captures threshold effects (e.g., queue near capacity) |
| **Random Forest Regressor** | Primary candidate | Handles nonlinearity and feature interactions well, robust to noise typical of network measurements, gives feature importances |
| **Gradient Boosting (scikit-learn `HistGradientBoostingRegressor` or XGBoost, if justified by results)** | Stretch candidate | Often strongest tabular performance; included only if it meaningfully beats Random Forest, to avoid "throwing every model at it" without justification |

We will not include deep learning (e.g., LSTM) in the core scope — dataset size from a course-timeline experiment budget is unlikely to be large enough to justify it, and it adds interpretability/viva-complexity cost. Noted as a future enhancement.

## 15. Evaluation Metrics

**Regression (primary):**
- **MAE** — average absolute error in loss percentage points; easiest to explain to a non-ML audience ("on average we're off by X%").
- **RMSE** — penalizes large misses more, relevant since large under-predictions of loss are operationally worse.
- **R²** — how much variance is explained vs. a mean-predictor baseline.
- All three reported against the naive persistence baseline to show genuine lift.

**Classification (secondary, if built):**
- Accuracy, Precision, Recall, F1 (macro-averaged, since HIGH-risk events are likely rare/imbalanced), and a confusion matrix to show where risk levels are confused (e.g., MODERATE vs HIGH boundary).

## 16. Data Leakage Risks

This is treated as a first-class concern given the temporal target.

1. **Target-interval contamination**: any feature accidentally computed using data from the *target* interval (e.g., a "current" throughput figure that actually averages across the boundary into the next interval due to a sampling/parsing bug) would leak the answer. Mitigation: strict interval boundary alignment in the data-collection parser, verified with unit tests on synthetic timestamps.
2. **Injected-parameter leakage**: if we used an explicit `netem loss%` knob as the *primary* loss mechanism and then included that knob as a feature, the model would trivially invert a formula instead of learning network behavior. Mitigation: as decided in Section 7, loss primarily emerges from real congestion, not an injected label; any runs using `netem` stochastic loss are a clearly separated subset (or excluded from the primary training set) rather than the default data-generation mode.
3. **Row-level random shuffling across time**: consecutive intervals within the same `experiment_id` are highly autocorrelated (same underlying configured conditions and evolving queue state). A naive random train/test split at the row level would put adjacent-in-time rows from the *same run* on both sides of the split, letting the model "peek" at the local trajectory it's being tested on. Mitigation: see Section 17 — split by `experiment_id`, never by row.
4. **Feature computed from the future within a run**: any rolling/lag feature must be computed using only `shift(+k)` (past), never `shift(-k)` (future), relative to the row's own interval.
5. **`packet_loss_current` is safe as a feature** because it describes the same interval as the other current-interval inputs and strictly precedes the target interval — but it must be double-checked that its measurement window doesn't overlap the target's window (see leakage risk 1).

## 17. Temporal Train/Test Strategy

- **Group by `experiment_id`, never split by individual row.** Entire experiment runs go wholly into train or wholly into test.
- Two complementary split strategies will both be evaluated:
  - **Random group split** (e.g., 70/15/15 train/val/test by `experiment_id`) — tests generalization across different configured conditions.
  - **Chronological split** (train on earlier-run experiments, test on later-run experiments, by run start time) — tests generalization over time / measurement drift, closer to real deployment ("train on past data, predict on future data").
- Cross-validation, if used for model selection, will use **GroupKFold** (grouped by `experiment_id`) rather than plain KFold, so folds never split a single run's time series across train and validation.
- Feature engineering (lag/rolling features) is computed **within each group independently** and only from that group's own past — never across group boundaries, and never using information from held-out groups.

## 18. System Architecture

```
Network Experiment (Mininet, WSL2)
        │  (dumbbell topology, tc/netem/tbf shaping, iperf3 + ping traffic)
        ▼
Data Collection (interval sampling scripts, parsers for iperf3/ping/tc output)
        ▼
Raw Dataset (per-experiment CSV/JSON, data/raw/)
        ▼
Data Validation (schema checks, range checks, missing-interval handling)
        ▼
Preprocessing (cleaning, type coercion, merging into unified table)
        ▼
Feature Engineering (lag/rolling features, target shift — group-aware)
        ▼
Train/Val/Test Split (GroupKFold / chronological by experiment_id)
        ▼
ML Training (baseline → Linear → Tree → Random Forest → [Gradient Boosting])
        ▼
Model Evaluation (MAE/RMSE/R², baseline comparison, feature importance)
        ▼
Prediction Module (loads trained model + preprocessing pipeline, scores new feature rows)
        ▼
Dashboard (Streamlit: current metrics → predicted loss → risk level → feature importance)
```

Each stage is a separate, independently runnable component (script or module) with a clear input/output contract (e.g., raw CSVs in → validated CSV out), so any team member can own and explain one stage without needing to understand the whole pipeline internals.

## 19. Proposed Directory Structure

```
ml-packet-loss-prediction/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── network_experiments/
│   ├── topology.py            # Mininet dumbbell topology definition
│   ├── run_experiment.py      # runs one experiment (given a parameter set), collects raw logs
│   ├── sweep_experiments.py   # drives many runs across a parameter grid/random sample
│   └── configs/
│       └── sweep_config.yaml  # ranges for bandwidth, delay, jitter, num_flows, etc.
├── data_collection/
│   ├── parsers.py             # parse iperf3/ping/tc raw output into interval rows
│   └── sampler.py             # interval-based sampling loop used during a run
├── data/
│   ├── raw/                   # one file (or folder) per experiment_id
│   └── processed/             # cleaned, merged, feature-engineered dataset
├── preprocessing/
│   ├── validate.py            # schema/range checks
│   └── clean.py                # merge raw files, handle missing intervals
├── features/
│   └── engineer.py            # lag/rolling features, target shift, group-aware
├── training/
│   ├── split.py                # GroupKFold / chronological split helpers
│   ├── baseline.py             # naive persistence baseline
│   ├── train_models.py         # trains Linear/Tree/RF/[GBM], saves artifacts
│   └── evaluate.py             # metrics, comparison table, feature importance plots
├── models/                     # saved trained model artifacts (joblib)
├── prediction/
│   └── predictor.py            # loads a model + preprocessing, scores new feature rows
├── dashboard/
│   └── app.py                  # Streamlit app
├── notebooks/
│   └── eda.ipynb               # exploratory data analysis
├── docs/
│   └── diagrams/                # architecture/topology diagrams for the report
└── tests/
    ├── test_parsers.py
    ├── test_features.py         # esp. leakage/shift correctness
    └── test_split.py            # esp. group-purity of splits
```

Kept deliberately shallow and script-first (no unnecessary web framework, no microservices) — appropriate for a course project timeline and for every member to be able to explain any file.

## 20. Dashboard Concept

A single-page **Streamlit** app (chosen over a custom frontend framework for minimal complexity and fast iteration):

- **Top panel — current network metrics**: RTT, jitter, throughput, utilization, queue length, active connections (either replayed from a recorded experiment run for demo purposes, or read live if a Mininet demo is running alongside).
- **Middle panel — prediction**: predicted packet loss for the next interval (large numeric display), plus a prediction interval/uncertainty band if using a model that supports it (e.g., quantile regression or RF's tree-spread as an approximate interval) — included only if it doesn't add disproportionate complexity.
- **Risk panel**: LOW/MODERATE/HIGH badge derived from the predicted value.
- **Explainability panel**: feature importance bar chart (native for tree models) so the demo can show *why* the model predicted what it did — strengthens the viva narrative ("the model flagged rising queue length and utilization, consistent with congestion theory").
- **History panel**: a simple time-series chart of recent actual vs. predicted loss for the loaded experiment run, to visually show tracking accuracy.

No user accounts, no database, no live production deployment — this is a demonstration tool reading from the project's own saved model + a chosen dataset/run.

## 21. Development Phases

| Phase | Deliverable |
|---|---|
| 0 | Environment setup: WSL2 + Mininet + iperf3 installed and verified with a manual "hello world" topology run |
| 1 | Dumbbell topology script + single manual experiment run producing one raw log |
| 2 | Data collection sampler + parsers turning raw logs into schema-conformant interval rows |
| 3 | Parameter sweep driver; run enough experiments to produce a first real dataset (`data/raw/`) |
| 4 | Data validation + EDA notebook: distributions, correlations, sanity checks, decide final feature list |
| 5 | Preprocessing + feature engineering with group-aware target shift; leakage unit tests |
| 6 | Baseline + candidate models trained with GroupKFold/chronological split; evaluation report with metric comparison table |
| 7 | Prediction module wrapping the chosen best model |
| 8 | Streamlit dashboard integrating the prediction module |
| 9 | Documentation, final report, and viva rehearsal (each member can explain their owned stage end-to-end) |

## 22. Team-Work Breakdown Suggestions

For a typical 3–4 person team:

- **Networking/Data-Generation owner**: Phases 0–3 (Mininet topology, experiment orchestration, raw data collection).
- **Data/ML owner**: Phases 4–6 (validation, EDA, feature engineering, model training/evaluation) — the leakage-safety work in Section 16/17 should be owned/reviewed by this person specifically.
- **Application/Dashboard owner**: Phases 7–8 (prediction module, Streamlit dashboard).
- **Docs/Integration/QA owner** (can overlap with above roles in a smaller team): keeps PROJECT_PLAN.md and the final report in sync with what was actually built, writes tests, coordinates the viva script.

All members should be able to explain the full pipeline at a high level even if they only implemented one stage, since a viva may probe any part.

## 23. Risks and Limitations

- **WSL2/Mininet setup friction**: environment setup varies across team members' Windows machines; mitigate by documenting exact setup steps and testing on more than one machine early (Phase 0).
- **Limited experiment volume/diversity**: a course-timeline data-generation budget will produce a modest dataset; parameter coverage may be narrow, risking a model that doesn't generalize far beyond the tested conditions — must be stated explicitly as a limitation in the report, not hidden.
- **Class imbalance for the optional risk classifier**: HIGH-loss events are expected to be rarer than LOW; macro-averaged metrics and possibly class weighting will be needed, and this should be called out rather than reporting misleading accuracy alone.
- **Host-machine measurement noise**: emulated timing/queueing results can be affected by CPU load on the machine running Mininet (since it's not dedicated hardware); should be documented as a reproducibility caveat, and ideally experiments run on a relatively idle machine.
- **Generalization boundary**: the model is trained on a specific emulated topology (dumbbell, particular link types); it should not be over-claimed as generalizing to arbitrary real-world networks — this is appropriate to state as scope, not a flaw.
- **Time budget**: the full phase list is ambitious for a college-course timeline; Phase 9 (dashboard polish) and the gradient-boosting stretch model are the first candidates to cut if time is short.

## 24. Future Enhancements

- Cross-validate the emulated dataset against an independent **ns-3** simulation to check whether findings transfer across data-generation methods.
- Sequence models (LSTM/GRU or temporal convolutional networks) if dataset volume grows large enough to justify them.
- Multi-hop / more complex topologies, and wireless emulation via `mininet-wifi`.
- Quantile regression or conformal prediction for principled prediction intervals (rather than an approximate tree-spread proxy).
- SHAP-based explainability instead of (or alongside) native feature importances, for per-prediction explanations.
- Online/incremental retraining as new experiment runs are collected.
- Validation against public real-world network trace datasets (e.g., MAWI), where licensing/availability permits, as an external generalization check.

---

*This plan is a living document. It should be updated if experimentation reveals that a planned feature is unmeasurable, a planned model underperforms, or scope needs to be trimmed for time — but changes should be reflected here, not just in code, so the plan stays the source of truth for the report and viva.*
