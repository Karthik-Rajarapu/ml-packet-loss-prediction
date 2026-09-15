# Machine Learning-Based Prediction of Packet Loss in Computer Networks

A Computer Networks + Machine Learning project: given currently observable network conditions, predict the packet-loss percentage in the **next** measurement interval, using a real Mininet-based experiment framework, a leakage-safe ML pipeline, an inference engine, and a Streamlit demonstration dashboard.

> **REAL NETWORK DATA STATUS: Not currently available.** Live Mininet experiments could not be executed in the current development environment (WSL2 blocked — see [Limitations](#limitations)). Every component below is implemented and tested against synthetic/fixture data; **no production model or empirical accuracy result exists yet.** See [docs/RESULTS.md](docs/RESULTS.md) for the full, honest breakdown of what is and isn't verified.

## Overview

Reactive packet-loss handling — detecting loss only after it degrades a connection — is the norm in most networks. This project builds a full pipeline to test whether a proactive, ML-based approach is viable: predict next-interval packet loss from current telemetry (RTT, jitter, throughput, utilization, queue depth, retransmissions, current loss) before it happens, then classify the predicted risk as LOW / MODERATE / HIGH.

## Problem

**Research question**: Can current/previous network conditions predict packet loss in the *next* interval, and does an ML model meaningfully outperform a naive "loss doesn't change" baseline?

## Key Idea

Packet loss should emerge **genuinely** from real network queueing under congestion (a real Mininet dumbbell topology, real `iperf3` traffic, real kernel queue drops) — never from an injected `netem loss%` parameter, which would itself be a data-leakage risk if used as the primary loss-generating mechanism. The dataset spans both healthy and congested conditions by sweeping offered load both below and above each run's bottleneck bandwidth.

## Architecture

```
Mininet Network (Dumbbell Topology)
        |
Network Data Collector
        |
Raw Dataset (CSV)  ->  Validation  ->  Feature Engineering
        |
Temporal / Group-Aware Split
        |
ML Models (Naive Baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
        |
Model Artifact  ->  Inference Engine  ->  Risk Classification (LOW/MODERATE/HIGH)
        |
Streamlit Dashboard
```
Full diagram and per-layer interfaces: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features

- Reusable, parameterized Mininet dumbbell topology, capped for resource-limited hardware
- Leakage-safe dataset generation with an independent post-hoc leakage auditor
- Group-aware (never row-level) train/test splitting — chronological or random-group
- 4 regression models + a naive temporal baseline, evaluated on MAE/RMSE/R² and residual distribution
- Native + permutation feature importance
- A production-ready inference engine with a strict feature contract and structural production/test-fixture separation
- A derived LOW/MODERATE/HIGH risk layer with documented, fixed thresholds
- A Streamlit dashboard that never silently substitutes a test model for a missing production one
- 223 automated tests

## ML Models

| Model | Role |
|---|---|
| Naive persistence | Baseline — predicted = current interval's own loss |
| Linear Regression | Simple linear reference (feature-scaled) |
| Decision Tree Regressor | Interpretable nonlinear model |
| Random Forest Regressor | Primary ensemble candidate |
| Gradient Boosting Regressor | Boosted-tree candidate |

## Network Experiment Design

A small dumbbell topology — left hosts → shaped bottleneck link → right hosts — built with Mininet's real Linux networking (not a discrete-event simulator), so packet loss is a genuine emergent property of queueing. See [docs/PHASE_1_NETWORK_EXPERIMENT.md](docs/PHASE_1_NETWORK_EXPERIMENT.md) for the full comparison of alternatives (ns-3, raw `tc`/netem) and why Mininet was chosen.

## Project Structure

```
src/
  network/     Mininet topology, experiment runner, data collectors, validation, sweep config
  ml/          dataset loading, splitting, preprocessing, models, training, evaluation,
               risk classification, model artifacts, inference engine, test-fixture demo data
  dashboard/   Streamlit support: feature inputs, model status, prediction history, report readers
app.py         Streamlit dashboard entry point
scripts/       CLI tools: run_experiment, generate_dataset, dataset_report, train_models,
               predict_packet_loss, smoke_test, benchmark_inference, system_check
tests/         229 automated tests (unit, integration, leakage-specific, end-to-end fixture)
docs/          phase-by-phase documentation, architecture, methodology, results, report, viva prep
configs/       reusable sweep configuration files (e.g. the Phase 5 pilot sweep)
data/raw/      generated experiment CSVs (empty — no real data yet)
models/        trained model artifacts (empty — no production model yet)
reports/       generated evaluation reports/plots (empty — no real training run yet)
requirements.txt
```

## Installation

```bash
git clone <this repository>
cd ml-packet-loss-prediction
pip install -r requirements.txt
```
Python packages only (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`, `pytest`). **System-level tools** for live network experiments — Mininet, `iperf3`, `tc`/iproute2, Open vSwitch — are Linux-only and installed via `apt` inside WSL2 Ubuntu, **not** via pip. See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for the full setup guide and current environment status.

## Running Tests

```bash
py -3 -m pytest tests/ -v
```
Expected: all non-live tests pass; the 2 live-Mininet tests (`tests/test_live_mininet.py`) skip automatically unless `mn`/`iperf3`/`tc`/`ovs-vsctl` are on PATH.

## Running Experiments (requires WSL2 + Mininet)

```bash
python3 scripts/smoke_test.py                                           # verify the environment
python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json --dry-run  # preview, no network touched
sudo python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json       # real run
```

## Training

```bash
python3 scripts/dataset_report.py --input data/raw     # data-quality report (refuses if no real CSVs exist)
python3 scripts/train_models.py --input data/raw        # refuses to run — and writes nothing — if no real dataset exists
```

## Running Inference

```bash
python3 scripts/predict_packet_loss.py --demo --baseline          # naive baseline, no model needed
python3 scripts/predict_packet_loss.py --demo --use-test-fixture  # TEST FIXTURE demo, clearly labeled
python3 scripts/predict_packet_loss.py --metrics-json my_metrics.json  # production model (fails clearly if none exists)
```

## Running the Dashboard

```bash
streamlit run app.py
```
Starts in DEMONSTRATION mode automatically if no production model is found in `models/` (the current state) — the UI states this plainly rather than pretending otherwise.

## Current Status

| Phase | Status |
|---|---|
| 1 — Network experiment framework | Code complete, tested |
| 2 — Dataset generation pipeline | Code complete, tested |
| 3 | No dedicated phase — plumbing folded into Phase 4 |
| 4 — ML modeling/evaluation pipeline | Code complete, tested |
| 5 — Real network data generation | **Blocked** (environment unavailable) |
| 6 — Inference engine | Code complete, tested |
| 7 — Streamlit dashboard | Code complete, tested |
| 8 — Integration, testing, performance | Complete — 223/223 tests passing |

Full status: [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md). Full results (engineering vs. empirical): [docs/RESULTS.md](docs/RESULTS.md).

## Limitations

- **No real network dataset, production model, or empirical accuracy result exists.** The WSL2 features required for Mininet remain disabled on the development machine; see [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md).
- No lag/rolling feature engineering is implemented — only raw per-interval measurements.
- The dashboard has been verified to start and run cleanly, but not click-tested in a real browser (bare-mode execution only).
- Reported inference-latency numbers (median ~12.6ms) are a single-laptop, test-fixture-model software benchmark — not network latency or production performance.

## Future Work

Resolve the WSL2/Mininet blocker; run the already-prepared pilot sweep (`configs/pilot_sweep.json`); train and evaluate the existing (unmodified) pipeline against real data; extend to calibrated prediction intervals and, if data volume justifies it, lag/rolling features; wire a live metrics collector into the existing inference contract.

## Documentation Index

[PROJECT_PLAN.md](PROJECT_PLAN.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/METHODOLOGY.md](docs/METHODOLOGY.md) · [docs/RESULTS.md](docs/RESULTS.md) · [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) · [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md) · [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) · [docs/VIVA_QA.md](docs/VIVA_QA.md) · [docs/PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md) · [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) · phase-by-phase docs in `docs/PHASE_1..8_*.md`.
