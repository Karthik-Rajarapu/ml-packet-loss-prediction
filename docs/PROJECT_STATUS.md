# PROJECT_STATUS.md
## Final Project Status Summary

Last updated: end of Phase 9 (final documentation/finalization phase). See `docs/RESULTS.md` for the full evidence behind every line here.

---

## IMPLEMENTED

All of the following are code-complete and covered by automated tests (223 passing):

- **Network experiment framework** — Mininet dumbbell topology, experiment runner, `iperf3`/`ping`/`tc` data collectors, leakage-safe target construction (`src/network/`)
- **Dataset generation pipeline** — parameterized sweep configuration, sequential experiment runner, manifest generation, dry-run support (`src/network/sweep.py`, `generate.py`, `scripts/generate_dataset.py`)
- **Validation** — row/schema validation, independent leakage auditor, data-quality reporting (`src/network/validation.py`, `src/ml/data_quality.py`)
- **Feature pipeline** — single-source-of-truth schema (`src/network/schema.py`), preprocessing with train-only fitting (`src/ml/preprocessing.py`)
- **ML models** — naive baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting (`src/ml/models.py`, `baselines.py`)
- **Model evaluation framework** — group-aware splitting, MAE/RMSE/R², residual statistics, native + permutation feature importance (`src/ml/split.py`, `train.py`, `evaluate.py`)
- **Model artifacts** — joblib + metadata persistence, production/test-fixture distinction (`src/ml/artifacts.py`)
- **Inference engine** — strict feature-contract validation, structural production/fixture separation, naive-baseline path (`src/ml/inference.py`)
- **Risk classification** — LOW/MODERATE/HIGH from predicted loss, fixed documented thresholds (`src/ml/risk.py`)
- **Streamlit dashboard** — production/demonstration mode toggle, no silent fallback, bounded in-session history (`app.py`, `src/dashboard/`)
- **Integration tests** — a full end-to-end fixture pipeline test (dataset → training → artifact → inference → risk → dashboard contract)
- **System checks** — `scripts/system_check.py` (software health check, separates REQUIRED from REQUIRED-FOR-LIVE-EXPERIMENTS), `scripts/smoke_test.py` (environment probe), `scripts/benchmark_inference.py` (software latency benchmark)

## EMPIRICALLY VALIDATED

**Nothing in this section, because no real network data has been collected.** Listed explicitly, rather than omitted, so the gap is unambiguous:

- Real dataset generation: **NOT VALIDATED**
- Production model training: **NOT VALIDATED**
- Real model performance (MAE/RMSE/R²): **NOT VALIDATED**
- Real feature importance ranking: **NOT VALIDATED**
- Real risk-level distribution: **NOT VALIDATED**
- Real actual-vs-predicted accuracy: **NOT VALIDATED**
- Real-time / live network monitoring integration: **NOT IMPLEMENTED, NOT VALIDATED**

## CURRENTLY BLOCKED

- **Real Mininet data generation** — blocked on WSL2 (`Microsoft-Windows-Subsystem-Linux` / `VirtualMachinePlatform` Windows features remain disabled on the development machine; `mn`/`iperf3`/`tc`/`ovs-vsctl` were never installed). Full troubleshooting history: `docs/ENVIRONMENT_SETUP.md`.
- **Production model training** — blocked transitively (no real data to train on). `scripts/train_models.py` is code-complete and correctly refuses to run against an empty `data/raw/`.
- **Real model performance metrics** — blocked transitively.
- **Real packet-loss prediction evaluation** — blocked transitively.

## Test Summary

```
223 / 223 automated tests passing
2 live-Mininet integration tests skipped (environment unavailable, not faked)
0 failing tests
```

## Bottom Line

This is a complete, tested, leakage-safe **software system**, ready to produce real results the moment the WSL2/Mininet environment blocker is resolved, with no further code changes required. It is not, and does not claim to be, an empirically validated packet-loss prediction model.
