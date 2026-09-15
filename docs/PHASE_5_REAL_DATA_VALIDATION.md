# PHASE_5_REAL_DATA_VALIDATION.md
## Real Network Data Generation & Empirical Model Validation

Status: **BLOCKED before any live-data step.** The required Mininet/WSL2 environment is unavailable on this development machine (confirmed again this phase — see Section 9). Per the Phase 5 Data Rule, nothing past environment verification was executed: no pilot, no full sweep, no real training, no real metrics. This document records what was built and verified in preparation, and states plainly what has **not** happened.

---

## 1. Objective

Generate a diverse, real, Mininet-based network-experiment dataset using the existing Phase 1/2 framework, validate it, and produce the project's first defensible empirical results — a real `Model | MAE | RMSE | R²` comparison table for the naive baseline plus four ML models, using Phase 4's existing, unmodified pipeline. **Not accomplished this phase** — see Section 9.

## 2. Experimental Topology (unchanged from Phase 1)

The dumbbell topology from `src/network/topology.py` — left-side hosts → shaped bottleneck → right-side hosts — was reused exactly as-is. No topology changes were made; Phase 5 is a data-generation and validation exercise on top of Phase 1/2/4, not a redesign of any of them (per the brief's own instruction not to rewrite working modules unnecessarily).

## 3. Network Parameters

Also unchanged: `network.config.ExperimentConfig` fields (`bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts`, `traffic_type`, `offered_load_mbps`, `active_connections`, `sample_interval_s`, `duration_s`, `target_horizon_intervals`). Phase 5 does not add new parameters — it adds a specific, documented *choice* of values for a pilot sweep (Section 5).

## 4. Parameter Sweep — Reused Infrastructure

`scripts/generate_dataset.py` and `src/network/sweep.py` (Phase 2) are reused unmodified. Confirmed this phase, by inspection, that the existing sweep design already satisfies Phase 5 Step 3's core requirement — that the dataset must contain both healthy and congested conditions — **by construction**: `SweepConfig.offered_load_factors` expresses offered load as a multiplier of each sweep's own bandwidth value (e.g. `0.7` = below bottleneck, `1.3` = above), and the default sweep already includes both a below- and an above-bottleneck factor. No dimension needed to be added to make this true; it was already a deliberate Phase 2 design choice (`docs/PHASE_2_DATA_GENERATION.md` Section 2), re-verified here rather than re-derived from scratch.

## 5. Pilot Experiment Design (`configs/pilot_sweep.json`)

A small, dedicated pilot sweep was authored this phase — see `configs/README.md` for the full rationale. Summary:

| Dimension | Values | Rationale |
|---|---|---|
| `bottleneck_bw_mbps_values` | `[2.0, 5.0]` | Two distinct capacities |
| `bottleneck_delay_ms_values` | `[0.0, 20.0]` | No delay vs. modest WAN-like delay |
| `offered_load_factors` | `[0.5, 1.0, 1.5]` | Below / at / above bottleneck — the "above" case is what should produce genuine, observable, queue-overflow packet loss |
| `n_flows_values` | `[1]` | Phase 1's runner drives exactly one sender/receiver pair — this dimension stays fixed until `src/network/experiment.py` is extended, a limitation inherited unchanged from Phase 2 |
| `repetitions` | `1` | Pilot is about configuration diversity, not repeated-run stability |

= **12 experiments, ~4 minutes of experiment time** (12 × 20s duration) — small, resource-conscious, sequential (Phase 1/2's runner is already strictly sequential; no concurrent-execution support exists or was added).

**Verified this phase** (dry-run only, no network touched):
```
$ python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json --dry-run --sweep-id pilot-verify
parameter combos:    12
repetitions:         1
total experiments:   12
estimated duration:  4.0 min (240s)
[... 12 correctly-named experiment_ids printed ...]
```
This confirms the pilot config is syntactically and semantically valid and ready to run — **it has not been run for real.**

## 6. Full Dataset Generation

**NOT EXECUTED.** No pilot results exist to validate before deciding on a full sweep's scale, and the environment itself is unavailable (Section 9). Once the pilot has been run and passed the checks in Section 8, the natural next step (not taken here) would be widening `configs/pilot_sweep.json`'s value lists (e.g. a third bandwidth value, a third delay value) and re-running via the same `--sweep-config` mechanism with a fresh `--sweep-id`.

## 7. Dataset Statistics

**NOT AVAILABLE.** `scripts/dataset_report.py` (built this phase, Section 10) would produce this once real data exists; run against the current (empty) `data/raw/`, it correctly refuses:
```
$ python3 scripts/dataset_report.py
No CSV files found under .../data/raw. No real dataset exists yet -- nothing to report on.
Run scripts/generate_dataset.py against a real Mininet environment first.
```
Confirmed on this machine, exit code 1, nothing written to `reports/`.

## 8. Packet-Loss Distribution

**NOT AVAILABLE** — no real data exists. `ml.data_quality.target_summary()` (Section 10) computes exactly this (min/max/mean/median/std, zero-loss %, non-zero-loss %) and is unit-tested against fixture data (`tests/test_data_quality.py`), but has never been run against real experiment output.

## 9. Environment Verification (Step 1 — performed this phase)

Re-checked, using the project's own existing infrastructure rather than ad hoc commands:
```
$ python3 scripts/smoke_test.py
Tool availability: {'mn': False, 'iperf3': False, 'ping': True, 'tc': False, 'ovs-vsctl': False}
Missing tool(s): ['mn', 'iperf3', 'tc', 'ovs-vsctl'] -- falling back to DRY-RUN mode.
[... 6/6 dry-run parser/schema/target checks PASSED ...]
DRY-RUN: all parser/schema/target checks PASSED. Live Mininet behavior is still UNVERIFIED on this machine.
```
Also re-confirmed directly: `wsl --status` still exits with code 50, and `Microsoft-Windows-Subsystem-Linux` / `VirtualMachinePlatform` Windows features still report `InstallState = 2` (Disabled) via `Get-CimInstance Win32_OptionalFeature`. This is the same state documented across every prior environment check in this project (`ENVIRONMENT_SETUP.md`) — nothing has changed since Phase 0.

**Per the Phase 5 Data Rule, this stops the live-data portion here.** Everything from Section 6 onward in the original Phase 5 task list (full sweep, real training, real metrics, real feature importance, real plots, real risk distribution, real artifacts) could not be attempted without either fabricating results or running against nothing — both explicitly forbidden. What follows instead is what the brief asks for in the blocked case: proof the code is ready, dry-run support, and tests.

## 10. What Was Built and Verified This Phase (code-only, environment-independent)

- **`configs/pilot_sweep.json`** + `configs/README.md` — the documented pilot design (Section 5), verified via dry-run.
- **`src/ml/data_quality.py`** — `dataset_overview`, `feature_statistics`, `target_summary`, `network_condition_summary`, `detect_issues` (duplicate rows, duplicate `(experiment_id, interval_index)` pairs, missing intervals, zero-target-variance experiments, out-of-range loss values, `packets_received > packets_sent`), `build_data_quality_report`, `render_markdown`, `save_report`. Reuses `ml.dataset.load_dataset()` (Phase 4) unmodified — it reports on exactly the validated, leakage-checked data Phase 4 training would use, not a separately-read copy.
- **`scripts/dataset_report.py`** — thin CLI wrapper (Phase 5 Step 6's "script/report"). Refuses cleanly with no real data (verified, Section 7).
- **`tests/test_live_mininet.py`** — genuine (not mocked) Mininet integration tests, explicitly separated from the rest of the suite via `pytest.mark.skipif(not check_environment().all_available, ...)` (Phase 5 Step 14's explicit requirement). Currently **SKIPPED** on this machine — reported as skipped, never as passed, since they have not actually run.
- **`tests/test_data_quality.py`** — 11 new tests for the data-quality module, using the existing labeled `ml_fixtures.py` TEST FIXTURE plus small hand-crafted edge cases (a deliberately duplicated row, a deliberately removed middle interval, a deliberately zero-variance target) to prove `detect_issues()` actually catches what it claims to.

No existing Phase 1/2/4 module was rewritten. `network.experiment.run_experiment()`, `network.targets.add_next_interval_target()`, `ml.dataset.load_dataset()`, `ml.split`, `ml.train`, `ml.evaluate`, `ml.risk`, `ml.artifacts` are all reused completely unchanged — inspected this phase (Step 0), no bug found in any of them.

## 11. Validation Checks

**NOT RUN on real data** (none exists). The checks themselves (`network.validation.validate_rows`, `validate_no_leakage`, plus the new `ml.data_quality.detect_issues`) are unchanged/new-but-tested code, ready to run the moment real CSVs exist — `ml.dataset.load_dataset()` runs the first two automatically on every load, real or fixture.

## 12. Leakage Checks

**NOT RUN on real data.** The leakage-safety machinery itself (`add_next_interval_target`, `validate_no_leakage`, the group-aware splits in `ml/split.py`) was inspected this phase per Step 0 and found unchanged from Phase 4 — no modification was needed or made. Its correctness against adversarial fixture cases (wrong-row target, cross-experiment leakage, non-null end-of-run target) remains verified only against fixtures, as it was in Phase 2/4 — there is no real dataset yet for these checks to run against for real.

## 13. Train/Test Methodology

Unchanged from Phase 4 (`docs/PHASE_4_ML_MODELING.md` Section 4): `chronological_split` (default) or `random_group_split`, both operating on whole `experiment_id` groups with a zero-overlap guarantee. Not exercised against real data this phase.

## 14. Model Results

**NOT AVAILABLE.**
```
Model | MAE | RMSE | R²
```
No row of this table exists. `scripts/train_models.py` (Phase 4, unmodified) was not run against real data because none exists; running it against the current empty `data/raw/` reproduces the same honest refusal documented in `PHASE_4_ML_MODELING.md` Section 16.

## 15. Baseline Comparison

**NOT AVAILABLE** — depends on Section 14.

## 16. Feature Importance

**NOT AVAILABLE** — depends on Section 14. `ml.evaluate.random_forest_feature_importance()` and `permutation_feature_importance()` (Phase 4) are unmodified and ready to run on a real fitted model; none exists.

## 17. Limitations

- **The entire empirical portion of this phase did not happen.** This is the headline limitation, not a footnote — see Section 9.
- The pilot design (Section 5) is a plan, verified only syntactically (dry-run). Whether 12 experiments at these specific bandwidth/delay/load values actually produce a good mix of zero and non-zero loss intervals is an empirical question this phase could not answer.
- `n_flows_values` stays fixed at `[1]` — a real limitation inherited from Phase 1, not something Phase 5 attempted to lift.
- `tests/test_live_mininet.py` exists and is well-formed, but has literally never executed — its correctness is unverified beyond "it imports and collects without error."

## 18. Reproducibility Commands

For whoever runs this once the environment is fixed (not executed here):

```bash
# 1. Generate the pilot dataset (requires Mininet inside WSL2 Ubuntu, run as root)
sudo python3 scripts/generate_dataset.py --sweep-config configs/pilot_sweep.json --sweep-id pilot-001

# 2. Validate the pilot + inspect its quality
python3 scripts/dataset_report.py --input data/raw --basename pilot-001_quality

# 3. If the pilot's non-zero-loss percentage looks too low (dataset_report.py
#    prints a warning below 5%), widen configs/pilot_sweep.json's
#    offered_load_factors upward and re-run steps 1-2 before proceeding.

# 4. Generate the full dataset (once the pilot looks right) -- widen the
#    value lists in a copy of configs/pilot_sweep.json first
sudo python3 scripts/generate_dataset.py --sweep-config configs/full_sweep.json --sweep-id full-001

# 5. Validate the full dataset
python3 scripts/dataset_report.py --input data/raw --basename full-001_quality

# 6. Train and evaluate all models on the real dataset
python3 scripts/train_models.py --input data/raw

# 7. View reports
#    - reports/dataset_quality.md / .json          (data-quality report)
#    - reports/modeling/model_comparison.csv        (Model | MAE | RMSE | R2)
#    - reports/modeling/*_actual_vs_predicted.png
#    - reports/modeling/*_residuals.png
#    - reports/modeling/*_feature_importance.png / .csv
#    - models/*.joblib, models/*_metadata.json       (saved best-by-MAE model)

# Live Mininet integration tests (only meaningful once the environment works):
python3 -m pytest tests/test_live_mininet.py -v
```

---

*This document will be rewritten with real dataset statistics, real validation/leakage-check results, a real model comparison table, real feature importance, and real plots once WSL2 + Mininet are confirmed working and the commands in Section 18 have actually been run.*
