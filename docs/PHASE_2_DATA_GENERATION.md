# PHASE_2_DATA_GENERATION.md
## Reproducible Data-Generation Pipeline

Status legend used throughout this document:
- **IMPLEMENTED** — code exists, is unit-tested, and its non-network behavior (dry-run, validation, manifest) has been executed and verified on this machine.
- **NOT YET EXECUTED** — code exists but requires Mininet/iperf3/tc/Open vSwitch, which are still unavailable on this development machine (see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)). No real network data has been generated. Nothing in this document should be read as claiming otherwise.

---

## 1. Architecture

Phase 2 adds an orchestration layer *around* Phase 1's `run_experiment()` — it does not modify Phase 1's topology, collectors, config validation, or leakage-safe target logic in any way (confirmed by inspection; see Section 6).

```
SweepConfig (src/network/sweep.py)
        │  expand_configs()
        ▼
list[ExperimentConfig]  ── reuses Phase 1's ExperimentConfig unchanged
        │
        ▼
run_sweep() (src/network/generate.py)
        │  calls run_experiment_fn(config, output_dir) per config
        │  -- defaults to Phase 1's real run_experiment, injectable for tests
        ▼
data/raw/<experiment_id>.csv  (one per experiment, Phase 1's own leakage-safe format)
        │
        ▼
build_manifest() (src/network/manifest.py)
        ▼
data/manifests/<sweep_id>_manifest.json
```

**IMPLEMENTED.**

## 2. Parameter Sweep

`src/network/sweep.py::SweepConfig` defines four swept dimensions plus fixed run parameters:

| Dimension | Default values | Rationale |
|---|---|---|
| `bottleneck_bw_mbps_values` | `[1.0, 2.0]` | Small, resource-appropriate bandwidths |
| `bottleneck_delay_ms_values` | `[0.0, 20.0]` | No delay vs. a modest WAN-like delay |
| `offered_load_factors` | `[0.7, 1.3]` | **Relative to each bandwidth value**, not absolute — 0.7 covers "below bottleneck" and 1.3 covers "above bottleneck" (genuine congestion) for every bandwidth tested, since "near/above bottleneck" only makes sense relative to whatever bandwidth that combination uses |
| `n_flows_values` | `[1]` | See Section 2.1 — intentionally restricted |

Default sweep size: 2 × 2 × 2 × 1 = **8 parameter combinations**. With `repetitions=1` (default), that's 8 experiments; the original request's example of `--repetitions 3` would make it 24. At the default `duration_s=20s` each, that's ~2.7 minutes of experiment time for the default 8-combination sweep — deliberately small per the "start small" instruction and this machine's RAM constraint (documented in ENVIRONMENT_SETUP.md).

All of this is overridable: via CLI flags (`--repetitions`) or a JSON file (`--sweep-config my_sweep.json`) that overrides any `SweepConfig` field, including widening the value lists once a live run has been validated.

**IMPLEMENTED.**

### 2.1 Why `n_flows_values` is restricted to `[1]`

Phase 1's `run_experiment()` drives exactly one sender/receiver pair (`h1` → first right-side host) — this is documented as a known limitation in `PHASE_1_NETWORK_EXPERIMENT.md` Section 9, not something Phase 2 is allowed to silently paper over. `SweepConfig` therefore raises `NotImplementedError` immediately (at construction time) if `n_flows_values` contains anything other than `1`, with a message pointing at exactly what would need to change (`src/network/experiment.py`) to support it. This is a deliberate, loud failure — not a silent skip — consistent with the instruction not to redesign Phase 1's architecture in this phase. `active_connections` is still recorded as metadata per experiment (reusing the existing schema column), it just cannot yet vary within a single sweep.

## 3. Dataset Generator (`scripts/generate_dataset.py`)

Thin CLI wrapper around `src/network/generate.py::run_sweep()` / `run_sweep_and_write_manifest()`, following the same pattern as Phase 1's `scripts/run_experiment.py`:

- Loads a `SweepConfig` (defaults, or `--sweep-config path.json`, with `--repetitions`/`--sweep-id` as convenience overrides).
- `--dry-run`: prints the full plan (combination count, repetitions, total experiments, estimated duration, output location, every planned `experiment_id`) and **exits without touching the network or the filesystem at all** — verified in Section 8.
- Live mode: calls `require_environment()` **once, up front**, before running experiment 1 of N — so a missing-tool failure is immediate, not partway through a sweep. `run_experiment()` (Phase 1) also checks this per-experiment as a second line of defense.
- Every experiment's outcome (success + row count + CSV path, or failure + error message) is recorded in `data/manifests/<sweep_id>_manifest.json`. A failed experiment does **not** silently vanish — it's printed immediately (`[FAILED] <experiment_id> -- <error>`) and shown in the manifest; the sweep continues to the next combination by default (see `run_sweep(..., stop_on_first_failure=False)`), and the script's exit code is non-zero if anything failed. This satisfies "don't silently skip" by making failure visible and recorded rather than by aborting the whole sweep on the first error — a deliberate design choice, not an oversight.
- Row counts in the manifest come from **actually reading back the CSV** Phase 1's `run_experiment()` wrote (`_count_rows()`), never from an assumed/expected count.

**IMPLEMENTED** (dry-run and the live-mode refusal path have been executed and verified; the actual experiment-running loop has not, since it requires Mininet).

## 4. Experiment Identification

Every experiment gets a deterministic, human-readable `experiment_id`:

```
<sweep_id>__bw<bandwidth>__delay<delay>__load<factor>__flows<n>__rep<repetition>
```

e.g. `demo-sweep__bw1__delay20__load1.3__flows1__rep0`. Built by `sweep.build_experiment_id()`. `sweep_id` defaults to `sweep-<unix timestamp>` and is itself overridable via `--sweep-id`. The **schema is unchanged from Phase 1** — `experiment_id`, `run_id` (still equal to `experiment_id`, per Phase 1's documented forward-compatibility note), `timestamp`, and every config/feature/target column are exactly the ones in `src/network/schema.py`; Phase 2 does not add or duplicate any identification columns.

**IMPLEMENTED.**

## 5. Repetitions

`SweepConfig.repetitions` (default 1, CLI-overridable via `--repetitions N`) makes `expand_configs()` emit `repetitions` separate `ExperimentConfig` instances per parameter combination, each with its own `rep{N}`-suffixed `experiment_id` (`build_experiment_id`'s last argument) — so every repetition has its own distinct identity and its own CSV file, never overwriting a sibling repetition's output. Expansion order is deterministic (`itertools.product`, repetitions innermost), so the same `SweepConfig` always produces the same ordered list of `experiment_id`s.

**IMPLEMENTED** — verified with `--dry-run --repetitions 2`, which correctly doubled the experiment count and produced 16 distinct, correctly-suffixed IDs (see Section 8 log).

## 6. Target Generation — Inspected, Not Modified

Per the explicit instruction to inspect before changing anything and to stop and report rather than silently altering the leakage-safe design: **`src/network/targets.py` was inspected and reused completely unchanged.** No bug was found. Specifically verified during inspection:
- `add_next_interval_target()` groups by `experiment_id` and orders by `interval_index` before shifting — a sweep's many `experiment_id`s (one per parameter combination × repetition) is exactly the grouping structure this function was already built for; nothing about Phase 2's sweep changes how it needs to behave.
- Each per-experiment CSV Phase 2 generates is written by the *same* `run_experiment()` call Phase 1 already validated, so the target column in every sweep-generated CSV is produced by the identical, previously-tested code path.

**No changes made. IMPLEMENTED (reused).**

## 7. Data Validation

Three layers, none duplicating the others:

1. **Phase 1's `validate_rows()`** (unchanged) — still runs inside `run_experiment()` for every individual experiment: required columns present, `interval_index` contiguous from 0, timestamps non-decreasing, `packets_received <= packets_sent`, `packet_loss_pct` in `[0, 100]`.
2. **New: `validate_no_leakage()`** (`src/network/validation.py`, added this phase) — an *independent* check of a dataset's target column against its source column, deliberately not trusting that `add_next_interval_target()` was used correctly upstream:
   - Every non-null target must equal `packet_loss_pct` measured exactly `horizon` intervals later, **within the same `experiment_id`**.
   - The last `horizon` row(s) of each experiment must have a null target (no fabricated end-of-run values).
   - An explicit cross-experiment guard: a target can never trace back to a *different* `experiment_id`'s value at the matching offset.
   - Covered by `tests/test_validate_no_leakage.py`, which manufactures exactly the corrupted datasets that would slip through if this check had a hole (wrong-row target, cross-experiment leakage, non-null end-of-run target, null target when future data exists).
3. **`require_environment()`** (unchanged) — the environment-level check described in Section 3.

Deliberately **not** added: per-value statistical checks (outlier detection, distribution checks) — out of scope for "don't over-engineer validation," and premature before any real data exists to characterize.

**IMPLEMENTED and unit-tested** (32 new tests across `test_sweep.py`, `test_manifest.py`, `test_generate.py`, `test_validate_no_leakage.py`, plus 2 added to `test_validation.py`).

## 8. Dataset Manifest

`src/network/manifest.py::build_manifest()` records, per sweep run:

- `sweep_id`, `generated_at` / `started_at` / `finished_at`, `output_dir`, `dry_run` flag
- The full `sweep_config` (every `SweepConfig` field, via `dataclasses.asdict`)
- `n_experiments_planned` / `n_experiments_succeeded` / `n_experiments_failed`
- `total_rows_written` (summed only over **successful** experiments — never counts rows for a failed one, verified by `test_build_manifest_never_counts_rows_for_failed_experiments`)
- `feature_columns` / `target_column` (from `src/network/schema.py`, so the manifest can never drift from the actual CSV schema)
- `environment`: Python version, platform string, and a best-effort `tool --version` capture for `mn`, `iperf3`, `tc`, `ovs-vsctl`, `ping` (each individually `None` if the tool isn't present — verified this returns cleanly on Windows, where none of the Linux networking tools exist, without raising)
- `experiments`: one entry per experiment attempted, with `status`, `csv_path`, `row_count`, and `error` (for failures)

Written to `data/manifests/<sweep_id>_manifest.json`. Both `data/raw/*.csv` and `data/manifests/*.json` are git-ignored (generated output, not source) — same convention Phase 1 already established for `data/raw/`.

**IMPLEMENTED and unit-tested** (`test_manifest.py`, including a real round-trip through `write_manifest()` → `json.loads()` on disk via `tmp_path`).

## 9. Testing

32 new tests, **zero requiring Mininet**:

| File | What it covers |
|---|---|
| `tests/test_sweep.py` | Combination/repetition counts, deterministic `experiment_id` generation, relative-load-factor math, reuse of `ExperimentConfig`'s own validation (two separate tests: one dimension `SweepConfig` itself rejects, one it doesn't but `ExperimentConfig` does), the `n_flows_values != [1]` refusal |
| `tests/test_manifest.py` | Success/failure counting, row-count accuracy, environment snapshot never raising even with all tools missing, JSON round-trip |
| `tests/test_generate.py` | Orchestration via an **injected fake** `run_experiment_fn` (never touches Mininet or writes anything claiming to be real network data) — all-succeed case, a failure that doesn't abort the rest of the sweep, a total-failure case that doesn't fabricate row counts, and proof that `describe_sweep()` (the dry-run helper) creates no directory even when asked about a nonexistent output path |
| `tests/test_validate_no_leakage.py` | Deliberately corrupts otherwise-valid shifted datasets in exactly the ways a leakage bug would, to prove the new check actually catches them |
| `tests/test_validation.py` (2 new tests added) | `require_environment()` actually raising with a clear message when tools are missing, and passing silently when its (empty) tool list is satisfied |

Executed: `py -3 -m pytest tests/ -v` → **66 passed, 0 failed** (34 from Phase 1 + 32 new). The existing 34 were not modified and did not regress.

**IMPLEMENTED and executed.**

## 10. Dry-Run Usage

```bash
python3 scripts/generate_dataset.py --dry-run
python3 scripts/generate_dataset.py --dry-run --repetitions 2 --sweep-id my-sweep
```

Prints the full plan and **exits 0 without creating any file or directory** — verified on this machine: ran both commands above, confirmed `data/raw/` and `data/manifests/` contained only their `.gitkeep` placeholders before and after.

**IMPLEMENTED and executed.**

## 11. Live Execution Usage (once Mininet is available)

```bash
sudo python3 scripts/generate_dataset.py --repetitions 2
sudo python3 scripts/generate_dataset.py --sweep-config custom_sweep.json
```

Currently, running this on this development machine fails immediately and clearly:

```
RuntimeError: Cannot run a live network experiment -- missing required tool(s):
mn, iperf3, tc, ovs-vsctl. These must be installed inside a Linux environment
(WSL2 Ubuntu); see docs/ENVIRONMENT_SETUP.md. Refusing to fabricate results --
no CSV will be written.
```
— verified on this machine: `data/raw/` and `data/manifests/` remained empty afterward.

**Code IMPLEMENTED; live execution NOT YET EXECUTED** (blocked on the same WSL2/Mininet environment dependency as Phase 1 — no new blocker introduced by Phase 2).

## 12. Current Environment Limitation

Unchanged from Phase 1: WSL2's `Microsoft-Windows-Subsystem-Linux` / `VirtualMachinePlatform` Windows features are not yet confirmed enabled on this machine, so `mn`, `iperf3`, `tc`, and `ovs-vsctl` are unavailable. Phase 2's code is fully written and tested against that reality — the dataset generator's dry-run and refusal paths work correctly *because* the environment is unavailable, and the exact same command becomes a real sweep the moment the environment is fixed, with no code changes required.

---

*This document will be updated with real manifest contents, actual row counts, and live PASS/FAIL sweep results once WSL2 + Mininet are confirmed working on this machine.*
