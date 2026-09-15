# PHASE_1_NETWORK_EXPERIMENT.md
## Networking Experiment + Data-Generation Foundation

Status: **Code complete, unit-tested (parser/schema/target logic). Live Mininet execution NOT YET verified** -- this machine's WSL2/Mininet environment is still being validated (see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md), currently blocked on the Windows WSL feature not showing as enabled). Nothing in this document claims a live network run succeeded unless it actually did.

---

## 1. Topology

```
    h1 --\                                  /-- h3
          s1 ====[ shaped bottleneck ]==== s2
    h2 --/                                  \-- h4
```

- Left-side hosts (`h1`, `h2`, ... up to `n_left_hosts`) connect to switch `s1` over fast, unshaped 1000 Mbps links.
- Right-side hosts (`h{n_left+1}`, ... up to `n_left_hosts + n_right_hosts`) connect to switch `s2` the same way.
- The **only** shaped link is `s1<->s2` -- bandwidth, delay, and queue size are applied there via Mininet's `TCLink` (which wraps Linux `tc`/`netem`/HTB). This is what makes it a genuine bottleneck, matching the standard dumbbell topology used in networking courses to demonstrate congestion (PROJECT_PLAN.md Section 6).
- Phase 1 default: `n_left_hosts=2, n_right_hosts=2` (4 hosts total), but the code (`src/network/topology.py`) builds this dynamically from `ExperimentConfig`, so it's reusable rather than hardcoded. `ExperimentConfig` hard-caps total hosts at 4 to respect this machine's RAM constraint (documented in ENVIRONMENT_SETUP.md) -- raising that cap is a deliberate code change, not an accident.

Traffic in Phase 1 flows from `h1` (sender) to the first right-side host (`h3` by default) via `iperf3`.

## 2. Networking Tools Used

| Tool | Role |
|---|---|
| Mininet (Python API) | Builds the topology, real Linux network namespaces + veth pairs |
| `iperf3` (`-J` JSON output) | Traffic generation; reports throughput, and for UDP: packets sent/lost/loss% per interval |
| `ping` | Independent RTT sampling (protocol-agnostic, doesn't depend on the iperf3 flow) |
| `tc -s qdisc show` | Reads bottleneck queue backlog (packets) directly from the kernel |
| Open vSwitch | Mininet's default switch implementation (pulled in by the `mininet` apt package) |

## 3. Why UDP is the Phase 1 default traffic type

iperf3's TCP mode reports **bytes and retransmits only** -- it does not report packet counts or a direct loss percentage, because TCP's own retransmission logic obscures the raw drop rate. iperf3's UDP mode, by contrast, reports `packets`, `lost_packets`, and `lost_percent` directly per interval, which is exactly the clean, current-interval `packet_loss_pct` measurement the whole prediction pipeline depends on (PROJECT_PLAN.md Section 10-12). `traffic_type='tcp'` is still implemented and selectable (`--traffic-type tcp`), but `retransmissions` is the only loss-related signal available in that mode -- `packets_sent`/`packets_received`/`packet_loss_pct` are left as `None` rather than estimated, so the code never invents numbers iperf3 didn't actually report.

## 4. Experiment Parameters (`src/network/config.py::ExperimentConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `n_left_hosts` / `n_right_hosts` | 2 / 2 | Topology size (capped at 4 total) |
| `bottleneck_bw_mbps` | 5.0 | Configured bottleneck bandwidth |
| `bottleneck_delay_ms` | 20.0 | Configured one-way delay on the bottleneck link |
| `bottleneck_queue_pkts` | 20 | Configured max queue depth (`tc` qdisc limit) on the bottleneck |
| `traffic_type` | `udp` | `udp` or `tcp` |
| `offered_load_mbps` | 8.0 | iperf3 target send rate -- deliberately set ABOVE `bottleneck_bw_mbps` so loss emerges from genuine queue overflow, not an injected loss knob (see Section 6) |
| `active_connections` | 1 | Number of concurrent iperf3 flows (recorded as a feature; Phase 1's runner drives a single flow) |
| `sample_interval_s` | 2.0 | How often measurements are sampled during the run |
| `duration_s` | 20.0 | Total experiment duration -- short by design for Phase 1 |
| `target_horizon_intervals` | 1 | Δ: how many intervals ahead the target looks (see Section 7) |

All overridable via `scripts/run_experiment.py` CLI flags.

## 5. Measurement Methodology

For each `sample_interval_s` tick during the run:
1. `ping -c 3 -W 1 <dst>` is run from the source host; its 3 RTT samples are averaged (`current_rtt_ms`) and their population std-dev is used as a protocol-agnostic jitter proxy (`current_jitter_ms`).
2. `tc -s qdisc show dev <bottleneck-iface>` is read on the switch adjacent to the bottleneck link; the `backlog Xb Yp` field's packet count becomes `queue_length` -- a point-in-time snapshot, not an interval average.
3. Meanwhile, a single `iperf3 -J -i <sample_interval_s>` client process runs for the whole experiment duration in the background; after it finishes, its JSON `intervals` array is parsed into per-interval `throughput_mbps`, `packets_sent`, `packets_received`, `packet_loss_pct` (UDP) or `retransmissions` (TCP).
4. The ping/tc samples and the iperf3 interval reports are merged **by interval index** (`_merge_rows` in `src/network/experiment.py`) -- both were collected over the same wall-clock duration at the same interval size, so index alignment is a reasonable approximation for this single-flow setup. A future phase with multiple concurrent flows sampled independently would need timestamp-based alignment instead of index alignment; that's flagged as a known limitation (Section 9), not silently assumed away.

## 6. Why Loss Emerges from Congestion, Not an Injected Knob

`offered_load_mbps` (default 8) is set above `bottleneck_bw_mbps` (default 5) on purpose. The bottleneck's queue (`bottleneck_queue_pkts`) is finite, so once the offered load exceeds capacity, packets genuinely queue and get dropped by the kernel's own qdisc -- this is real tail-drop behavior, not a `netem loss%` parameter dictating the outcome. This directly implements the design decision from PROJECT_PLAN.md Section 7: an explicit injected-loss knob used as a feature would let a model trivially invert our own parameter instead of learning real network behavior. `target_next_packet_loss_pct` in this dataset will therefore reflect actual queueing dynamics driven by `bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts`, and `offered_load_mbps` interacting -- not a single dial.

## 7. Target Definition (Leakage Prevention)

At prediction cutoff `T` (the end of interval `i`):
- **Features** = every measurement whose collection window ends at or before `T` (interval `i` and earlier). This includes `packet_loss_pct` for interval `i` itself -- safe, because it describes the past, not the future.
- **Target** (`target_next_packet_loss_pct`) = `packet_loss_pct` measured during interval `i + target_horizon_intervals` (default: the very next interval, `Δ = sample_interval_s` = 2 seconds by default).

**Structural leakage prevention, not just a convention**: the live collector (`src/network/experiment.py`) never writes `target_next_packet_loss_pct` itself -- it only ever writes `packet_loss_pct` for the interval it just measured. The target column is added afterwards by a single, separate, pure function: `add_next_interval_target()` in `src/network/targets.py`. That function:
- Groups rows by `experiment_id` before shifting, so interval `i`'s target is **never** taken from a different experiment run.
- Orders rows by `interval_index` before shifting (not by input order), so out-of-order data still shifts correctly.
- Leaves `target_next_packet_loss_pct = None` for the last `horizon` row(s) of each experiment (no future data exists) and for any row whose `i + horizon` interval is missing (e.g. a dropped sample) -- it never fabricates a target from the nearest available row.
- Is covered by `tests/test_targets.py`, which specifically tries to break the experiment-boundary and out-of-order cases (see Section 10).

`drop_rows_without_target()` removes the unlabeled trailing rows before any future ML training step uses the data.

## 8. CSV Schema

Full column-by-column definitions (name, unit, role, description, leakage-safety rationale) live in code as the single source of truth: `src/network/schema.py::SCHEMA`. Summary:

| Column | Role | Leakage-safe because... |
|---|---|---|
| `experiment_id`, `run_id`, `interval_index`, `timestamp`, `source_host`, `destination_host` | identifier | Structural metadata, not a measurement |
| `bottleneck_bw_mbps`, `bottleneck_delay_ms`, `bottleneck_queue_pkts`, `traffic_type`, `active_connections` | config | Fixed before the run starts -- cannot encode future information by construction |
| `current_rtt_ms`, `current_jitter_ms`, `throughput_mbps`, `bandwidth_utilization_pct`, `packet_rate_pps`, `queue_length`, `retransmissions`, `packets_sent`, `packets_received`, `packet_loss_pct` | feature | Each describes interval `i` only, strictly before target interval `i + horizon` |
| `target_next_packet_loss_pct` | target | Added in a separate post-processing step (Section 7), never written by the live collector |

`run_id` is currently identical to `experiment_id` (one script execution = one run); kept as a distinct column for forward compatibility with future multi-run parameter sweeps, per the schema module's own docstring.

## 9. Known Limitations

- **Index-based (not timestamp-based) alignment** between ping/tc samples and iperf3 interval reports -- fine for Phase 1's single-flow setup, would need revisiting for concurrent multi-flow sampling.
- **`queue_length` reads the first qdisc match** in `tc -s qdisc show` output (via `parse_tc_qdisc_backlog`'s regex), which on a link with chained HTB+netem qdiscs may report the outer qdisc's backlog rather than specifically netem's. Documented here rather than silently assumed precise; worth revisiting once real `tc -s qdisc` output from this machine's actual Mininet install can be inspected.
- **`retransmissions` is only meaningful for `traffic_type='tcp'`** -- always 0 for UDP runs (UDP has no retransmission concept).
- **Single sender/receiver pair per run** in Phase 1's `run_experiment()`, even though the topology supports up to 4 hosts -- `active_connections` is recorded as a configured value but Phase 1 does not yet drive multiple concurrent flows. That's an intentional scope boundary for this phase, not an oversight (see PROJECT_PLAN.md Phase 2/3 for the sweep-driven multi-flow expansion).
- **Live Mininet execution is unverified on this development machine** as of this writing -- see Section 11.

## 10. How to Run

**Unit tests (parser/schema/target logic -- runs anywhere with Python 3, including Windows, no Mininet needed):**
```bash
pip install -r requirements.txt
pytest tests/ -v
```
Verified on this machine: `py -3 -m pytest tests/ -v` → **34 passed**.

**Smoke test (auto-detects environment):**
```bash
python3 scripts/smoke_test.py
```
- If `mn`/`iperf3`/`tc`/`ping`/`ovs-vsctl` are all on PATH: runs a real short 2-host experiment and reports PASS/FAIL per check.
- Otherwise: falls back to DRY-RUN mode, which exercises the same parsing/target logic against fixture text with zero live network involved, and says so explicitly in its output.

**Live experiment (requires WSL2 Ubuntu + Mininet installed, run as root):**
```bash
sudo python3 scripts/run_experiment.py --duration-s 20 --sample-interval-s 2
```
Writes `data/raw/<experiment_id>.csv`. Refuses to run (raises `RuntimeError`, writes nothing) if any required tool is missing -- see Section 11.

## 11. What Was Actually Executed On This Machine (Phase 1)

| Check | Result |
|---|---|
| `pytest tests/ -v` (Windows, Python 3.14.0 via `py -3`) | **34/34 PASSED** -- parsers, schema, target-shift leakage safety, config validation, row validation all verified against fixture data |
| `scripts/smoke_test.py` | Ran in **DRY-RUN mode** (Windows lacks `mn`, `iperf3`, `tc`, `ovs-vsctl` -- only `ping` is present natively). All 6 dry-run checks **PASSED**. Explicitly logged: "Live Mininet behavior is still UNVERIFIED on this machine." |
| `scripts/run_experiment.py --duration-s 4 --sample-interval-s 2` | Correctly **refused to run** with `RuntimeError: Cannot run a live network experiment -- missing required tool(s): mn, iperf3, tc, ovs-vsctl ... Refusing to fabricate results -- no CSV will be written.` Confirmed `data/raw/` contains no CSV afterward. |
| Live Mininet dumbbell topology, real traffic, real queueing/loss | **NOT YET RUN** -- blocked on WSL2/Mininet installation, which is still being validated separately (see ENVIRONMENT_SETUP.md) |

No fabricated dataset exists anywhere in this repository. `data/raw/` contains only a `.gitkeep` placeholder.

---

*This document will be updated with real `sudo python3 scripts/run_experiment.py` output, an actual generated CSV sample, and live smoke-test PASS/FAIL results once WSL2 + Mininet are confirmed working on this machine.*
