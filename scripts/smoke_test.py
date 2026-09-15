#!/usr/bin/env python3
"""Phase 1 smoke test -- proves the code works, at whatever level this
machine currently supports. Two distinct modes, never blended:

  LIVE mode   (requires mn/iperf3/tc/ping/ovs-vsctl on PATH): builds a
              real 2-host dumbbell, runs a short real experiment, and
              validates the resulting CSV. Must be run as root inside
              WSL2 Ubuntu: `sudo python3 scripts/smoke_test.py`.

  DRY-RUN mode (used automatically when the tools above are missing,
              e.g. on Windows without WSL2 set up yet): exercises the
              parsers/schema/target-shift logic against fixture text,
              with ZERO live network involved. This proves the code is
              correct; it does NOT prove Mininet itself works, and the
              output says so explicitly.

This script will never silently produce fabricated experiment results
labeled as if they were real.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network.collectors import (  # noqa: E402
    parse_iperf3_udp_json,
    parse_ping_rtts,
    parse_tc_qdisc_backlog,
    summarize_rtts,
)
from network.config import ExperimentConfig  # noqa: E402
from network.schema import CSV_COLUMNS  # noqa: E402
from network.targets import add_next_interval_target, drop_rows_without_target  # noqa: E402
from network.validation import check_environment, validate_rows  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def run_dry_run() -> int:
    print("=" * 70)
    print("DRY-RUN MODE -- no live network, no Mininet. Parser/schema/target")
    print("logic only. This does NOT prove Mininet/iperf3/tc actually work.")
    print("=" * 70)

    failures = 0

    def check(label: str, fn) -> None:
        nonlocal failures
        try:
            fn()
            print(f"[PASS] {label}")
        except Exception as exc:  # noqa: BLE001 -- smoke test wants to report, not crash
            failures += 1
            print(f"[FAIL] {label}: {exc}")

    def t_iperf3_udp_parse():
        raw = (FIXTURES / "iperf3_udp_sample.json").read_text()
        rows = parse_iperf3_udp_json(raw)
        assert len(rows) > 0, "expected at least one interval"
        assert "packet_loss_pct" in rows[0]

    def t_ping_parse():
        raw = (FIXTURES / "ping_sample.txt").read_text()
        rtts = parse_ping_rtts(raw)
        assert len(rtts) >= 1
        stats = summarize_rtts(rtts)
        assert stats["current_rtt_ms"] > 0

    def t_tc_parse():
        raw = (FIXTURES / "tc_qdisc_sample.txt").read_text()
        backlog = parse_tc_qdisc_backlog(raw)
        assert backlog >= 0

    def t_target_shift_no_leakage():
        rows = [
            {"experiment_id": "e1", "interval_index": 0, "packet_loss_pct": 1.0},
            {"experiment_id": "e1", "interval_index": 1, "packet_loss_pct": 2.0},
            {"experiment_id": "e2", "interval_index": 0, "packet_loss_pct": 99.0},
        ]
        shifted = add_next_interval_target(rows)
        e1_row0 = next(r for r in shifted if r["experiment_id"] == "e1" and r["interval_index"] == 0)
        assert e1_row0["target_next_packet_loss_pct"] == 2.0
        e1_row1 = next(r for r in shifted if r["experiment_id"] == "e1" and r["interval_index"] == 1)
        assert e1_row1["target_next_packet_loss_pct"] is None, "last row of a group must have no target"
        kept = drop_rows_without_target(shifted)
        assert all(r["experiment_id"] != "e1" or r["interval_index"] != 1 for r in kept)
        assert not any(
            r["experiment_id"] == "e1" and r["interval_index"] == 1 and r["target_next_packet_loss_pct"] == 99.0
            for r in shifted
        ), "target must never cross an experiment_id boundary"

    def t_config_validates():
        cfg = ExperimentConfig()
        assert cfg.n_left_hosts + cfg.n_right_hosts <= 4

    def t_validate_rows_catches_bad_data():
        bad_rows = [{
            "experiment_id": "e1", "interval_index": 0, "timestamp": 1.0,
            "packets_sent": 10, "packets_received": 999, "packet_loss_pct": 5.0,
        }]
        try:
            validate_rows(bad_rows, ["experiment_id", "interval_index", "timestamp"])
        except Exception:
            return  # expected: received > sent should be rejected
        raise AssertionError("validate_rows should have rejected packets_received > packets_sent")

    check("parse_iperf3_udp_json against fixture", t_iperf3_udp_parse)
    check("parse_ping_rtts + summarize_rtts against fixture", t_ping_parse)
    check("parse_tc_qdisc_backlog against fixture", t_tc_parse)
    check("add_next_interval_target is leakage-safe across experiment boundaries", t_target_shift_no_leakage)
    check("ExperimentConfig respects Phase 1 resource cap", t_config_validates)
    check("validate_rows rejects inconsistent packet counts", t_validate_rows_catches_bad_data)

    print()
    print(f"CSV schema ({len(CSV_COLUMNS)} columns): {', '.join(CSV_COLUMNS)}")
    print()
    if failures:
        print(f"DRY-RUN: {failures} check(s) FAILED.")
        return 1
    print("DRY-RUN: all parser/schema/target checks PASSED. "
          "Live Mininet behavior is still UNVERIFIED on this machine.")
    return 0


def run_live() -> int:
    from network.config import ExperimentConfig as Cfg
    from network.experiment import run_experiment

    print("=" * 70)
    print("LIVE MODE -- required tools detected, running a real short experiment.")
    print("=" * 70)

    config = Cfg(
        n_left_hosts=1, n_right_hosts=1,
        bottleneck_bw_mbps=2.0, bottleneck_delay_ms=10.0,
        duration_s=10.0, sample_interval_s=2.0,
        offered_load_mbps=4.0,
    )
    output_path = run_experiment(config, REPO_ROOT / "data" / "raw")
    print(f"[PASS] Live experiment completed, wrote {output_path}")
    return 0


def main() -> int:
    env = check_environment()
    print(f"Tool availability: {env.available}")
    if env.all_available:
        return run_live()
    print(f"Missing tool(s): {env.missing} -- falling back to DRY-RUN mode.\n")
    return run_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
