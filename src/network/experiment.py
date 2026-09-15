"""Phase 1 experiment orchestration: build topology, generate traffic,
sample measurements, write a leakage-safe CSV.

Requires Mininet + iperf3 + tc + ping inside a Linux (WSL2 Ubuntu)
environment. On Windows (or anywhere Mininet is unavailable) importing
this module still works, but calling run_experiment() raises
MininetUnavailableError via validation.require_environment() --
by design, this code never fabricates measurements when the real
environment isn't available.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from network.collectors import (
    parse_iperf3_tcp_json,
    parse_iperf3_udp_json,
    parse_ping_rtts,
    parse_tc_qdisc_backlog,
    summarize_rtts,
)
from network.config import ExperimentConfig
from network.schema import CSV_COLUMNS
from network.targets import add_next_interval_target
from network.topology import build_dumbbell_net
from network.validation import require_environment, validate_rows


def _bottleneck_interface(net, left_switch_name: str = "s1", right_switch_name: str = "s2") -> str:
    """Find the interface on the left switch that faces the right switch.

    Uses Mininet's own Node.connectionsTo() instead of guessing an
    interface-naming convention, so this stays correct even if link
    creation order in topology.py changes.
    """
    s1 = net.get(left_switch_name)
    s2 = net.get(right_switch_name)
    connections = s1.connectionsTo(s2)
    if not connections:
        raise RuntimeError(f"No direct link found between {left_switch_name} and {right_switch_name}")
    intf1, _intf2 = connections[0]
    return intf1.name


def run_experiment(config: ExperimentConfig, output_dir: Path) -> Path:
    """Run one short experiment and write its CSV. Returns the CSV path.

    Raises RuntimeError (via require_environment) instead of running if
    Mininet/iperf3/tc/ping/ovs-vsctl aren't available -- no fake data.
    """
    require_environment()

    net = build_dumbbell_net(config)
    try:
        h_src = net.get("h1")
        dst_name = f"h{config.n_left_hosts + 1}"
        h_dst = net.get(dst_name)
        bottleneck_iface = _bottleneck_interface(net)

        h_dst.cmd("iperf3 -s -D")  # daemonized server, stays up for the client run

        proto_flag = "-u" if config.traffic_type == "udp" else ""
        client_cmd = (
            f"iperf3 -c {h_dst.IP()} {proto_flag} -b {config.offered_load_mbps}M "
            f"-t {int(config.duration_s)} -i {config.sample_interval_s} -J"
        )
        client_proc = h_src.popen(client_cmd, shell=True)

        n_intervals = int(config.duration_s // config.sample_interval_s)
        raw_samples: list[dict[str, Any]] = []
        interval_start = time.time()
        for interval_index in range(n_intervals):
            time.sleep(config.sample_interval_s)
            ping_out = h_src.cmd(f"ping -c 3 -W 1 {h_dst.IP()}")
            rtt_stats = summarize_rtts(parse_ping_rtts(ping_out))
            tc_out = net.get("s1").cmd(f"tc -s qdisc show dev {bottleneck_iface}")
            queue_length = parse_tc_qdisc_backlog(tc_out)
            raw_samples.append({
                "interval_index": interval_index,
                "timestamp": time.time(),
                "current_rtt_ms": rtt_stats["current_rtt_ms"],
                "current_jitter_ms": rtt_stats["current_jitter_ms"],
                "queue_length": queue_length,
            })

        stdout, stderr = client_proc.communicate(timeout=30)
        if client_proc.returncode != 0:
            raise RuntimeError(f"iperf3 client exited with code {client_proc.returncode}: {stderr.decode(errors='replace')}")

        raw_json = stdout.decode(errors="replace")
        if config.traffic_type == "udp":
            iperf_intervals = parse_iperf3_udp_json(raw_json)
        else:
            iperf_intervals = parse_iperf3_tcp_json(raw_json)

        rows = _merge_rows(config, raw_samples, iperf_intervals, h_src.name, h_dst.name)
        validate_rows(rows, CSV_COLUMNS[:-1])  # target column is added after this, not part of raw collection

        rows_with_target = add_next_interval_target(rows, horizon=config.target_horizon_intervals)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.experiment_id}.csv"
        _write_csv(rows_with_target, output_path)
        return output_path
    finally:
        net.stop()


def _merge_rows(
    config: ExperimentConfig,
    raw_samples: list[dict[str, Any]],
    iperf_intervals: list[dict[str, Any]],
    source_host: str,
    destination_host: str,
) -> list[dict[str, Any]]:
    """Align ping/tc samples with iperf3's own interval reports by index.

    Both were collected over the same wall-clock duration at the same
    sample_interval_s, so index alignment is a reasonable approximation
    for this controlled, single-flow Phase 1 setup. A future phase with
    multiple concurrent flows would need timestamp-based alignment instead.
    """
    if len(raw_samples) != len(iperf_intervals):
        n = min(len(raw_samples), len(iperf_intervals))
        raw_samples, iperf_intervals = raw_samples[:n], iperf_intervals[:n]

    rows = []
    for raw, iperf_row in zip(raw_samples, iperf_intervals):
        duration = max(iperf_row["end_s"] - iperf_row["start_s"], 1e-9)
        packets_sent = iperf_row.get("packets_sent")
        row = {
            "experiment_id": config.experiment_id,
            "run_id": config.experiment_id,
            "interval_index": raw["interval_index"],
            "timestamp": raw["timestamp"],
            "source_host": source_host,
            "destination_host": destination_host,
            "bottleneck_bw_mbps": config.bottleneck_bw_mbps,
            "bottleneck_delay_ms": config.bottleneck_delay_ms,
            "bottleneck_queue_pkts": config.bottleneck_queue_pkts,
            "traffic_type": config.traffic_type,
            "active_connections": config.active_connections,
            "current_rtt_ms": raw["current_rtt_ms"],
            "current_jitter_ms": raw["current_jitter_ms"],
            "throughput_mbps": iperf_row["throughput_mbps"],
            "bandwidth_utilization_pct": iperf_row["throughput_mbps"] / config.bottleneck_bw_mbps * 100.0,
            "packet_rate_pps": (packets_sent / duration) if packets_sent is not None else None,
            "queue_length": raw["queue_length"],
            "retransmissions": iperf_row.get("retransmits", 0),
            "packets_sent": packets_sent,
            "packets_received": iperf_row.get("packets_received"),
            "packet_loss_pct": iperf_row.get("packet_loss_pct"),
        }
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
