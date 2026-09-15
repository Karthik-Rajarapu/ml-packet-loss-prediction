"""Canonical schema for network-experiment measurement rows.

This module is the single source of truth for what a row of collected
data means. Both the CSV writer (experiment.py) and the target-shift
step (targets.py) import CSV_COLUMNS from here so the on-disk format
never drifts out of sync with the documentation in
docs/PHASE_1_NETWORK_EXPERIMENT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ColumnSpec:
    name: str
    unit: str
    role: str  # "feature", "target", "config", "identifier"
    description: str


# Order here is the exact CSV column order written to disk.
SCHEMA: list[ColumnSpec] = [
    ColumnSpec("experiment_id", "string", "identifier",
               "Unique id for one experiment run (configuration + start time). Grouping key for leakage-safe train/test splitting."),
    ColumnSpec("run_id", "string", "identifier",
               "Currently identical to experiment_id (one script execution = one run). Kept as a separate column for forward compatibility with multi-run sweeps."),
    ColumnSpec("interval_index", "int", "identifier",
               "0-based position of this row within its experiment_id's time series."),
    ColumnSpec("timestamp", "unix seconds (float)", "identifier",
               "Wall-clock time at the END of this sampling interval."),
    ColumnSpec("source_host", "string", "identifier", "Mininet host name of the traffic sender, e.g. 'h1'."),
    ColumnSpec("destination_host", "string", "identifier", "Mininet host name of the traffic receiver, e.g. 'h3'."),

    # Configured conditions: known BEFORE the run starts, not measurements.
    # Safe as features by construction -- they cannot leak future information
    # because they are fixed for the entire experiment.
    ColumnSpec("bottleneck_bw_mbps", "Mbps", "config", "Configured bottleneck link bandwidth for this experiment."),
    ColumnSpec("bottleneck_delay_ms", "ms", "config", "Configured one-way delay added to the bottleneck link."),
    ColumnSpec("bottleneck_queue_pkts", "packets", "config", "Configured max queue size (tc qdisc limit) on the bottleneck link."),
    ColumnSpec("traffic_type", "string", "config", "'udp' or 'tcp'. Phase 1 defaults to UDP -- see docs for why."),
    ColumnSpec("active_connections", "int", "config", "Number of concurrent iperf3 flows configured for this run."),

    # Measured, current-interval features. Safe as inputs: each describes
    # only interval [t-1, t], strictly before the target interval [t, t+1].
    ColumnSpec("current_rtt_ms", "ms", "feature", "Mean RTT from ping samples taken during this interval."),
    ColumnSpec("current_jitter_ms", "ms", "feature", "Std. deviation of RTT samples during this interval (protocol-agnostic jitter proxy)."),
    ColumnSpec("throughput_mbps", "Mbps", "feature", "iperf3-reported throughput for this interval."),
    ColumnSpec("bandwidth_utilization_pct", "%", "feature", "throughput_mbps / bottleneck_bw_mbps * 100."),
    ColumnSpec("packet_rate_pps", "packets/sec", "feature", "packets_sent in this interval / interval duration."),
    ColumnSpec("queue_length", "packets", "feature", "Snapshot of bottleneck qdisc backlog (tc -s qdisc) taken during this interval."),
    ColumnSpec("retransmissions", "count", "feature", "TCP retransmits reported by iperf3 for this interval. Always 0 for UDP traffic (no retransmission concept) -- only meaningful when traffic_type='tcp'."),
    ColumnSpec("packets_sent", "count", "feature", "Packets sent during this interval (UDP only; iperf3 does not report packet counts for TCP)."),
    ColumnSpec("packets_received", "count", "feature", "packets_sent - lost_packets for this interval (UDP only)."),
    ColumnSpec("packet_loss_pct", "%", "feature", "Loss percentage MEASURED DURING THIS interval (iperf3 UDP lost_percent). Safe as an input feature for the NEXT row's target -- see docs leakage section."),

    # Target: added in a separate post-processing step (targets.py), never
    # written directly by the live collector, so the raw collection output
    # can never accidentally contain future information.
    ColumnSpec("target_next_packet_loss_pct", "%", "target",
               "packet_loss_pct measured `horizon` intervals AHEAD, within the same experiment_id. NaN for the last `horizon` row(s) of each experiment (no future data exists)."),
]

CSV_COLUMNS: list[str] = [c.name for c in SCHEMA]

FEATURE_COLUMNS: list[str] = [c.name for c in SCHEMA if c.role in ("feature", "config")]
TARGET_COLUMN = "target_next_packet_loss_pct"


def describe() -> str:
    """Human-readable rendering of the schema, used by docs generation / debugging."""
    lines = []
    for c in SCHEMA:
        lines.append(f"{c.name} ({c.unit}, {c.role}): {c.description}")
    return "\n".join(lines)
