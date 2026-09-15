"""Alias-based detection of the project's canonical feature columns
(network.schema.FEATURE_COLUMNS) in an arbitrary uploaded CSV.

Never silently accepts an ambiguous match: only an exact (normalized)
name/alias match is "high" confidence; anything else is "low" or "none"
and must be explicitly confirmed or corrected by the user before it's
used for anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from network.schema import FEATURE_COLUMNS

# Config-role features may legitimately be supplied as ONE constant value
# for the whole dataset (e.g. a single Mininet run's bottleneck bandwidth
# doesn't vary row-to-row) instead of requiring a per-row column -- a
# real, common case, not a workaround. Measured features must come from
# an actual column; a constant wouldn't represent a genuine measurement.
CONSTANT_ELIGIBLE_FEATURES: set[str] = {
    "bottleneck_bw_mbps", "bottleneck_delay_ms", "bottleneck_queue_pkts",
    "active_connections", "traffic_type",
}

FEATURE_ALIASES: dict[str, list[str]] = {
    "bottleneck_bw_mbps": ["bandwidth", "bottleneck_bw_mbps", "bandwidth_mbps", "link_bandwidth", "bw"],
    "bottleneck_delay_ms": ["delay", "bottleneck_delay_ms", "link_delay", "delay_ms", "latency_ms_config"],
    "bottleneck_queue_pkts": ["queue_pkts", "bottleneck_queue_pkts", "max_queue", "queue_capacity", "queue_limit"],
    "traffic_type": ["traffic_type", "protocol", "proto"],
    "active_connections": ["active_connections", "flows", "concurrent_connections", "num_flows", "connections"],
    "current_rtt_ms": ["rtt", "latency", "round_trip_time", "current_rtt_ms", "rtt_ms"],
    "current_jitter_ms": ["jitter", "rtt_jitter", "current_jitter_ms", "jitter_ms"],
    "throughput_mbps": ["throughput", "throughput_mbps", "bitrate", "goodput", "throughput_mbit"],
    "bandwidth_utilization_pct": ["bandwidth_utilization_pct", "utilization", "bandwidth_utilization", "link_utilization"],
    "packet_rate_pps": ["packet_rate", "packet_rate_pps", "pps", "packets_per_second"],
    "queue_length": ["queue_length", "queue", "current_queue", "queue_len", "queue_occupancy"],
    "retransmissions": ["retransmissions", "retransmits", "retx", "retrans"],
    "packets_sent": ["packets_sent", "sent_packets", "pkts_sent", "tx_packets"],
    "packets_received": ["packets_received", "received_packets", "pkts_received", "rx_packets"],
    "packet_loss_pct": ["packet_loss", "packet_loss_pct", "loss", "loss_percentage", "loss_pct", "packetloss"],
}

# Every FEATURE_COLUMNS entry must have an alias list, or it could never
# be detected -- enforced by a test (tests/test_column_mapping.py), not
# just this comment.


@dataclass(frozen=True)
class ColumnDetection:
    canonical_feature: str
    detected_column: str | None
    confidence: str  # "high" | "low" | "none"


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def detect_column_mapping(uploaded_columns: list[str]) -> dict[str, ColumnDetection]:
    """One ColumnDetection per canonical feature.

    'high': an uploaded column's normalized name exactly matches the
    canonical name or one of its known aliases.
    'low': an uploaded column's normalized name partially overlaps an
    alias (substring match) -- must be explicitly confirmed, never
    auto-applied.
    'none': nothing found -- the user must map manually or, for
    CONSTANT_ELIGIBLE_FEATURES, supply a fixed value instead.
    """
    normalized_lookup = {_normalize(c): c for c in uploaded_columns}
    results: dict[str, ColumnDetection] = {}

    for feature in FEATURE_COLUMNS:
        aliases = FEATURE_ALIASES.get(feature, [])
        normalized_aliases = sorted({_normalize(a) for a in aliases} | {_normalize(feature)})

        exact = next((normalized_lookup[a] for a in normalized_aliases if a in normalized_lookup), None)
        if exact is not None:
            results[feature] = ColumnDetection(feature, exact, "high")
            continue

        partial = None
        for norm_col, original_col in normalized_lookup.items():
            if any(len(alias) > 2 and (alias in norm_col or norm_col in alias) for alias in normalized_aliases):
                partial = original_col
                break
        results[feature] = ColumnDetection(feature, partial, "low" if partial else "none")

    return results


def apply_column_mapping(
    df: pd.DataFrame,
    confirmed_mapping: dict[str, str],
    constant_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Rename `df`'s columns to canonical feature names per
    `confirmed_mapping` ({canonical_feature: uploaded_column}), and add
    any `constant_values` columns ({canonical_feature: value}) for
    features the user chose to supply as one fixed value rather than a
    per-row column. Returns a NEW DataFrame; never mutates the input.
    """
    rename_map = {uploaded_col: feature for feature, uploaded_col in confirmed_mapping.items()}
    working = df.rename(columns=rename_map).copy()

    for feature, value in (constant_values or {}).items():
        working[feature] = value

    return working
