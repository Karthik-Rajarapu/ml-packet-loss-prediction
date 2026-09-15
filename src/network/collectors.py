"""Pure parsing functions for iperf3 / ping / tc output.

None of these functions touch the network or Mininet -- they take text
that some other component captured and return plain dicts/lists. Keeping
them pure means they can be unit-tested on any machine (including one
without Mininet/WSL), using fixture text captured from real tool output.
"""

from __future__ import annotations

import json
import re
import statistics
from typing import Any


def parse_iperf3_udp_json(raw_json: str) -> list[dict[str, Any]]:
    """Parse `iperf3 -c <host> -u -J -i <interval>` output.

    Returns one dict per reported interval with keys:
    start_s, end_s, throughput_mbps, packets_sent, packets_lost, packet_loss_pct.

    Raises ValueError if the JSON is missing the fields we depend on --
    we never silently substitute zeros for a parse failure.
    """
    data = json.loads(raw_json)
    try:
        intervals = data["intervals"]
    except KeyError as exc:
        raise ValueError("iperf3 JSON missing 'intervals' -- unexpected output format") from exc

    rows = []
    for interval in intervals:
        summary = interval.get("sum")
        if summary is None:
            raise ValueError("iperf3 interval missing 'sum' block")
        try:
            packets_sent = int(summary["packets"])
            packets_lost = int(summary["lost_packets"])
            throughput_mbps = float(summary["bits_per_second"]) / 1_000_000.0
        except KeyError as exc:
            raise ValueError(f"iperf3 interval summary missing expected field: {exc}") from exc

        loss_pct = (packets_lost / packets_sent * 100.0) if packets_sent > 0 else 0.0
        rows.append({
            "start_s": float(summary["start"]),
            "end_s": float(summary["end"]),
            "throughput_mbps": throughput_mbps,
            "packets_sent": packets_sent,
            "packets_lost": packets_lost,
            "packets_received": packets_sent - packets_lost,
            "packet_loss_pct": loss_pct,
        })
    return rows


_TCP_RETRANSMIT_RE = re.compile(r'"retransmits"\s*:\s*(\d+)')


def parse_iperf3_tcp_json(raw_json: str) -> list[dict[str, Any]]:
    """Parse `iperf3 -c <host> -J -i <interval>` (TCP) output.

    TCP mode has no packet-count/loss fields (iperf3 reports bytes, not
    packets, for TCP) -- only throughput and retransmits are reliable.
    packets_sent/packets_received/packet_loss_pct are left as None so
    callers don't mistake "no data" for "zero loss".
    """
    data = json.loads(raw_json)
    try:
        intervals = data["intervals"]
    except KeyError as exc:
        raise ValueError("iperf3 JSON missing 'intervals' -- unexpected output format") from exc

    rows = []
    for interval in intervals:
        summary = interval.get("sum")
        if summary is None:
            raise ValueError("iperf3 interval missing 'sum' block")
        rows.append({
            "start_s": float(summary["start"]),
            "end_s": float(summary["end"]),
            "throughput_mbps": float(summary["bits_per_second"]) / 1_000_000.0,
            "retransmits": int(summary.get("retransmits", 0)),
        })
    return rows


_PING_REPLY_RE = re.compile(r"time[=<]([\d.]+)\s*ms")


def parse_ping_rtts(raw_output: str) -> list[float]:
    """Extract each reply's RTT (ms) from `ping` stdout.

    Works on the standard iputils-ping line format:
    '64 bytes from 10.0.0.3: icmp_seq=1 ttl=64 time=1.23 ms'
    """
    rtts = [float(m.group(1)) for m in _PING_REPLY_RE.finditer(raw_output)]
    if not rtts:
        raise ValueError("No RTT samples found in ping output -- host may be unreachable")
    return rtts


def summarize_rtts(rtts: list[float]) -> dict[str, float]:
    """Mean RTT and std-dev-as-jitter-proxy for one sampling interval."""
    if not rtts:
        raise ValueError("Cannot summarize an empty RTT list")
    mean_rtt = statistics.mean(rtts)
    jitter = statistics.pstdev(rtts) if len(rtts) > 1 else 0.0
    return {"current_rtt_ms": mean_rtt, "current_jitter_ms": jitter}


_TC_BACKLOG_RE = re.compile(r"backlog\s+\d+b\s+(\d+)p")


def parse_tc_qdisc_backlog(raw_output: str) -> int:
    """Extract queue occupancy (packets) from `tc -s qdisc show dev <iface>`.

    Looks for the 'backlog Xb Yp' field in netem/tbf statistics output.
    Returns 0 if the qdisc is idle (field present with 0p, or absent
    entirely on a freshly-created interface with no traffic yet) --
    that is a legitimate "empty queue" reading, not a parse failure.
    """
    match = _TC_BACKLOG_RE.search(raw_output)
    if match is None:
        return 0
    return int(match.group(1))
