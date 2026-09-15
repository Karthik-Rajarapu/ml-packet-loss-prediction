"""Unit tests for the pure parsing functions -- no Mininet required.

Run with: py -3 -m pytest tests/  (Windows)  or  python3 -m pytest tests/  (Linux/WSL)
"""

from pathlib import Path

import pytest

from network.collectors import (
    parse_iperf3_tcp_json,
    parse_iperf3_udp_json,
    parse_ping_rtts,
    parse_tc_qdisc_backlog,
    summarize_rtts,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_iperf3_udp_json_returns_one_row_per_interval():
    raw = (FIXTURES / "iperf3_udp_sample.json").read_text()
    rows = parse_iperf3_udp_json(raw)
    assert len(rows) == 3


def test_parse_iperf3_udp_json_computes_loss_pct_from_counts():
    raw = (FIXTURES / "iperf3_udp_sample.json").read_text()
    rows = parse_iperf3_udp_json(raw)
    # interval 2: lost_packets=48, packets=733 -> 48/733*100
    assert rows[1]["packets_lost"] == 48
    assert rows[1]["packets_sent"] == 733
    assert rows[1]["packet_loss_pct"] == pytest.approx(48 / 733 * 100, rel=1e-6)


def test_parse_iperf3_udp_json_loss_increases_across_intervals():
    """Fixture models a bottleneck-induced congestion ramp -- loss should rise."""
    raw = (FIXTURES / "iperf3_udp_sample.json").read_text()
    rows = parse_iperf3_udp_json(raw)
    losses = [r["packet_loss_pct"] for r in rows]
    assert losses == sorted(losses), "fixture is expected to show monotonically increasing loss"


def test_parse_iperf3_udp_json_missing_intervals_raises():
    with pytest.raises(ValueError):
        parse_iperf3_udp_json('{"start": {}}')


def test_parse_iperf3_tcp_json_has_no_packet_counts():
    raw_tcp = (
        '{"intervals": [{"sum": {"start": 0, "end": 2, "bits_per_second": 5000000.0, "retransmits": 3}}]}'
    )
    rows = parse_iperf3_tcp_json(raw_tcp)
    assert rows[0]["retransmits"] == 3
    assert "packets_sent" not in rows[0], "TCP mode must not fabricate packet counts iperf3 doesn't report"


def test_parse_ping_rtts_extracts_all_replies():
    raw = (FIXTURES / "ping_sample.txt").read_text()
    rtts = parse_ping_rtts(raw)
    assert rtts == [1.23, 21.05, 1.10]


def test_parse_ping_rtts_empty_raises():
    with pytest.raises(ValueError):
        parse_ping_rtts("no replies here")


def test_summarize_rtts_mean_and_jitter():
    stats = summarize_rtts([1.23, 21.05, 1.10])
    assert stats["current_rtt_ms"] == pytest.approx((1.23 + 21.05 + 1.10) / 3)
    assert stats["current_jitter_ms"] > 0


def test_summarize_rtts_single_sample_has_zero_jitter():
    stats = summarize_rtts([5.0])
    assert stats["current_jitter_ms"] == 0.0


def test_parse_tc_qdisc_backlog_finds_packet_count():
    raw = (FIXTURES / "tc_qdisc_sample.txt").read_text()
    backlog = parse_tc_qdisc_backlog(raw)
    assert backlog >= 0
    assert isinstance(backlog, int)


def test_parse_tc_qdisc_backlog_idle_queue_returns_zero():
    assert parse_tc_qdisc_backlog("qdisc noqueue 0: root refcnt 2") == 0
