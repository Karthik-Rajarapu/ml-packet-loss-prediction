"""LIVE Mininet integration tests -- explicitly separated from the rest of
the suite (Phase 5 Step 14: "Live Mininet tests should be explicitly
separated from normal unit tests").

Every test here is skipped automatically unless mn/iperf3/tc/ping/ovs-vsctl
are all actually on PATH (network.validation.check_environment()) -- on
this development machine that is currently true for none of them except
`ping`, so this entire module is expected to show as SKIPPED, not PASSED,
in the test run reported for Phase 5. A "skipped" count here must never be
reported as "passed" -- it means "not verified on this machine", not
"verified and working".

Once WSL2/Mininet/iperf3/tc/Open vSwitch are available, these tests start
running for real and exercise the actual network stack -- nothing here is
mocked.
"""

from __future__ import annotations

import pytest

from network.config import ExperimentConfig
from network.experiment import run_experiment
from network.validation import check_environment

_env = check_environment()
requires_live_mininet = pytest.mark.skipif(
    not _env.all_available,
    reason=f"Live Mininet environment unavailable on this machine (missing: {_env.missing}) -- "
           f"see docs/ENVIRONMENT_SETUP.md",
)


@requires_live_mininet
def test_live_dumbbell_experiment_produces_a_valid_csv(tmp_path):
    """The real Phase 1 smoke test, as an actual pytest case: build a
    minimal 2-host dumbbell, run a short real experiment, and check the
    CSV Phase 1's own run_experiment() writes is well-formed."""
    config = ExperimentConfig(
        n_left_hosts=1, n_right_hosts=1,
        bottleneck_bw_mbps=2.0, bottleneck_delay_ms=10.0,
        duration_s=10.0, sample_interval_s=2.0, offered_load_mbps=4.0,
    )
    csv_path = run_experiment(config, tmp_path)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0


@requires_live_mininet
def test_live_experiment_produces_observable_congestion(tmp_path):
    """offered_load_mbps > bottleneck_bw_mbps should produce at least some
    non-zero packet_loss_pct reading -- the core premise the whole project
    depends on (PROJECT_PLAN.md Section 7)."""
    import csv as csv_module

    config = ExperimentConfig(
        n_left_hosts=1, n_right_hosts=1,
        bottleneck_bw_mbps=1.0, bottleneck_delay_ms=5.0, bottleneck_queue_pkts=10,
        duration_s=12.0, sample_interval_s=2.0, offered_load_mbps=4.0,  # 4x over bottleneck
    )
    csv_path = run_experiment(config, tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    losses = [float(r["packet_loss_pct"]) for r in rows if r["packet_loss_pct"] not in ("", None)]
    assert any(loss > 0 for loss in losses), (
        "Expected at least one interval with observable packet loss when offered "
        "load (4 Mbps) exceeds bottleneck bandwidth (1 Mbps) -- if this fails, the "
        "experiment's congestion conditions need investigating before a real sweep."
    )
