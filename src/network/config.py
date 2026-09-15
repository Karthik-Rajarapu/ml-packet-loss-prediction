"""Experiment configuration.

Plain dataclass with defaults deliberately sized for a resource-limited
development machine (2-4 hosts, short duration) -- see the RAM/CPU notes
in docs/ENVIRONMENT_SETUP.md. Override individual fields when calling
run_experiment.py rather than editing these defaults in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Topology size -- kept small deliberately (Phase 1 resource constraint).
    n_left_hosts: int = 2
    n_right_hosts: int = 2

    # Bottleneck link (the only shaped link in the dumbbell topology).
    bottleneck_bw_mbps: float = 5.0
    bottleneck_delay_ms: float = 20.0
    bottleneck_queue_pkts: int = 20

    # Traffic generation.
    traffic_type: str = "udp"          # 'udp' (recommended for Phase 1) or 'tcp'
    offered_load_mbps: float = 8.0      # intentionally > bottleneck_bw_mbps to force real queueing/loss
    active_connections: int = 1         # number of concurrent iperf3 flows

    # Sampling.
    sample_interval_s: float = 2.0
    duration_s: float = 20.0            # short by design -- Phase 1 is a smoke-scale run
    target_horizon_intervals: int = 1   # Delta: how many intervals ahead the target looks

    # Identifiers.
    experiment_id: str = field(default_factory=lambda: f"exp-{int(time.time())}")

    def __post_init__(self) -> None:
        if self.n_left_hosts < 1 or self.n_right_hosts < 1:
            raise ValueError("Topology needs at least one host on each side")
        if self.n_left_hosts + self.n_right_hosts > 4:
            raise ValueError(
                "Phase 1 caps total hosts at 4 to fit this project's resource-constrained "
                "dev machine -- raise this deliberately, not by accident"
            )
        if self.bottleneck_bw_mbps <= 0:
            raise ValueError("bottleneck_bw_mbps must be positive")
        if self.traffic_type not in ("udp", "tcp"):
            raise ValueError("traffic_type must be 'udp' or 'tcp'")
        if self.duration_s / self.sample_interval_s < 2:
            raise ValueError("duration_s must span at least 2 sample intervals to produce a usable target")
