"""Parameter-sweep configuration for Phase 2 dataset generation.

Builds on top of Phase 1's ExperimentConfig (src/network/config.py)
rather than duplicating any of its validation -- this module only
decides WHICH ExperimentConfig instances to construct and how to name
them; ExperimentConfig itself still enforces every Phase 1 constraint
(host cap, positive bandwidth, minimum duration, etc.) via its own
__post_init__.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from network.config import ExperimentConfig


@dataclass
class SweepConfig:
    """A small, deliberately conservative default sweep.

    Offered load is expressed as a FACTOR of each bandwidth value
    (not an absolute Mbps figure), because "below/near/above bottleneck"
    only makes sense relative to whatever bottleneck_bw_mbps is being
    tested in that combination -- e.g. factor 0.7 means "70% of
    whatever bandwidth this combination uses", covering the
    below-bottleneck case for every bandwidth value tested, and factor
    1.3 covers the above-bottleneck (genuine congestion) case for all
    of them (see PHASE_1_NETWORK_EXPERIMENT.md Section 6 for why
    exceeding bandwidth -- not an injected loss knob -- is what should
    drive loss).

    Defaults: 2 bandwidths x 2 delays x 2 load factors x 1 flow-count
    x 1 repetition = 8 experiments. Deliberately small for a
    resource-constrained dev machine; widen these lists once a live
    run has actually been validated.
    """

    bottleneck_bw_mbps_values: list[float] = field(default_factory=lambda: [1.0, 2.0])
    bottleneck_delay_ms_values: list[float] = field(default_factory=lambda: [0.0, 20.0])
    offered_load_factors: list[float] = field(default_factory=lambda: [0.7, 1.3])
    n_flows_values: list[int] = field(default_factory=lambda: [1])

    traffic_type: str = "udp"
    bottleneck_queue_pkts: int = 20
    sample_interval_s: float = 2.0
    duration_s: float = 20.0
    target_horizon_intervals: int = 1
    repetitions: int = 1

    sweep_id: str = field(default_factory=lambda: f"sweep-{int(time.time())}")

    def __post_init__(self) -> None:
        for name, values in (
            ("bottleneck_bw_mbps_values", self.bottleneck_bw_mbps_values),
            ("bottleneck_delay_ms_values", self.bottleneck_delay_ms_values),
            ("offered_load_factors", self.offered_load_factors),
            ("n_flows_values", self.n_flows_values),
        ):
            if not values:
                raise ValueError(f"{name} must contain at least one value")
        if any(v <= 0 for v in self.bottleneck_bw_mbps_values):
            raise ValueError("bottleneck_bw_mbps_values must all be positive")
        if any(v < 0 for v in self.bottleneck_delay_ms_values):
            raise ValueError("bottleneck_delay_ms_values must all be non-negative")
        if any(v <= 0 for v in self.offered_load_factors):
            raise ValueError("offered_load_factors must all be positive")
        if any(n != 1 for n in self.n_flows_values):
            raise NotImplementedError(
                "n_flows_values with more than a single concurrent flow is not yet "
                "supported: Phase 1's run_experiment() drives exactly one sender/receiver "
                "pair (see docs/PHASE_1_NETWORK_EXPERIMENT.md Section 9). Set "
                "n_flows_values=[1], or extend src/network/experiment.py to drive "
                "multiple concurrent iperf3 flows before sweeping this dimension."
            )
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")

    @property
    def n_combinations(self) -> int:
        return (
            len(self.bottleneck_bw_mbps_values)
            * len(self.bottleneck_delay_ms_values)
            * len(self.offered_load_factors)
            * len(self.n_flows_values)
        )

    @property
    def n_experiments(self) -> int:
        return self.n_combinations * self.repetitions


def _fmt(value: float) -> str:
    """Compact, filename-safe number formatting (1.0 -> '1', 0.7 -> '0.7')."""
    return f"{value:g}"


def build_experiment_id(sweep_id: str, bw: float, delay: float, load_factor: float, n_flows: int, rep: int) -> str:
    return (
        f"{sweep_id}__bw{_fmt(bw)}__delay{_fmt(delay)}"
        f"__load{_fmt(load_factor)}__flows{n_flows}__rep{rep}"
    )


def expand_configs(sweep: SweepConfig) -> list[ExperimentConfig]:
    """Turn a SweepConfig into the full, ordered list of ExperimentConfig runs.

    Order is deterministic (itertools.product order, repetitions innermost)
    so the same SweepConfig always expands to the same experiment_ids in
    the same order -- part of what makes the dataset reproducible.
    """
    combos = itertools.product(
        sweep.bottleneck_bw_mbps_values,
        sweep.bottleneck_delay_ms_values,
        sweep.offered_load_factors,
        sweep.n_flows_values,
    )
    configs: list[ExperimentConfig] = []
    for bw, delay, load_factor, n_flows in combos:
        for rep in range(sweep.repetitions):
            experiment_id = build_experiment_id(sweep.sweep_id, bw, delay, load_factor, n_flows, rep)
            configs.append(ExperimentConfig(
                bottleneck_bw_mbps=bw,
                bottleneck_delay_ms=delay,
                offered_load_mbps=bw * load_factor,
                active_connections=n_flows,
                traffic_type=sweep.traffic_type,
                bottleneck_queue_pkts=sweep.bottleneck_queue_pkts,
                sample_interval_s=sweep.sample_interval_s,
                duration_s=sweep.duration_s,
                target_horizon_intervals=sweep.target_horizon_intervals,
                experiment_id=experiment_id,
            ))
    return configs
