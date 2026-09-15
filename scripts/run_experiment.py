#!/usr/bin/env python3
"""CLI entry point for a single Phase 1 network experiment.

Must be run as root inside WSL2 Ubuntu (Mininet requires root to create
network namespaces/veth pairs):

    sudo python3 scripts/run_experiment.py

Optional overrides:

    sudo python3 scripts/run_experiment.py \\
        --bottleneck-bw-mbps 3 --duration-s 30 --sample-interval-s 2

Refuses to run (raises, writes nothing) if Mininet/iperf3/tc/ping/
ovs-vsctl aren't available -- see network.validation.require_environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network.config import ExperimentConfig  # noqa: E402
from network.experiment import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottleneck-bw-mbps", type=float, default=5.0)
    parser.add_argument("--bottleneck-delay-ms", type=float, default=20.0)
    parser.add_argument("--bottleneck-queue-pkts", type=int, default=20)
    parser.add_argument("--traffic-type", choices=["udp", "tcp"], default="udp")
    parser.add_argument("--offered-load-mbps", type=float, default=8.0)
    parser.add_argument("--sample-interval-s", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        bottleneck_bw_mbps=args.bottleneck_bw_mbps,
        bottleneck_delay_ms=args.bottleneck_delay_ms,
        bottleneck_queue_pkts=args.bottleneck_queue_pkts,
        traffic_type=args.traffic_type,
        offered_load_mbps=args.offered_load_mbps,
        sample_interval_s=args.sample_interval_s,
        duration_s=args.duration_s,
    )
    print(f"Running experiment {config.experiment_id} "
          f"(bw={config.bottleneck_bw_mbps}Mbps, delay={config.bottleneck_delay_ms}ms, "
          f"duration={config.duration_s}s)...")
    output_path = run_experiment(config, args.output_dir)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
