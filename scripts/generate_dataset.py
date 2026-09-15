#!/usr/bin/env python3
"""Phase 2 CLI: run a parameter sweep of Phase 1 experiments and write a
manifest. Must be run as root inside WSL2 Ubuntu for real (non-dry-run)
execution -- see docs/PHASE_1_NETWORK_EXPERIMENT.md and
docs/ENVIRONMENT_SETUP.md.

Dry run (safe anywhere, no network/Mininet touched, writes nothing):

    python3 scripts/generate_dataset.py --dry-run

Live run (requires Mininet/iperf3/tc/ping/ovs-vsctl on PATH; fails fast
with a clear error otherwise -- never fabricates data):

    sudo python3 scripts/generate_dataset.py --repetitions 2

Custom sweep dimensions via JSON (see SweepConfig fields):

    sudo python3 scripts/generate_dataset.py --sweep-config my_sweep.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network.generate import describe_sweep, run_sweep_and_write_manifest  # noqa: E402
from network.sweep import SweepConfig  # noqa: E402
from network.validation import require_environment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-config", type=Path, default=None,
                         help="JSON file overriding SweepConfig defaults (see src/network/sweep.py)")
    parser.add_argument("--repetitions", type=int, default=None,
                         help="Override repetitions from the sweep config")
    parser.add_argument("--sweep-id", type=str, default=None,
                         help="Override the auto-generated sweep_id")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifests")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the planned sweep and exit -- no network, no files written")
    return parser.parse_args()


def load_sweep(args: argparse.Namespace) -> SweepConfig:
    if args.sweep_config is not None:
        data = json.loads(args.sweep_config.read_text())
        sweep = SweepConfig(**data)
    else:
        sweep = SweepConfig()
    if args.repetitions is not None:
        sweep = dataclasses.replace(sweep, repetitions=args.repetitions)
    if args.sweep_id is not None:
        sweep = dataclasses.replace(sweep, sweep_id=args.sweep_id)
    return sweep


def print_dry_run(sweep: SweepConfig, output_dir: Path) -> None:
    summary = describe_sweep(sweep, output_dir)
    print("=" * 70)
    print("DRY RUN -- no network touched, no files written")
    print("=" * 70)
    print(f"sweep_id:            {summary['sweep_id']}")
    print(f"parameter combos:    {summary['n_combinations']}")
    print(f"repetitions:         {summary['repetitions']}")
    print(f"total experiments:   {summary['n_experiments']}")
    print(f"estimated duration:  {summary['estimated_duration_human']} "
          f"({summary['estimated_duration_s']:.0f}s)")
    print(f"output dir:          {summary['output_dir']}")
    print()
    print("parameters:")
    for key, value in summary["parameters"].items():
        print(f"  {key}: {value}")
    print()
    print("planned experiment_ids:")
    for experiment_id in summary["experiment_ids"]:
        print(f"  {experiment_id}")


def main() -> int:
    args = parse_args()
    sweep = load_sweep(args)

    if args.dry_run:
        print_dry_run(sweep, args.output_dir)
        return 0

    # Fail fast, before running experiment 1 of N, rather than partway through.
    require_environment()

    manifest, manifest_path = run_sweep_and_write_manifest(sweep, args.output_dir, args.manifest_dir)

    print()
    print(f"Manifest written to {manifest_path}")
    print(f"{manifest['n_experiments_succeeded']}/{manifest['n_experiments_planned']} experiments succeeded, "
          f"{manifest['total_rows_written']} total rows")

    return 0 if manifest["n_experiments_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
