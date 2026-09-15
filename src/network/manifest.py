"""Dataset manifest: records exactly what a generation run attempted and
produced, for reproducibility and for the project report. Never records
a "success" for an experiment that wasn't actually run and validated.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from network.schema import FEATURE_COLUMNS, TARGET_COLUMN
from network.sweep import SweepConfig


@dataclass
class ExperimentResult:
    experiment_id: str
    status: str  # "success" or "failed"
    csv_path: str | None = None
    row_count: int | None = None
    error: str | None = None


def _tool_version(tool: str, version_flag: str = "--version") -> str | None:
    """Best-effort `tool --version` capture. Returns None if unavailable --
    never raises, since this is metadata-gathering, not a correctness check.
    """
    if shutil.which(tool) is None:
        return None
    try:
        result = subprocess.run(
            [tool, version_flag], capture_output=True, text=True, timeout=5, check=False,
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        return output[0] if output else None
    except Exception:  # noqa: BLE001 -- metadata only, must never break dataset generation
        return None


def environment_snapshot() -> dict[str, Any]:
    """Best-effort record of the software environment, for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "tool_versions": {
            tool: _tool_version(tool) for tool in ("mn", "iperf3", "tc", "ovs-vsctl", "ping")
        },
    }


def build_manifest(
    sweep: SweepConfig,
    results: list[ExperimentResult],
    started_at: str,
    finished_at: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    succeeded = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]
    return {
        "sweep_id": sweep.sweep_id,
        "dry_run": dry_run,
        "generated_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "output_dir": str(output_dir),
        "sweep_config": asdict(sweep),
        "n_experiments_planned": len(results),
        "n_experiments_succeeded": len(succeeded),
        "n_experiments_failed": len(failed),
        "total_rows_written": sum(r.row_count or 0 for r in succeeded),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "environment": environment_snapshot(),
        "experiments": [asdict(r) for r in results],
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
