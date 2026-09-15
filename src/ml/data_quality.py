"""Dataset quality report (Phase 5, Step 6).

Summarizes a COMBINED, already-validated dataset (from ml.dataset.load_dataset,
which already applies Phase 2's validate_rows/validate_no_leakage and drops
end-of-experiment rows) -- this module does not re-implement or weaken that
validation, it reports on top of it. Read-only: nothing here ever modifies
the input DataFrame or the underlying raw CSVs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from network.schema import TARGET_COLUMN

CONFIG_COLUMNS = ["bottleneck_bw_mbps", "bottleneck_delay_ms", "bottleneck_queue_pkts",
                  "traffic_type", "active_connections"]
NUMERIC_REPORT_COLUMNS = [
    "current_rtt_ms", "current_jitter_ms", "throughput_mbps", "bandwidth_utilization_pct",
    "packet_rate_pps", "queue_length", "retransmissions", "packets_sent", "packets_received",
    "packet_loss_pct",
]


@dataclass
class DatasetOverview:
    total_rows: int
    n_experiment_groups: int
    n_unique_configurations: int
    repetitions_per_configuration: dict[str, int]
    timestamp_min: float
    timestamp_max: float


@dataclass
class TargetSummary:
    min: float
    max: float
    mean: float
    median: float
    std: float
    zero_loss_pct: float
    nonzero_loss_pct: float


@dataclass
class DataIssues:
    duplicated_experiment_interval_pairs: int
    fully_duplicated_rows: int
    experiments_with_missing_intervals: list[str]
    experiments_with_zero_target_variance: list[str]
    out_of_range_loss_rows: int
    packets_received_exceeds_sent_rows: int


@dataclass
class DataQualityReport:
    overview: DatasetOverview
    feature_statistics: dict[str, dict[str, float]]
    target_summary: TargetSummary
    network_conditions: dict[str, dict[str, float]]
    issues: DataIssues
    source_files: list[str] = field(default_factory=list)


def _configuration_key(df: pd.DataFrame) -> pd.Series:
    """A string key identifying the distinct (non-repetition) network
    configuration a row belongs to -- everything an experiment_id encodes
    except its repetition suffix."""
    return df[CONFIG_COLUMNS].astype(str).agg("|".join, axis=1)


def dataset_overview(df: pd.DataFrame) -> DatasetOverview:
    config_keys = _configuration_key(df)
    per_experiment_config = df.assign(_config=config_keys).groupby("experiment_id")["_config"].first()
    repetitions_per_configuration = per_experiment_config.value_counts().to_dict()
    return DatasetOverview(
        total_rows=len(df),
        n_experiment_groups=int(df["experiment_id"].nunique()),
        n_unique_configurations=int(per_experiment_config.nunique()),
        repetitions_per_configuration={str(k): int(v) for k, v in repetitions_per_configuration.items()},
        timestamp_min=float(df["timestamp"].min()),
        timestamp_max=float(df["timestamp"].max()),
    )


def feature_statistics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in NUMERIC_REPORT_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "min": float(series.min()) if series.count() else float("nan"),
            "max": float(series.max()) if series.count() else float("nan"),
            "mean": float(series.mean()) if series.count() else float("nan"),
            "std": float(series.std()) if series.count() > 1 else 0.0,
        }
    return stats


def target_summary(df: pd.DataFrame) -> TargetSummary:
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna()
    if len(target) == 0:
        raise ValueError("No non-null target values to summarize")
    zero_pct = float((target == 0).mean() * 100.0)
    return TargetSummary(
        min=float(target.min()), max=float(target.max()), mean=float(target.mean()),
        median=float(target.median()), std=float(target.std()) if len(target) > 1 else 0.0,
        zero_loss_pct=zero_pct, nonzero_loss_pct=100.0 - zero_pct,
    )


def network_condition_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    condition_columns = ["bottleneck_bw_mbps", "throughput_mbps", "current_rtt_ms",
                          "current_jitter_ms", "queue_length", "active_connections"]
    summary = {}
    for col in condition_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "min": float(series.min()), "max": float(series.max()),
            "mean": float(series.mean()), "std": float(series.std()) if series.count() > 1 else 0.0,
        }
    return summary


def detect_issues(df: pd.DataFrame) -> DataIssues:
    """Read-only checks for problems the underlying validators wouldn't
    necessarily surface on their own (duplicates, gaps, degenerate
    experiments) -- complementary to, not a replacement for,
    network.validation.validate_rows / validate_no_leakage."""
    pair_counts = df.groupby(["experiment_id", "interval_index"]).size()
    duplicated_pairs = int((pair_counts > 1).sum())

    fully_duplicated = int(df.duplicated(keep=False).sum())

    missing_interval_experiments = []
    zero_variance_experiments = []
    for experiment_id, group in df.groupby("experiment_id"):
        indices = sorted(group["interval_index"].tolist())
        if indices and indices != list(range(indices[0], indices[0] + len(indices))):
            missing_interval_experiments.append(str(experiment_id))
        target_values = pd.to_numeric(group[TARGET_COLUMN], errors="coerce").dropna()
        if len(target_values) > 1 and target_values.nunique() == 1:
            zero_variance_experiments.append(str(experiment_id))

    loss = pd.to_numeric(df["packet_loss_pct"], errors="coerce")
    out_of_range = int(((loss < 0) | (loss > 100)).sum())

    sent = pd.to_numeric(df["packets_sent"], errors="coerce")
    received = pd.to_numeric(df["packets_received"], errors="coerce")
    exceeds = int((received > sent).sum())

    return DataIssues(
        duplicated_experiment_interval_pairs=duplicated_pairs,
        fully_duplicated_rows=fully_duplicated,
        experiments_with_missing_intervals=sorted(missing_interval_experiments),
        experiments_with_zero_target_variance=sorted(zero_variance_experiments),
        out_of_range_loss_rows=out_of_range,
        packets_received_exceeds_sent_rows=exceeds,
    )


def build_data_quality_report(df: pd.DataFrame, source_files: list[str] | None = None) -> DataQualityReport:
    return DataQualityReport(
        overview=dataset_overview(df),
        feature_statistics=feature_statistics(df),
        target_summary=target_summary(df),
        network_conditions=network_condition_summary(df),
        issues=detect_issues(df),
        source_files=source_files or [],
    )


def render_markdown(report: DataQualityReport, title: str = "Dataset Quality Report") -> str:
    ov, tgt, iss = report.overview, report.target_summary, report.issues
    lines = [f"# {title}", ""]
    lines += [
        "## Overview", "",
        f"- Total rows: {ov.total_rows}",
        f"- Experiment groups: {ov.n_experiment_groups}",
        f"- Unique configurations: {ov.n_unique_configurations}",
        f"- Repetitions per configuration: {ov.repetitions_per_configuration}",
        f"- Timestamp range: {ov.timestamp_min} - {ov.timestamp_max}", "",
    ]
    lines += [
        "## Target (`target_next_packet_loss_pct`)", "",
        f"- min={tgt.min:.4f}  max={tgt.max:.4f}  mean={tgt.mean:.4f}  median={tgt.median:.4f}  std={tgt.std:.4f}",
        f"- Zero-loss intervals: {tgt.zero_loss_pct:.2f}%",
        f"- Non-zero-loss intervals: {tgt.nonzero_loss_pct:.2f}%", "",
    ]
    lines += ["## Feature Statistics", "", "| Feature | Count | Missing | Min | Max | Mean | Std |", "|---|---|---|---|---|---|---|"]
    for name, s in report.feature_statistics.items():
        lines.append(f"| {name} | {s['count']} | {s['missing']} | {s['min']:.4g} | {s['max']:.4g} | {s['mean']:.4g} | {s['std']:.4g} |")
    lines.append("")
    lines += ["## Network Conditions", "", "| Condition | Min | Max | Mean | Std |", "|---|---|---|---|---|"]
    for name, s in report.network_conditions.items():
        lines.append(f"| {name} | {s['min']:.4g} | {s['max']:.4g} | {s['mean']:.4g} | {s['std']:.4g} |")
    lines.append("")
    lines += [
        "## Data Quality Issues", "",
        f"- Duplicated (experiment_id, interval_index) pairs: {iss.duplicated_experiment_interval_pairs}",
        f"- Fully duplicated rows: {iss.fully_duplicated_rows}",
        f"- Experiments with missing intervals: {iss.experiments_with_missing_intervals or 'none'}",
        f"- Experiments with zero target variance: {iss.experiments_with_zero_target_variance or 'none'}",
        f"- Rows with out-of-range packet_loss_pct: {iss.out_of_range_loss_rows}",
        f"- Rows with packets_received > packets_sent: {iss.packets_received_exceeds_sent_rows}", "",
    ]
    if report.source_files:
        lines += ["## Source Files", ""] + [f"- {f}" for f in report.source_files]
    return "\n".join(lines)


def save_report(report: DataQualityReport, out_dir: Path, basename: str = "dataset_quality") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{basename}.json"
    md_path = out_dir / f"{basename}.md"
    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
