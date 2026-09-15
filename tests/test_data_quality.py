"""Tests for ml.data_quality (Phase 5 Step 6). Uses the synthetic TEST
FIXTURE (ml_fixtures.py) and small hand-crafted edge cases -- never real
data, since none exists yet. Only checks structural/arithmetic
correctness, never a real-world quality judgement.
"""

import pandas as pd
import pytest

from ml.data_quality import (
    build_data_quality_report,
    detect_issues,
    feature_statistics,
    render_markdown,
    save_report,
    target_summary,
)
from ml.dataset import load_dataset
from ml_fixtures import make_fixture_dataframe
from network.schema import CSV_COLUMNS


def _labeled_df(n_experiments=4, n_intervals=5, tmp_path=None, id_prefix="fixture-exp"):
    """Route through the real load_dataset() (as scripts/dataset_report.py
    does) so these tests exercise the exact same validated/dropped shape
    a real report would see."""
    raw_df = make_fixture_dataframe(n_experiments=n_experiments, n_intervals=n_intervals, id_prefix=id_prefix)
    path = tmp_path / "fixture.csv"
    raw_df.to_csv(path, index=False)
    return load_dataset([path])


def test_dataset_overview_counts(tmp_path):
    df = _labeled_df(n_experiments=3, n_intervals=5, tmp_path=tmp_path)
    report = build_data_quality_report(df)
    assert report.overview.total_rows == len(df)
    assert report.overview.n_experiment_groups == 3
    # each fixture experiment has a distinct bandwidth -> distinct configuration
    assert report.overview.n_unique_configurations == 3


def test_target_summary_zero_and_nonzero_percentages(tmp_path):
    df = _labeled_df(n_experiments=1, n_intervals=6, tmp_path=tmp_path)
    summary = target_summary(df)
    assert summary.zero_loss_pct + summary.nonzero_loss_pct == pytest.approx(100.0)
    assert summary.min <= summary.mean <= summary.max


def test_target_summary_all_zero_loss():
    df = pd.DataFrame([
        {"experiment_id": "e1", "target_next_packet_loss_pct": 0.0},
        {"experiment_id": "e1", "target_next_packet_loss_pct": 0.0},
    ])
    summary = target_summary(df)
    assert summary.zero_loss_pct == 100.0
    assert summary.nonzero_loss_pct == 0.0


def test_feature_statistics_reports_every_numeric_feature(tmp_path):
    df = _labeled_df(n_experiments=2, n_intervals=4, tmp_path=tmp_path)
    stats = feature_statistics(df)
    assert "current_rtt_ms" in stats
    assert "packet_loss_pct" in stats
    assert stats["current_rtt_ms"]["missing"] == 0
    assert stats["current_rtt_ms"]["min"] <= stats["current_rtt_ms"]["max"]


def test_detect_issues_clean_data_has_no_issues(tmp_path):
    df = _labeled_df(n_experiments=2, n_intervals=5, tmp_path=tmp_path)
    issues = detect_issues(df)
    assert issues.duplicated_experiment_interval_pairs == 0
    assert issues.fully_duplicated_rows == 0
    assert issues.experiments_with_missing_intervals == []
    assert issues.out_of_range_loss_rows == 0
    assert issues.packets_received_exceeds_sent_rows == 0


def test_detect_issues_catches_fully_duplicated_row(tmp_path):
    df = _labeled_df(n_experiments=1, n_intervals=5, tmp_path=tmp_path)
    corrupted = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    issues = detect_issues(corrupted)
    assert issues.fully_duplicated_rows >= 2  # original + its duplicate


def test_detect_issues_catches_missing_interval(tmp_path):
    df = _labeled_df(n_experiments=1, n_intervals=6, tmp_path=tmp_path)
    # Remove one middle interval to create a gap in the sequence.
    gapped = df[df["interval_index"] != 2].reset_index(drop=True)
    issues = detect_issues(gapped)
    assert "fixture-exp-0" in issues.experiments_with_missing_intervals


def test_detect_issues_catches_zero_target_variance():
    rows = [
        {"experiment_id": "e1", "interval_index": i, "target_next_packet_loss_pct": 3.0,
         "packet_loss_pct": 3.0, "packets_sent": 10, "packets_received": 10}
        for i in range(4)
    ]
    df = pd.DataFrame(rows)
    issues = detect_issues(df)
    assert "e1" in issues.experiments_with_zero_target_variance


def test_render_markdown_contains_key_sections(tmp_path):
    df = _labeled_df(n_experiments=2, n_intervals=4, tmp_path=tmp_path)
    report = build_data_quality_report(df)
    markdown = render_markdown(report)
    for heading in ("## Overview", "## Target", "## Feature Statistics", "## Data Quality Issues"):
        assert heading in markdown


def test_save_report_writes_json_and_markdown(tmp_path):
    df = _labeled_df(n_experiments=2, n_intervals=4, tmp_path=tmp_path)
    report = build_data_quality_report(df, source_files=["fixture.csv"])
    out_dir = tmp_path / "out"
    json_path, md_path = save_report(report, out_dir, basename="test_report")
    assert json_path.exists() and json_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    assert json_path.parent == out_dir


def test_build_report_requires_full_schema_columns(tmp_path):
    """A sanity check that the report operates on the real CSV_COLUMNS
    shape (via load_dataset), not some looser ad-hoc subset."""
    df = _labeled_df(n_experiments=1, n_intervals=4, tmp_path=tmp_path)
    assert set(CSV_COLUMNS).issubset(set(df.columns))
