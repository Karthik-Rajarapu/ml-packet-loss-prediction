#!/usr/bin/env python3
"""Phase 5 Step 6: data-quality report over a REAL (or, for a dry check,
any already-generated) dataset under data/raw/.

Reuses ml.dataset.load_dataset() -- the same validated, leakage-checked
loader Phase 4 training uses -- so this report is describing exactly the
data that would be trained on, not a separately-read copy of it.

    python3 scripts/dataset_report.py --input data/raw

Refuses (prints a clear message, writes nothing) if no CSVs exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.data_quality import build_data_quality_report, save_report  # noqa: E402
from ml.dataset import DatasetError, discover_csv_files, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--basename", type=str, default="dataset_quality")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = discover_csv_files(args.input)
    if not csv_paths:
        print(f"No CSV files found under {args.input}. No real dataset exists yet -- nothing to report on.")
        print("Run scripts/generate_dataset.py against a real Mininet environment first.")
        return 1

    try:
        df = load_dataset(csv_paths)
    except DatasetError as exc:
        print(f"Dataset failed to load/validate: {exc}")
        return 1

    report = build_data_quality_report(df, source_files=[str(p) for p in csv_paths])
    json_path, md_path = save_report(report, args.output_dir, args.basename)

    print(f"Rows: {report.overview.total_rows}, experiments: {report.overview.n_experiment_groups}, "
          f"unique configurations: {report.overview.n_unique_configurations}")
    print(f"Target: min={report.target_summary.min:.4f} max={report.target_summary.max:.4f} "
          f"mean={report.target_summary.mean:.4f} zero-loss={report.target_summary.zero_loss_pct:.2f}%")
    if report.target_summary.nonzero_loss_pct < 5.0:
        print("WARNING: fewer than 5% of intervals have non-zero packet loss -- the dataset may not contain "
              "enough congestion to learn from. Consider adjusting offered_load_factors upward before a full sweep.")
    print(f"Report written to {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
