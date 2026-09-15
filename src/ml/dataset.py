"""Load and validate a modeling dataset from Phase 1/2's raw experiment CSVs.

No Phase 3 exists in this repository to reuse a "combine + validate"
step from (see docs/PHASE_4_ML_MODELING.md Section 1) -- this module is
where Phase 1/2's per-experiment CSVs first get combined into a single
table for modeling, and it deliberately re-runs Phase 2's own row and
leakage validation (network.validation) on the COMBINED result rather
than trusting each file blindly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from network.schema import CSV_COLUMNS
from network.targets import drop_rows_without_target
from network.validation import SchemaValidationError, validate_no_leakage, validate_rows


class DatasetError(RuntimeError):
    """Raised instead of returning a partially-invalid or empty dataset."""


def discover_csv_files(input_path: Path) -> list[Path]:
    """Find raw experiment CSVs: a single file, or every *.csv in a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return []


def load_dataset(csv_paths: list[Path]) -> pd.DataFrame:
    """Load, combine, and validate one or more Phase 1/2 experiment CSVs.

    Raises DatasetError -- never returns a partially-invalid frame -- if:
      - no files are given (the "no real dataset" case)
      - any file is missing a required column
      - the combined data fails Phase 2's row/leakage validation
      - no rows have a usable target after dropping end-of-experiment rows

    Rows without a target (the last interval of each experiment_id, where
    no future measurement exists) are dropped here -- they are structurally
    unusable for supervised training, not an error in the data itself.
    """
    if not csv_paths:
        raise DatasetError(
            "No CSV files provided -- no real dataset exists yet. Refusing to "
            "train on nothing. Run scripts/generate_dataset.py against a real "
            "Mininet environment first (see docs/PHASE_2_DATA_GENERATION.md)."
        )

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = set(CSV_COLUMNS) - set(df.columns)
        if missing:
            raise DatasetError(f"{path}: missing required column(s): {sorted(missing)}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # pandas represents an empty CSV field (our written-out None) as NaN.
    # DataFrame.where(mask, None) on its own is NOT enough to fix this: a
    # float64 column cannot hold a distinct None, so pandas silently
    # coerces it straight back to NaN, and Phase 2's validators check
    # `is None` (not `pd.isna`) -- so NaN would slip past that check
    # undetected. astype(object) first forces every column to a dtype
    # that CAN hold a real None, so the substitution actually sticks.
    null_mask = combined.notnull()
    combined = combined.astype(object).where(null_mask, None)

    rows = combined.to_dict("records")
    try:
        validate_rows(rows, CSV_COLUMNS[:-1])
        validate_no_leakage(rows)
    except SchemaValidationError as exc:
        raise DatasetError(f"Combined dataset failed validation: {exc}") from exc

    labeled_rows = drop_rows_without_target(rows)
    if not labeled_rows:
        raise DatasetError(
            "No rows have a usable target after dropping end-of-experiment rows -- "
            "dataset is too small to train on (need at least 2 intervals per experiment)."
        )

    return pd.DataFrame(labeled_rows, columns=CSV_COLUMNS)
