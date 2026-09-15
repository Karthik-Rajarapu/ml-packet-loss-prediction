import pandas as pd
import pytest

from ml.dataset_upload import (
    DatasetUploadError,
    MIN_ROWS_FOR_TRAINING,
    read_uploaded_csv,
    summarize_raw_dataset,
)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _sample_df(n=10) -> pd.DataFrame:
    return pd.DataFrame({
        "rtt": [10.0 + i for i in range(n)],
        "loss": [0.0, 1.0] * (n // 2),
        "note": ["x"] * n,
    })


def test_read_uploaded_csv_rejects_non_csv_extension():
    with pytest.raises(DatasetUploadError, match="Only .csv"):
        read_uploaded_csv(b"a,b\n1,2\n", "data.txt")


def test_read_uploaded_csv_rejects_empty_file():
    with pytest.raises(DatasetUploadError, match="empty"):
        read_uploaded_csv(b"", "data.csv")


def test_read_uploaded_csv_rejects_malformed_content():
    with pytest.raises(DatasetUploadError, match="could not be read"):
        read_uploaded_csv(b"\x00\x01\x02\x03not,a,csv\xff\xfe", "data.csv")


def test_read_uploaded_csv_rejects_too_few_rows():
    tiny = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(DatasetUploadError, match="too few rows"):
        read_uploaded_csv(_csv_bytes(tiny), "data.csv")


def test_read_uploaded_csv_rejects_oversized_file(monkeypatch):
    from ml import dataset_upload as du
    monkeypatch.setattr(du, "MAX_FILE_SIZE_BYTES", 50)  # small threshold, so a tiny payload trips it
    with pytest.raises(du.DatasetUploadError, match="too large"):
        du.read_uploaded_csv(b"a,b\n" + b"1,2\n" * 20, "data.csv")


def test_read_uploaded_csv_accepts_valid_csv():
    df = _sample_df(MIN_ROWS_FOR_TRAINING + 2)
    result = read_uploaded_csv(_csv_bytes(df), "data.csv")
    assert len(result) == len(df)
    assert list(result.columns) == list(df.columns)


def test_summarize_raw_dataset_reports_real_values_not_hardcoded():
    df = _sample_df(12)
    df.loc[0, "rtt"] = None
    df.loc[1, "rtt"] = None
    summary = summarize_raw_dataset(df, "myfile.csv")
    assert summary.filename == "myfile.csv"
    assert summary.n_rows == 12
    assert summary.n_columns == 3
    assert summary.missing_value_counts["rtt"] == 2
    assert summary.total_missing_values == 2
    assert "rtt" in summary.numeric_columns
    assert "note" in summary.categorical_columns


def test_summarize_raw_dataset_detects_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    summary = summarize_raw_dataset(df, "dupes.csv")
    assert summary.duplicate_row_count == 1


def test_summarize_raw_dataset_detects_likely_timestamp_column():
    df = pd.DataFrame({"event_timestamp": [1, 2, 3], "value": [1, 2, 3]})
    summary = summarize_raw_dataset(df, "f.csv")
    assert summary.likely_timestamp_column == "event_timestamp"


def test_summarize_raw_dataset_no_timestamp_column_returns_none():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    summary = summarize_raw_dataset(df, "f.csv")
    assert summary.likely_timestamp_column is None


def test_summarize_raw_dataset_detects_likely_target_column():
    df = pd.DataFrame({"packet_loss_pct": [0.0, 1.0], "x": [1, 2]})
    summary = summarize_raw_dataset(df, "f.csv")
    assert summary.likely_target_column == "packet_loss_pct"


def test_summarize_raw_dataset_preview_is_limited():
    df = _sample_df(50)
    summary = summarize_raw_dataset(df, "f.csv", preview_rows=5)
    assert len(summary.preview) == 5
