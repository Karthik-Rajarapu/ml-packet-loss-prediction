from network.schema import CSV_COLUMNS, SCHEMA, TARGET_COLUMN


def test_every_column_has_a_spec():
    names = {c.name for c in SCHEMA}
    assert names == set(CSV_COLUMNS)


def test_target_column_is_last():
    """Keeps the CSV writer's column order matching docs; also acts as a
    reminder that the target is appended, never part of raw collection."""
    assert CSV_COLUMNS[-1] == TARGET_COLUMN


def test_target_column_role_is_target():
    target_spec = next(c for c in SCHEMA if c.name == TARGET_COLUMN)
    assert target_spec.role == "target"


def test_no_duplicate_column_names():
    assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS))
