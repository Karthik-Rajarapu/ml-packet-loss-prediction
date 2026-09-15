from dashboard.feature_inputs import TRAFFIC_TYPE_OPTIONS, build_feature_input_specs
from ml.demo import DEMO_CURRENT_METRICS
from network.schema import FEATURE_COLUMNS


def test_build_feature_input_specs_covers_every_feature_column():
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    assert [s.name for s in specs] == list(FEATURE_COLUMNS)


def test_traffic_type_uses_select_widget_with_valid_options():
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    traffic_spec = next(s for s in specs if s.name == "traffic_type")
    assert traffic_spec.widget == "select"
    assert traffic_spec.options == TRAFFIC_TYPE_OPTIONS
    assert traffic_spec.default in TRAFFIC_TYPE_OPTIONS


def test_numeric_features_use_number_widget_with_bounds():
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    for spec in specs:
        if spec.name == "traffic_type":
            continue
        assert spec.widget == "number"
        assert isinstance(spec.default, float)
        assert spec.min_value is not None
        assert spec.max_value is not None
        assert spec.min_value <= spec.default <= spec.max_value


def test_percent_bounded_feature_widget_matches_validation_bound():
    """packet_loss_pct is capped at 100 by ml.inference.validate_feature_input
    -- the widget bound must agree, so the UI doesn't offer values the
    engine will just reject."""
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    loss_spec = next(s for s in specs if s.name == "packet_loss_pct")
    assert loss_spec.max_value == 100.0


def test_missing_default_raises_clearly():
    incomplete_defaults = {k: v for k, v in DEMO_CURRENT_METRICS.items() if k != "current_rtt_ms"}
    try:
        build_feature_input_specs(incomplete_defaults)
        raised = False
    except KeyError:
        raised = True
    assert raised, "a defaults dict missing a required feature must fail loudly, not render an incomplete form"


def test_every_spec_unit_and_description_come_from_schema():
    from network.schema import SCHEMA
    schema_by_name = {c.name: c for c in SCHEMA}
    specs = build_feature_input_specs(DEMO_CURRENT_METRICS)
    for spec in specs:
        assert spec.unit == schema_by_name[spec.name].unit
        assert spec.description == schema_by_name[spec.name].description
