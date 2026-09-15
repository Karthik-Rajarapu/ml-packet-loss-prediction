"""Streamlit input-widget metadata, derived directly from
network.schema.SCHEMA -- never invents a feature name or unit; this
module only adds UI-presentation hints (widget type, slider bounds) on
top of the single existing source of truth.

Pure Python, no Streamlit import -- fully unit-testable without a
running app.
"""

from __future__ import annotations

from dataclasses import dataclass

from network.schema import FEATURE_COLUMNS, SCHEMA

TRAFFIC_TYPE_OPTIONS: tuple[str, ...] = ("udp", "tcp")


@dataclass(frozen=True)
class FeatureInputSpec:
    name: str
    unit: str
    description: str
    widget: str  # "number" | "select"
    default: float | str
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[str, ...] | None = None


# Widget bounds are a PRESENTATION convenience only -- they keep sliders
# usable. They are deliberately NOT the validation rule: src/ml/inference
# ::validate_feature_input is the sole authority on what's actually
# accepted (e.g. it does not cap bandwidth_utilization_pct at 100, since
# utilization can legitimately exceed 100% under offered load above
# bottleneck capacity -- PHASE_1_NETWORK_EXPERIMENT.md Section 6). A
# value at a slider's max is not guaranteed to be accepted, and the
# dashboard must surface a rejection rather than silently clamp anything.
_WIDGET_HINTS: dict[str, dict[str, float]] = {
    "bottleneck_bw_mbps": {"min_value": 0.1, "max_value": 1000.0, "step": 0.5},
    "bottleneck_delay_ms": {"min_value": 0.0, "max_value": 1000.0, "step": 1.0},
    "bottleneck_queue_pkts": {"min_value": 0.0, "max_value": 1000.0, "step": 1.0},
    "active_connections": {"min_value": 0.0, "max_value": 100.0, "step": 1.0},
    "current_rtt_ms": {"min_value": 0.0, "max_value": 2000.0, "step": 1.0},
    "current_jitter_ms": {"min_value": 0.0, "max_value": 500.0, "step": 0.5},
    "throughput_mbps": {"min_value": 0.0, "max_value": 1000.0, "step": 0.5},
    "bandwidth_utilization_pct": {"min_value": 0.0, "max_value": 500.0, "step": 1.0},
    "packet_rate_pps": {"min_value": 0.0, "max_value": 100_000.0, "step": 10.0},
    "queue_length": {"min_value": 0.0, "max_value": 10_000.0, "step": 1.0},
    "retransmissions": {"min_value": 0.0, "max_value": 100_000.0, "step": 1.0},
    "packets_sent": {"min_value": 0.0, "max_value": 10_000_000.0, "step": 10.0},
    "packets_received": {"min_value": 0.0, "max_value": 10_000_000.0, "step": 10.0},
    "packet_loss_pct": {"min_value": 0.0, "max_value": 100.0, "step": 0.1},
}


def build_feature_input_specs(defaults: dict) -> list[FeatureInputSpec]:
    """One FeatureInputSpec per network.schema.FEATURE_COLUMNS entry, in
    canonical order, pre-filled from `defaults` (the dashboard passes
    ml.demo.DEMO_CURRENT_METRICS -- explicitly synthetic, never real).
    Raises KeyError if `defaults` is missing any required feature, so a
    mismatch between this module and the schema fails loudly rather than
    silently rendering an incomplete form.
    """
    schema_by_name = {c.name: c for c in SCHEMA}
    specs: list[FeatureInputSpec] = []
    for name in FEATURE_COLUMNS:
        col = schema_by_name[name]
        default_value = defaults[name]
        if name == "traffic_type":
            specs.append(FeatureInputSpec(
                name=name, unit=col.unit, description=col.description, widget="select",
                default=str(default_value), options=TRAFFIC_TYPE_OPTIONS,
            ))
        else:
            hints = _WIDGET_HINTS.get(name, {})
            specs.append(FeatureInputSpec(
                name=name, unit=col.unit, description=col.description, widget="number",
                default=float(default_value), **hints,
            ))
    return specs
