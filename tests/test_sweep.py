import pytest

from network.sweep import SweepConfig, build_experiment_id, expand_configs


def test_default_sweep_combination_count():
    sweep = SweepConfig()
    # 2 bw x 2 delay x 2 load factors x 1 flow count = 8
    assert sweep.n_combinations == 8


def test_repetitions_multiply_experiment_count():
    sweep = SweepConfig(repetitions=3)
    assert sweep.n_experiments == sweep.n_combinations * 3


def test_expand_configs_produces_expected_count():
    sweep = SweepConfig(repetitions=2)
    configs = expand_configs(sweep)
    assert len(configs) == sweep.n_experiments


def test_expand_configs_experiment_ids_are_unique():
    sweep = SweepConfig(repetitions=3)
    configs = expand_configs(sweep)
    ids = [c.experiment_id for c in configs]
    assert len(ids) == len(set(ids))


def test_expand_configs_offered_load_relative_to_bandwidth():
    sweep = SweepConfig(
        bottleneck_bw_mbps_values=[2.0],
        bottleneck_delay_ms_values=[0.0],
        offered_load_factors=[0.5, 1.5],
        n_flows_values=[1],
        repetitions=1,
    )
    configs = expand_configs(sweep)
    loads = sorted(c.offered_load_mbps for c in configs)
    assert loads == [1.0, 3.0]  # 2.0 * 0.5, 2.0 * 1.5


def test_expand_configs_is_deterministic():
    sweep = SweepConfig(sweep_id="fixed-id")
    ids_a = [c.experiment_id for c in expand_configs(sweep)]
    ids_b = [c.experiment_id for c in expand_configs(sweep)]
    assert ids_a == ids_b


def test_invalid_dimension_value_rejected_at_sweep_construction():
    """A sweep dimension that would violate ExperimentConfig's own rules
    (e.g. non-positive bandwidth) must still be rejected -- SweepConfig
    checks this itself at construction time (fail as early as possible)."""
    with pytest.raises(ValueError):
        SweepConfig(bottleneck_bw_mbps_values=[0.0])


def test_expand_configs_reuses_experiment_config_validation():
    """duration_s vs sample_interval_s consistency is a rule ExperimentConfig
    enforces, not SweepConfig (SweepConfig accepts this combination fine --
    it has no opinion on it). The failure must come from expand_configs()
    actually constructing ExperimentConfig instances, proving Phase 2 reuses
    Phase 1's validation instead of duplicating or weakening it."""
    sweep = SweepConfig(duration_s=1.0, sample_interval_s=2.0)  # SweepConfig itself allows this
    with pytest.raises(ValueError):
        expand_configs(sweep)  # ExperimentConfig.__post_init__ must reject it


def test_multiple_flows_rejected_clearly():
    with pytest.raises(NotImplementedError):
        SweepConfig(n_flows_values=[1, 2])


def test_empty_dimension_rejected():
    with pytest.raises(ValueError):
        SweepConfig(bottleneck_bw_mbps_values=[])


def test_zero_repetitions_rejected():
    with pytest.raises(ValueError):
        SweepConfig(repetitions=0)


def test_build_experiment_id_is_readable_and_stable():
    eid = build_experiment_id("sweep-x", 2.0, 20.0, 1.3, 1, 0)
    assert eid == "sweep-x__bw2__delay20__load1.3__flows1__rep0"
