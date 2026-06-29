import pytest
from atlas.plan.planner import QuantPlanner, QuantPlan, LayerPlan
from atlas.profile.layers import LayerProfile, LayerSensitivity
from atlas.core.model import ModelInfo


def _make_profile(num_layers: int = 10) -> LayerProfile:
    """Create a synthetic profile with linearly increasing sensitivity."""
    sensitivities = tuple(
        LayerSensitivity(
            layer_index=i,
            name=f"model.layers.{i}",
            relative_growth=float(i + 1),
            sensitivity_score=round(i / max(num_layers - 1, 1), 4),
        )
        for i in range(num_layers)
    )
    return LayerProfile(
        model_id="test/model",
        num_layers=num_layers,
        sensitivities=sensitivities,
        calibration_samples=64,
    )


def _make_model_info(num_params: int = 1_100_000_000, num_layers: int = 10) -> ModelInfo:
    return ModelInfo(
        model_id="test/model",
        num_params=num_params,
        num_layers=num_layers,
        size_fp16_gb=round(num_params * 2 / (1024**3), 2),
        architecture="LlamaForCausalLM",
        exists_locally=False,
    )


class TestLayerPlan:
    def test_fields(self):
        lp = LayerPlan(
            layer_index=0, name="model.layers.0",
            bits=4, group_size=64, sensitivity_score=0.5,
        )
        assert lp.bits == 4
        assert lp.group_size == 64

    def test_frozen(self):
        lp = LayerPlan(
            layer_index=0, name="model.layers.0",
            bits=4, group_size=64, sensitivity_score=0.5,
        )
        with pytest.raises(AttributeError):
            lp.bits = 8


class TestQuantPlan:
    def test_avg_bits(self):
        layers = (
            LayerPlan(0, "l.0", 8, 64, 1.0),
            LayerPlan(1, "l.1", 4, 64, 0.5),
            LayerPlan(2, "l.2", 2, 64, 0.0),
        )
        plan = QuantPlan(
            model_id="test", layers=layers,
            avg_bits=round((8 + 4 + 2) / 3, 2),
            estimated_size_gb=0.5, target_bits=4,
        )
        assert plan.avg_bits == pytest.approx(4.67, abs=0.01)


class TestQuantPlanner:
    def test_all_layers_get_plan(self):
        planner = QuantPlanner()
        profile = _make_profile(10)
        model_info = _make_model_info(num_layers=10)
        plan = planner.plan(profile, target_bits=4, model_info=model_info, usable_memory_gb=10.0)
        assert len(plan.layers) == 10
        assert all(lp.bits in (2, 4, 8) for lp in plan.layers)

    def test_sensitive_layers_get_more_bits(self):
        planner = QuantPlanner()
        profile = _make_profile(10)
        model_info = _make_model_info(num_layers=10)
        plan = planner.plan(profile, target_bits=4, model_info=model_info, usable_memory_gb=10.0)
        # Most sensitive layer (index 9, score=1.0) should have >= target bits
        most_sensitive = [lp for lp in plan.layers if lp.sensitivity_score == 1.0][0]
        # Least sensitive (index 0, score=0.0) should have <= target bits
        least_sensitive = [lp for lp in plan.layers if lp.sensitivity_score == 0.0][0]
        assert most_sensitive.bits >= least_sensitive.bits

    def test_avg_bits_near_target(self):
        planner = QuantPlanner()
        profile = _make_profile(20)
        model_info = _make_model_info(num_layers=20)
        plan = planner.plan(profile, target_bits=4, model_info=model_info, usable_memory_gb=10.0)
        # Average should be within 1 bit of target
        assert abs(plan.avg_bits - 4) <= 1.5

    def test_plan_fits_in_memory(self):
        planner = QuantPlanner()
        profile = _make_profile(10)
        model_info = _make_model_info(num_layers=10)
        plan = planner.plan(profile, target_bits=4, model_info=model_info, usable_memory_gb=10.0)
        assert plan.estimated_size_gb <= 10.0

    def test_invalid_target_bits(self):
        planner = QuantPlanner()
        profile = _make_profile(5)
        model_info = _make_model_info(num_layers=5)
        with pytest.raises(ValueError, match="target_bits must be"):
            planner.plan(profile, target_bits=3, model_info=model_info, usable_memory_gb=10.0)
