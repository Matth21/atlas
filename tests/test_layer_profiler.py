import pytest
from unittest.mock import patch, MagicMock
import mlx.core as mx

from atlas.profile.layers import LayerProfiler, LayerProfile, LayerSensitivity


class TestLayerSensitivity:
    def test_fields(self):
        ls = LayerSensitivity(
            layer_index=0, name="model.layers.0",
            activation_norm=15.3, sensitivity_score=0.85,
        )
        assert ls.layer_index == 0
        assert ls.sensitivity_score == 0.85

    def test_frozen(self):
        ls = LayerSensitivity(
            layer_index=0, name="model.layers.0",
            activation_norm=15.3, sensitivity_score=0.85,
        )
        with pytest.raises(AttributeError):
            ls.sensitivity_score = 0.5


class TestLayerProfile:
    def test_fields(self):
        sens = (
            LayerSensitivity(0, "model.layers.0", 10.0, 0.5),
            LayerSensitivity(1, "model.layers.1", 20.0, 1.0),
        )
        lp = LayerProfile(
            model_id="test/model", num_layers=2,
            sensitivities=sens, calibration_samples=64,
        )
        assert lp.num_layers == 2
        assert len(lp.sensitivities) == 2
        assert lp.sensitivities[1].sensitivity_score == 1.0


class TestLayerProfiler:
    def test_invalid_num_samples(self):
        profiler = LayerProfiler()
        with pytest.raises(ValueError, match="num_samples must be"):
            profiler.profile("model", num_samples=0)

    @patch("atlas.profile.layers._load_calibration_samples")
    @patch("atlas.profile.layers._compute_layer_norms")
    def test_profile_returns_normalized_scores(self, mock_norms, mock_samples):
        mock_samples.return_value = ["sample text"] * 10
        # 4 layers with different activation norms
        mock_norms.return_value = [
            ("model.layers.0", 10.0),
            ("model.layers.1", 30.0),
            ("model.layers.2", 20.0),
            ("model.layers.3", 5.0),
        ]

        profiler = LayerProfiler()
        result = profiler.profile("test/model", num_samples=10)

        assert isinstance(result, LayerProfile)
        assert result.num_layers == 4
        assert result.calibration_samples == 10
        # Layer 1 has highest norm → score 1.0
        scores = {s.layer_index: s.sensitivity_score for s in result.sensitivities}
        assert scores[1] == 1.0
        # Layer 3 has lowest norm → score 0.0
        assert scores[3] == 0.0
        # All scores in [0, 1]
        assert all(0.0 <= s.sensitivity_score <= 1.0 for s in result.sensitivities)
