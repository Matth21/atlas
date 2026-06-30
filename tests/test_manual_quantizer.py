import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import mlx.core as mx

from atlas.quant.manual import ManualLayerQuantizer, ManualQuantResult
from atlas.plan.planner import QuantPlan, LayerPlan


def _make_plan(num_layers: int = 2) -> QuantPlan:
    layers = tuple(
        LayerPlan(layer_index=i, name=f"model.layers.{i}", bits=4, group_size=64, sensitivity_score=0.5)
        for i in range(num_layers)
    )
    return QuantPlan(model_id="test/model", layers=layers, avg_bits=4.0,
                     estimated_size_gb=0.1, target_bits=4)


class TestManualQuantResult:
    def test_frozen(self):
        result = ManualQuantResult(
            output_path=Path("/tmp/test"),
            plan=_make_plan(),
            quantized_size_mb=100.0,
            original_size_mb=400.0,
            bias_corrections=None,
        )
        with pytest.raises(AttributeError):
            result.quantized_size_mb = 0.0

    def test_bias_corrections_none_when_disabled(self):
        quantizer = ManualLayerQuantizer()
        plan = _make_plan()

        fake_mixed_result = MagicMock()
        fake_mixed_result.output_path = Path("/tmp/fake")
        fake_mixed_result.quantized_size_mb = 100.0
        fake_mixed_result.original_size_mb = 400.0

        with patch("atlas.quant.manual.MixedQuantizer") as MockMixed:
            MockMixed.return_value.quantize.return_value = fake_mixed_result
            with patch("atlas.quant.manual._compute_bias_corrections", return_value=None):
                result = quantizer.quantize("test/model", plan, enable_compensation=False)

        assert result.bias_corrections is None

    def test_bias_corrections_present_when_enabled(self):
        quantizer = ManualLayerQuantizer()
        plan = _make_plan(num_layers=2)

        fake_mixed_result = MagicMock()
        fake_mixed_result.output_path = Path("/tmp/fake")
        fake_mixed_result.quantized_size_mb = 100.0
        fake_mixed_result.original_size_mb = 400.0

        fake_biases = (mx.zeros((64,)), mx.zeros((64,)))

        with patch("atlas.quant.manual.MixedQuantizer") as MockMixed:
            MockMixed.return_value.quantize.return_value = fake_mixed_result
            with patch("atlas.quant.manual._compute_bias_corrections", return_value=fake_biases):
                result = quantizer.quantize("test/model", plan, enable_compensation=True)

        assert result.bias_corrections is not None
        assert len(result.bias_corrections) == 2
