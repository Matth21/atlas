import shutil
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

    def test_bias_corrections_always_none(self):
        """bias_corrections è sempre None — SmoothQuant bakes corrections nei pesi."""
        result = ManualQuantResult(
            output_path=Path("/tmp/test"),
            plan=_make_plan(),
            quantized_size_mb=100.0,
            original_size_mb=400.0,
            bias_corrections=None,
        )
        assert result.bias_corrections is None


class TestManualLayerQuantizerNoComp:
    def test_no_compensation_calls_mixed_directly(self):
        """enable_compensation=False usa MixedQuantizer direttamente, senza smooth."""
        quantizer = ManualLayerQuantizer()
        plan = _make_plan()

        fake_mixed_result = MagicMock()
        fake_mixed_result.output_path = Path("/tmp/fake")
        fake_mixed_result.quantized_size_mb = 100.0
        fake_mixed_result.original_size_mb = 400.0

        with patch("atlas.quant.manual.MixedQuantizer") as MockMixed, \
             patch("atlas.quant.manual.smooth_model_dir") as mock_smooth:
            MockMixed.return_value.quantize.return_value = fake_mixed_result
            result = quantizer.quantize("test/model", plan, enable_compensation=False)

        mock_smooth.assert_not_called()
        assert result.bias_corrections is None
        assert result.output_path == Path("/tmp/fake")


class TestManualLayerQuantizerWithComp:
    def test_compensation_calls_smooth_model_dir(self):
        """enable_compensation=True chiama smooth_model_dir poi MixedQuantizer."""
        quantizer = ManualLayerQuantizer()
        plan = _make_plan()

        fake_mixed_result = MagicMock()
        fake_mixed_result.output_path = Path("/tmp/smooth_out")
        fake_mixed_result.quantized_size_mb = 90.0
        fake_mixed_result.original_size_mb = 400.0

        fake_smooth_dir = Path("/tmp/atlas_smooth_xyz")

        with patch("atlas.quant.manual.smooth_model_dir", return_value=fake_smooth_dir) as mock_smooth, \
             patch("atlas.quant.manual.MixedQuantizer") as MockMixed, \
             patch("atlas.quant.manual.shutil.rmtree") as mock_rm:
            MockMixed.return_value.quantize.return_value = fake_mixed_result
            result = quantizer.quantize("test/model", plan, enable_compensation=True)

        mock_smooth.assert_called_once_with("test/model", alpha=0.5)
        MockMixed.return_value.quantize.assert_called_once()
        mock_rm.assert_called_once_with(fake_smooth_dir, ignore_errors=True)
        assert result.bias_corrections is None

    def test_smooth_dir_cleaned_on_exception(self):
        """smooth_dir viene eliminata anche se MixedQuantizer solleva eccezione."""
        quantizer = ManualLayerQuantizer()
        plan = _make_plan()
        fake_smooth_dir = Path("/tmp/atlas_smooth_xyz")

        with patch("atlas.quant.manual.smooth_model_dir", return_value=fake_smooth_dir), \
             patch("atlas.quant.manual.MixedQuantizer") as MockMixed, \
             patch("atlas.quant.manual.shutil.rmtree") as mock_rm:
            MockMixed.return_value.quantize.side_effect = RuntimeError("quantize failed")
            with pytest.raises(RuntimeError, match="quantize failed"):
                quantizer.quantize("test/model", plan, enable_compensation=True)

        mock_rm.assert_called_once_with(fake_smooth_dir, ignore_errors=True)
