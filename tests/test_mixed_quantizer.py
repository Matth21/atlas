import pytest
from unittest.mock import patch
from pathlib import Path

from atlas.quant.mixed import MixedQuantizer, MixedQuantResult, _build_predicate
from atlas.plan.planner import QuantPlan, LayerPlan


def _make_plan() -> QuantPlan:
    layers = (
        LayerPlan(0, "model.layers.0", 8, 64, 1.0),
        LayerPlan(1, "model.layers.1", 4, 64, 0.5),
        LayerPlan(2, "model.layers.2", 2, 64, 0.0),
    )
    return QuantPlan(
        model_id="test/model", layers=layers,
        avg_bits=4.67, estimated_size_gb=0.5, target_bits=4,
    )


class TestBuildPredicate:
    def test_predicate_returns_correct_bits_per_layer(self):
        plan = _make_plan()
        pred = _build_predicate(plan)

        # Layer 0 submodule -> 8-bit
        result = pred("model.layers.0.mlp.down_proj", None)
        assert result["bits"] == 8

        # Layer 1 submodule -> 4-bit
        result = pred("model.layers.1.self_attn.q_proj", None)
        assert result["bits"] == 4

        # Layer 2 submodule -> 2-bit
        result = pred("model.layers.2.mlp.gate_proj", None)
        assert result["bits"] == 2

    def test_predicate_lm_head_uses_highest_bits(self):
        plan = _make_plan()
        pred = _build_predicate(plan)
        result = pred("lm_head", None)
        assert result["bits"] == 8

    def test_predicate_unknown_path_uses_target(self):
        plan = _make_plan()
        pred = _build_predicate(plan)
        result = pred("model.embed_tokens", None)
        assert result["bits"] == plan.target_bits

    def test_predicate_returns_dict_with_group_size_and_mode(self):
        plan = _make_plan()
        pred = _build_predicate(plan)
        result = pred("model.layers.1.self_attn.q_proj", None)
        assert result["group_size"] == 64
        assert result["mode"] == "affine"

    def test_predicate_double_digit_layer_index(self):
        layers = (
            LayerPlan(0, "model.layers.0", 8, 64, 1.0),
            LayerPlan(10, "model.layers.10", 2, 64, 0.1),
        )
        plan = QuantPlan(
            model_id="test/model", layers=layers,
            avg_bits=5.0, estimated_size_gb=0.5, target_bits=4,
        )
        pred = _build_predicate(plan)
        result = pred("model.layers.10.mlp.down_proj", None)
        assert result["bits"] == 2


class TestMixedQuantResult:
    def test_fields(self):
        result = MixedQuantResult(
            output_path=Path("/tmp/out"), plan=_make_plan(),
            quantized_size_mb=500.0, original_size_mb=2000.0,
        )
        assert result.quantized_size_mb == 500.0
        assert result.plan.avg_bits == 4.67

    def test_frozen(self):
        result = MixedQuantResult(
            output_path=Path("/tmp/out"), plan=_make_plan(),
            quantized_size_mb=500.0, original_size_mb=2000.0,
        )
        with pytest.raises(AttributeError):
            result.quantized_size_mb = 100.0


class TestMixedQuantizer:
    @patch("atlas.quant.mixed.mlx_lm_convert")
    def test_quantize_passes_predicate(self, mock_convert, tmp_path):
        output_dir = tmp_path / "mixed"
        output_dir.mkdir()
        (output_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        plan = _make_plan()
        q = MixedQuantizer()
        result = q.quantize("test/model", plan, output_dir=output_dir)

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args
        assert call_kwargs.kwargs.get("quantize") is True
        assert callable(call_kwargs.kwargs.get("quant_predicate"))
        assert result.plan == plan

    @patch("atlas.quant.mixed.mlx_lm_convert")
    def test_quantize_passes_hf_path_and_mlx_path(self, mock_convert, tmp_path):
        output_dir = tmp_path / "mixed"
        output_dir.mkdir()
        (output_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        plan = _make_plan()
        q = MixedQuantizer()
        q.quantize("test/model", plan, output_dir=output_dir)

        call_kwargs = mock_convert.call_args
        assert call_kwargs.kwargs.get("hf_path") == "test/model"
        assert call_kwargs.kwargs.get("mlx_path") == str(output_dir)

    @patch("atlas.quant.mixed.mlx_lm_convert")
    def test_quantize_removes_existing_output_dir(self, mock_convert, tmp_path):
        output_dir = tmp_path / "mixed"
        output_dir.mkdir()
        stale_file = output_dir / "stale.txt"
        stale_file.write_text("stale")

        def fake_convert(**kwargs):
            # mlx_lm.convert recreates the dir itself
            Path(kwargs["mlx_path"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["mlx_path"]) / "model.safetensors").write_bytes(b"\x00" * 2048)

        mock_convert.side_effect = fake_convert

        plan = _make_plan()
        q = MixedQuantizer()
        q.quantize("test/model", plan, output_dir=output_dir)

        assert not stale_file.exists()

    @patch("atlas.quant.mixed.mlx_lm_convert")
    def test_quantize_computes_sizes(self, mock_convert, tmp_path):
        output_dir = tmp_path / "mixed"

        def fake_convert(**kwargs):
            Path(kwargs["mlx_path"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["mlx_path"]) / "model.safetensors").write_bytes(
                b"\x00" * (1024 * 1024)
            )

        mock_convert.side_effect = fake_convert

        plan = _make_plan()
        q = MixedQuantizer()
        result = q.quantize("test/model", plan, output_dir=output_dir)

        assert result.quantized_size_mb == pytest.approx(1.0, rel=0.01)
        assert result.original_size_mb > result.quantized_size_mb

    @patch("atlas.quant.mixed.mlx_lm_convert")
    def test_quantize_default_output_dir(self, mock_convert, tmp_path):
        def fake_convert(**kwargs):
            Path(kwargs["mlx_path"]).mkdir(parents=True, exist_ok=True)
            (Path(kwargs["mlx_path"]) / "model.safetensors").write_bytes(b"\x00" * 512)

        mock_convert.side_effect = fake_convert

        plan = _make_plan()
        q = MixedQuantizer()
        with patch("atlas.quant.mixed.CACHE_DIR", tmp_path):
            result = q.quantize("test/model", plan)
            assert "mlx-mixed" in str(result.output_path)
            assert result.output_path.exists()
