import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from atlas.quant.mlx_quantizer import MLXQuantizer, QuantResult


class TestQuantResult:
    def test_quant_result_fields(self):
        result = QuantResult(
            output_path=Path("/tmp/test"),
            bits=4,
            group_size=64,
            original_size_mb=2048.0,
            quantized_size_mb=512.0,
        )
        assert result.bits == 4
        assert result.group_size == 64
        assert result.original_size_mb == 2048.0
        assert result.quantized_size_mb == 512.0
        assert result.output_path == Path("/tmp/test")

    def test_quant_result_is_frozen(self):
        result = QuantResult(
            output_path=Path("/tmp/test"),
            bits=4,
            group_size=64,
            original_size_mb=2048.0,
            quantized_size_mb=512.0,
        )
        with pytest.raises(AttributeError):
            result.bits = 8


class TestMLXQuantizer:
    def test_invalid_bits_raises(self):
        q = MLXQuantizer()
        with pytest.raises(ValueError, match="bits must be one of"):
            q.quantize("some/model", bits=3)

    def test_invalid_group_size_raises(self):
        q = MLXQuantizer()
        with pytest.raises(ValueError, match="group_size must be one of"):
            q.quantize("some/model", bits=4, group_size=99)

    @patch("atlas.quant.mlx_quantizer.mlx_lm_convert")
    def test_quantize_calls_mlx_lm_convert(self, mock_convert, tmp_path):
        output_dir = tmp_path / "quantized"
        output_dir.mkdir()
        # Create fake safetensors file so size calculation works
        fake_weights = output_dir / "model.safetensors"
        fake_weights.write_bytes(b"\x00" * 1024)

        q = MLXQuantizer()
        result = q.quantize(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            bits=4,
            group_size=64,
            output_dir=output_dir,
        )

        mock_convert.assert_called_once_with(
            hf_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            mlx_path=str(output_dir),
            quantize=True,
            q_bits=4,
            q_group_size=64,
        )
        assert result.bits == 4
        assert result.group_size == 64
        assert result.output_path == output_dir

    @patch("atlas.quant.mlx_quantizer.mlx_lm_convert")
    def test_quantize_default_output_dir(self, mock_convert, tmp_path):
        q = MLXQuantizer()
        # Patch the cache dir to use tmp_path
        with patch("atlas.quant.mlx_quantizer.CACHE_DIR", tmp_path):
            cache_subdir = tmp_path / "TinyLlama_TinyLlama-1.1B-Chat-v1.0" / "mlx-4bit-g64"
            cache_subdir.mkdir(parents=True)
            fake_weights = cache_subdir / "model.safetensors"
            fake_weights.write_bytes(b"\x00" * 512)

            result = q.quantize(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                bits=4,
                group_size=64,
            )
            assert "mlx-4bit-g64" in str(result.output_path)
