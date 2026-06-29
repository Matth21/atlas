import json
import math
import pytest

from atlas.core.pipeline import Pipeline


@pytest.mark.slow
class TestEndToEnd:
    def test_full_pipeline_tinyllama_4bit(self, tmp_path):
        """Full quantization pipeline on TinyLlama 1.1B at 4-bit."""
        pipeline = Pipeline()
        # Override packer output to tmp_path to avoid polluting the working directory
        from atlas.pack.mlx_packer import MLXPacker
        pipeline._packer = MLXPacker(output_base=tmp_path / "output")

        result = pipeline.run(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target="auto",
            quality=99.0,
            output_format="mlx",
            mode="uniform",
        )

        # Model should fit in memory (1.1B is tiny)
        assert result.fits_in_memory

        # Quantization happened
        qr = result.quant_result
        assert qr is not None
        assert qr.bits == 4
        assert qr.quantized_size_mb > 0
        assert qr.quantized_size_mb < qr.original_size_mb

        # Eval happened and PPL is sane
        er = result.eval_result
        assert er is not None
        assert er.ppl_baseline > 0
        assert er.ppl_quantized > 0
        assert not math.isinf(er.ppl_baseline)
        assert not math.isinf(er.ppl_quantized)
        assert er.ppl_delta_pct < 50  # sanity: should be single-digit %

        # Package produced
        pi = result.package_info
        assert pi is not None
        assert pi.output_path.exists()
        assert (pi.output_path / "metadata.json").exists()

        meta = json.loads((pi.output_path / "metadata.json").read_text())
        assert meta["atlas_version"] == "0.1.0"
        assert meta["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert meta["bits"] == 4

        # Verify safetensors files exist
        safetensors_files = list(pi.output_path.glob("*.safetensors"))
        assert len(safetensors_files) > 0


@pytest.mark.slow
class TestMixedModeE2E:
    def test_mixed_quantization_tinyllama(self, tmp_path):
        """Full mixed quantization on TinyLlama — profile, plan, quantize, eval, pack."""
        pipeline = Pipeline()
        from atlas.pack.mlx_packer import MLXPacker
        pipeline._packer = MLXPacker(output_base=tmp_path / "output")

        result = pipeline.run(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            target="auto",
            quality=99.0,
            output_format="mlx",
            mode="mixed",
        )

        assert result.fits_in_memory
        assert result.quant_plan is not None
        assert len(result.quant_plan.layers) == 22

        # Verify mixed bits — not all layers should have the same bit width
        bit_widths = set(lp.bits for lp in result.quant_plan.layers)
        assert len(bit_widths) > 1, "Mixed mode should produce different bit widths"

        # Eval happened
        er = result.eval_result
        assert er is not None
        assert er.ppl_quantized > 0
        assert er.ppl_delta_pct < 50

        # Package produced
        pi = result.package_info
        assert pi is not None
        assert pi.output_path.exists()

        meta = json.loads((pi.output_path / "metadata.json").read_text())
        assert meta["atlas_version"] == "0.1.0"
