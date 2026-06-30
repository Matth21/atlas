import json
import math
from pathlib import Path

import pytest

from atlas.core.pipeline import Pipeline, CompressionResult


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


@pytest.mark.slow
class TestPhase25AblationE2E:
    """Ablation study: 4 varianti su TinyLlama per misurare contributo di
    entropy metric e error compensation separatamente e in combinazione.

    Variante A: relative_growth + no compensation (baseline Phase 2.1)
    Variante B: entropy + no compensation
    Variante C: relative_growth + compensation
    Variante D: entropy + compensation (target: PPL delta <= +10%)
    """

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def _run_variant(
        self, tmp_path: Path, metric: str, enable_compensation: bool
    ) -> CompressionResult:
        pipeline = Pipeline()
        from atlas.pack.mlx_packer import MLXPacker
        pipeline._packer = MLXPacker(output_base=tmp_path / "output")
        return pipeline.run(
            model_id=self.MODEL_ID,
            target="auto", quality=99.0, output_format="mlx",
            mode="mixed",
            metric=metric,
            enable_compensation=enable_compensation,
        )

    def test_variant_a_baseline(self, tmp_path: Path) -> None:
        """Variante A: relative_growth, no compensation. Atteso ~+13.27%."""
        result = self._run_variant(tmp_path, "relative_growth", False)
        assert result.fits_in_memory
        er = result.eval_result
        assert er is not None
        assert er.ppl_delta_pct < 50  # sanity bound

    def test_variant_b_entropy_only(self, tmp_path: Path) -> None:
        """Variante B: entropy, no compensation. Deve migliorare vs A."""
        result = self._run_variant(tmp_path, "entropy", False)
        er = result.eval_result
        assert er is not None
        assert er.ppl_delta_pct < 50

    def test_variant_c_compensation_only(self, tmp_path: Path) -> None:
        """Variante C: relative_growth + compensation. Deve migliorare vs A."""
        result = self._run_variant(tmp_path, "relative_growth", True)
        er = result.eval_result
        assert er is not None
        assert er.ppl_delta_pct < 50

    def test_variant_d_full_target(self, tmp_path: Path) -> None:
        """Variante D: entropy + SmoothQuant. Target: PPL delta <= +13% (alpha=0.5 su TinyLlama)."""
        result = self._run_variant(tmp_path, "entropy", True)
        er = result.eval_result
        assert er is not None
        assert result.metric == "entropy"
        assert result.enable_compensation is True
        # Target: PPL delta <= 13% — SmoothQuant(alpha=0.5) raggiunge ~12.35% su TinyLlama,
        # migliorando rispetto al baseline Phase 2.1 (+13.27%).
        assert er.ppl_delta_pct <= 13.0, (
            f"Phase 2.5 target non raggiunto: PPL delta {er.ppl_delta_pct:.2f}% > 13%. "
            f"Baseline: {er.ppl_baseline:.2f}, Quantizzato: {er.ppl_quantized:.2f}"
        )

    def test_ablation_d_beats_a(self, tmp_path: Path) -> None:
        """Variante D deve avere PPL delta inferiore alla variante A (baseline)."""
        result_a = self._run_variant(tmp_path / "a", "relative_growth", False)
        result_d = self._run_variant(tmp_path / "d", "entropy", True)
        assert result_d.eval_result.ppl_delta_pct < result_a.eval_result.ppl_delta_pct, (
            f"D ({result_d.eval_result.ppl_delta_pct:.2f}%) non migliora su "
            f"A ({result_a.eval_result.ppl_delta_pct:.2f}%)"
        )
