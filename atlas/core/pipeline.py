from dataclasses import dataclass
from pathlib import Path

from atlas.profile.hardware import HardwareProfiler, HardwareSpec
from atlas.core.model import ModelLoader, ModelInfo
from atlas.quant.mlx_quantizer import MLXQuantizer, QuantResult
from atlas.eval.perplexity import PerplexityEval, EvalResult
from atlas.pack.mlx_packer import MLXPacker, PackageInfo
from atlas.profile.layers import LayerProfiler
from atlas.plan.planner import QuantPlanner, QuantPlan
from atlas.quant.mixed import MixedQuantizer
from atlas.quant.manual import ManualLayerQuantizer


@dataclass(frozen=True)
class CompressionResult:
    model_id: str
    hardware: HardwareSpec
    model_info: ModelInfo
    fits_in_memory: bool
    estimated_bits: float
    estimated_size_gb: float
    quant_result: QuantResult | None = None
    eval_result: EvalResult | None = None
    package_info: PackageInfo | None = None
    quant_plan: QuantPlan | None = None
    metric: str = "relative_growth"
    enable_compensation: bool = False
    sgsr_mode: bool = False
    sgsrq_mode: bool = False
    qi_mode: bool = False
    adaptive_alpha: bool = False


class Pipeline:
    def __init__(self):
        self._profiler = HardwareProfiler()
        self._loader = ModelLoader()
        self._quantizer = MLXQuantizer()
        self._evaluator = PerplexityEval()
        self._packer = MLXPacker()
        self._layer_profiler = LayerProfiler()
        self._planner = QuantPlanner()
        self._mixed_quantizer = MixedQuantizer()
        self._manual_quantizer = ManualLayerQuantizer()

    def run(
        self,
        model_id: str,
        target: str,
        quality: float,
        output_format: str,
        mode: str = "mixed",
        dry_run: bool = False,
        metric: str = "entropy",
        enable_compensation: bool = True,
        smooth_alpha: float = 0.5,
        sgsr_mode: bool = False,
        sgsrq_mode: bool = False,
        qi_mode: bool = False,
        error_lambda: float = 0.3,
        adaptive_alpha: bool = False,
    ) -> CompressionResult:
        hardware = self._profiler.detect()
        usable_gb = self._profiler.usable_memory_gb()
        model_info = self._loader.load_metadata(model_id)

        target_bits = self._estimate_bits(quality)
        estimated_size_gb = (model_info.num_params * target_bits / 8) / (1024 ** 3)
        fits = estimated_size_gb <= usable_gb

        quant_result = None
        eval_result = None
        package_info = None
        quant_plan = None

        if fits and not dry_run:
            if mode == "mixed":
                layer_profile = self._layer_profiler.profile(model_id, metric=metric)
                quant_plan = self._planner.plan(
                    layer_profile, int(target_bits), model_info, usable_gb,
                    sgsr_mode=sgsr_mode,
                    sgsrq_mode=sgsrq_mode,
                )
                manual_result = self._manual_quantizer.quantize(
                    model_id, quant_plan,
                    enable_compensation=enable_compensation,
                    smooth_alpha=smooth_alpha,
                    qi_mode=qi_mode,
                    error_lambda=error_lambda,
                    adaptive_alpha=adaptive_alpha,
                )

                eval_result = self._evaluator.evaluate(
                    manual_result.output_path, model_id,
                    bias_corrections=manual_result.bias_corrections,
                )

                quant_result = QuantResult(
                    output_path=manual_result.output_path,
                    bits=round(quant_plan.avg_bits),
                    group_size=64,
                    original_size_mb=manual_result.original_size_mb,
                    quantized_size_mb=manual_result.quantized_size_mb,
                )

                package_info = self._packer.package(
                    quantized_path=manual_result.output_path,
                    model_id=model_id,
                    quant_result=quant_result,
                    eval_result=eval_result,
                    hardware=hardware,
                    quant_plan=quant_plan,
                )
            else:
                quant_result = self._quantizer.quantize(model_id, bits=int(target_bits))

                eval_result = self._evaluator.evaluate(
                    quant_result.output_path, model_id
                )

                package_info = self._packer.package(
                    quantized_path=quant_result.output_path,
                    model_id=model_id,
                    quant_result=quant_result,
                    eval_result=eval_result,
                    hardware=hardware,
                )

        return CompressionResult(
            model_id=model_id,
            hardware=hardware,
            model_info=model_info,
            fits_in_memory=fits,
            estimated_bits=target_bits,
            estimated_size_gb=round(estimated_size_gb, 2),
            quant_result=quant_result,
            eval_result=eval_result,
            package_info=package_info,
            quant_plan=quant_plan,
            metric=metric,
            enable_compensation=enable_compensation,
            sgsr_mode=sgsr_mode,
            sgsrq_mode=sgsrq_mode,
            qi_mode=qi_mode,
            adaptive_alpha=adaptive_alpha,
        )

    def _estimate_bits(self, quality: float) -> int:
        """Map quality target to MLX-valid bit width (2, 4, or 8)."""
        if quality >= 99.5:
            return 8
        elif quality >= 98.0:
            return 4
        else:
            return 2
