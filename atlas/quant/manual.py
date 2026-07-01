"""Quantizzatore con SmoothQuant / QI-SmoothQuant / AdaptiveSmooth per Atlas.

enable_compensation=False                          → plain MixedQuantizer
enable_compensation=True, qi_mode=False,
  adaptive_alpha=False                             → SmoothQuant (Phase 2.5)
enable_compensation=True, qi_mode=True             → QI-SmoothQuant (Phase 2.8)
enable_compensation=True, adaptive_alpha=True      → AdaptiveSmooth (Phase 2.9, novel)
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from atlas.plan.planner import QuantPlan
from atlas.quant.mixed import MixedQuantizer, CACHE_DIR
from atlas.quant.smooth import smooth_model_dir
from atlas.quant.qi_smooth import qi_smooth_model_dir
from atlas.quant.adaptive_smooth import adaptive_smooth_model_dir


@dataclass(frozen=True)
class ManualQuantResult:
    output_path: Path
    plan: QuantPlan
    quantized_size_mb: float
    original_size_mb: float
    bias_corrections: tuple[mx.array, ...] | None  # sempre None


class ManualLayerQuantizer:
    """Quantizza con SmoothQuant o QI-SmoothQuant (quando enable_compensation=True)."""

    def quantize(
        self,
        model_id: str,
        plan: QuantPlan,
        enable_compensation: bool = True,
        smooth_alpha: float = 0.5,
        qi_mode: bool = False,
        error_lambda: float = 0.3,
        adaptive_alpha: bool = False,
    ) -> ManualQuantResult:
        safe_name = model_id.replace("/", "_")

        if enable_compensation:
            if adaptive_alpha:
                smooth_dir = adaptive_smooth_model_dir(model_id)
                tag = f"mlx-adaptive-smooth-avg{plan.avg_bits}bit"
            elif qi_mode:
                smooth_dir = qi_smooth_model_dir(
                    model_id, alpha=smooth_alpha, error_lambda=error_lambda
                )
                tag = f"mlx-qi-smooth-avg{plan.avg_bits}bit"
            else:
                smooth_dir = smooth_model_dir(model_id, alpha=smooth_alpha)
                tag = f"mlx-smooth-avg{plan.avg_bits}bit"

            output_dir = CACHE_DIR / safe_name / tag
            try:
                mixed_result = MixedQuantizer().quantize(
                    str(smooth_dir), plan, output_dir=output_dir
                )
            finally:
                shutil.rmtree(smooth_dir, ignore_errors=True)
        else:
            mixed_result = MixedQuantizer().quantize(model_id, plan)

        return ManualQuantResult(
            output_path=mixed_result.output_path,
            plan=plan,
            quantized_size_mb=mixed_result.quantized_size_mb,
            original_size_mb=mixed_result.original_size_mb,
            bias_corrections=None,
        )
