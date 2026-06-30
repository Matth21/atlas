"""Quantizzatore con SmoothQuant pre-processing per Atlas Phase 2.5.

Quando enable_compensation=True: applica SmoothQuant (scala pesi pre-quantizzazione)
prima di chiamare MixedQuantizer. Le correzioni sono baked nei pesi quantizzati —
nessuna correzione runtime, nessun cambio di distribuzione durante l'inference.

Quando enable_compensation=False: delega direttamente a MixedQuantizer.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from atlas.plan.planner import QuantPlan
from atlas.quant.mixed import MixedQuantizer, CACHE_DIR
from atlas.quant.smooth import smooth_model_dir


@dataclass(frozen=True)
class ManualQuantResult:
    output_path: Path
    plan: QuantPlan
    quantized_size_mb: float
    original_size_mb: float
    bias_corrections: tuple[mx.array, ...] | None  # sempre None — rimosso bias approach


class ManualLayerQuantizer:
    """Quantizza con SmoothQuant (quando enable_compensation=True) o plain MixedQuantizer."""

    def quantize(
        self,
        model_id: str,
        plan: QuantPlan,
        enable_compensation: bool = True,
        smooth_alpha: float = 0.5,
    ) -> ManualQuantResult:
        safe_name = model_id.replace("/", "_")

        if enable_compensation:
            smooth_dir = smooth_model_dir(model_id, alpha=smooth_alpha)
            output_dir = CACHE_DIR / safe_name / f"mlx-smooth-avg{plan.avg_bits}bit"
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
