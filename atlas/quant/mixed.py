"""Mixed-precision MLX quantization for Atlas.

Wraps ``mlx_lm.convert`` with a custom ``quant_predicate`` that applies a
per-layer bit-width allocation produced by Task 2's QuantPlanner. Unlike
MLXQuantizer (uniform N-bit quantization), MixedQuantizer assigns different
bit widths to different transformer layers based on their measured
sensitivity, trading a small amount of complexity for better quality-per-byte.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from mlx_lm import convert as mlx_lm_convert

from atlas.plan.planner import QuantPlan


CACHE_DIR = Path.home() / ".cache" / "atlas"

_LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class MixedQuantResult:
    output_path: Path
    plan: QuantPlan
    quantized_size_mb: float
    original_size_mb: float


def _build_predicate(plan: QuantPlan):
    """Builds a quant_predicate closure that looks up bits by layer index.

    - Submodules under ``model.layers.{i}.*`` get the bit width planned for
      layer ``i``.
    - ``lm_head`` always gets the highest bit width in the plan, since the
      output projection is disproportionately sensitive to quantization
      error and is small relative to the rest of the model.
    - Anything else (e.g. embeddings) falls back to the plan's target_bits.
    """
    layer_bits = {
        lp.layer_index: {"bits": lp.bits, "group_size": lp.group_size, "mode": "affine"}
        for lp in plan.layers
    }

    max_bits = max((lp.bits for lp in plan.layers), default=plan.target_bits)
    default_config = {"bits": plan.target_bits, "group_size": 64, "mode": "affine"}
    head_config = {"bits": max_bits, "group_size": 64, "mode": "affine"}

    def predicate(path: str, module) -> Union[bool, dict]:
        if "lm_head" in path:
            return head_config

        match = _LAYER_PATTERN.search(path)
        if match:
            idx = int(match.group(1))
            if idx in layer_bits:
                return layer_bits[idx]

        return default_config

    return predicate


class MixedQuantizer:
    """Quantizes a Hugging Face model to mixed-precision MLX format.

    Applies a per-layer QuantPlan (from Task 2's QuantPlanner) via
    ``mlx_lm.convert``'s ``quant_predicate`` hook, rather than a single
    uniform bit width.
    """

    def quantize(
        self,
        model_id: str,
        plan: QuantPlan,
        output_dir: Path | None = None,
    ) -> MixedQuantResult:
        if output_dir is None:
            safe_name = model_id.replace("/", "_")
            output_dir = CACHE_DIR / safe_name / f"mlx-mixed-avg{plan.avg_bits}bit"

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        predicate = _build_predicate(plan)

        mlx_lm_convert(
            hf_path=model_id,
            mlx_path=str(output_dir),
            quantize=True,
            quant_predicate=predicate,
        )

        quantized_size_mb = _dir_size_mb(output_dir)
        original_size_mb = (
            quantized_size_mb * (16 / plan.avg_bits) if plan.avg_bits > 0 else 0.0
        )

        return MixedQuantResult(
            output_path=output_dir,
            plan=plan,
            quantized_size_mb=round(quantized_size_mb, 1),
            original_size_mb=round(original_size_mb, 1),
        )


def _dir_size_mb(directory: Path) -> float:
    total = sum(f.stat().st_size for f in directory.glob("*.safetensors"))
    return total / (1024 * 1024)
