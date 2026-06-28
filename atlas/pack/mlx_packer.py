"""Packaging for quantized MLX models with Atlas metadata.

Bundles a quantized model's files together with a ``metadata.json``
describing the quantization, evaluation, and hardware context under
which the package was produced.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from atlas.quant.mlx_quantizer import QuantResult
from atlas.eval.perplexity import EvalResult
from atlas.profile.hardware import HardwareSpec

ATLAS_VERSION = "0.1.0"


@dataclass(frozen=True)
class PackageInfo:
    output_path: Path
    total_size_mb: float
    metadata: dict


class MLXPacker:
    """Packages a quantized MLX model directory with Atlas metadata."""

    def __init__(self, output_base: Optional[Path] = None):
        self._output_base = output_base or Path.cwd() / "atlas-output"

    def package(
        self,
        quantized_path: Path,
        model_id: str,
        quant_result: QuantResult,
        eval_result: EvalResult,
        hardware: HardwareSpec,
    ) -> PackageInfo:
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        dir_name = f"{model_name}-{quant_result.bits}bit-atlas"
        output_dir = self._output_base / dir_name

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        for f in quantized_path.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)

        metadata = {
            "atlas_version": ATLAS_VERSION,
            "model_id": model_id,
            "bits": quant_result.bits,
            "group_size": quant_result.group_size,
            "original_size_mb": quant_result.original_size_mb,
            "quantized_size_mb": quant_result.quantized_size_mb,
            "compression_ratio": round(
                quant_result.original_size_mb / max(quant_result.quantized_size_mb, 0.01), 2
            ),
            "ppl_baseline": eval_result.ppl_baseline,
            "ppl_quantized": eval_result.ppl_quantized,
            "ppl_delta_pct": eval_result.ppl_delta_pct,
            "eval_samples": eval_result.num_samples,
            "hardware": {
                "chip": hardware.chip,
                "ram_total_gb": hardware.ram_total_gb,
                "platform": hardware.platform,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
        total_size_mb = round(total_size / (1024 * 1024), 1)

        return PackageInfo(
            output_path=output_dir,
            total_size_mb=total_size_mb,
            metadata=metadata,
        )
