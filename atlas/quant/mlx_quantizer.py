"""MLX-based N-bit model quantization for Atlas.

Wraps ``mlx_lm.convert`` to quantize a Hugging Face model into the MLX
format at a chosen bit width and group size, caching the result on disk
and reporting the resulting size reduction.
"""

from dataclasses import dataclass
from pathlib import Path

from mlx_lm import convert as mlx_lm_convert


CACHE_DIR = Path.home() / ".cache" / "atlas"
VALID_BITS = {2, 4, 8}
VALID_GROUP_SIZES = {32, 64, 128}


@dataclass(frozen=True)
class QuantResult:
    output_path: Path
    bits: int
    group_size: int
    original_size_mb: float
    quantized_size_mb: float


class MLXQuantizer:
    """Quantizes Hugging Face models to N-bit MLX format via mlx-lm."""

    def quantize(
        self,
        model_id: str,
        bits: int = 4,
        group_size: int = 64,
        output_dir: Path | None = None,
    ) -> QuantResult:
        if bits not in VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(VALID_BITS)}, got {bits}")
        if group_size not in VALID_GROUP_SIZES:
            raise ValueError(
                f"group_size must be one of {sorted(VALID_GROUP_SIZES)}, got {group_size}"
            )

        if output_dir is None:
            safe_name = model_id.replace("/", "_")
            output_dir = CACHE_DIR / safe_name / f"mlx-{bits}bit-g{group_size}"
        output_dir.mkdir(parents=True, exist_ok=True)

        mlx_lm_convert(
            hf_path=model_id,
            mlx_path=str(output_dir),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
        )

        quantized_size_mb = _dir_size_mb(output_dir, "*.safetensors")
        original_size_mb = quantized_size_mb * (16 / bits)

        return QuantResult(
            output_path=output_dir,
            bits=bits,
            group_size=group_size,
            original_size_mb=round(original_size_mb, 1),
            quantized_size_mb=round(quantized_size_mb, 1),
        )


def _dir_size_mb(directory: Path, pattern: str) -> float:
    total = sum(f.stat().st_size for f in directory.glob(pattern))
    return total / (1024 * 1024)
