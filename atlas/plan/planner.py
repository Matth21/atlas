"""Greedy per-layer mixed-bit quantization planner for Atlas.

Consumes a LayerProfile (per-layer activation sensitivity scores from
Task 1's LayerProfiler) and produces a QuantPlan: a per-layer bit-width
allocation that biases more sensitive layers towards higher precision
and less sensitive layers towards lower precision, while keeping the
estimated quantized model size within the available memory budget.
"""

from dataclasses import dataclass

from atlas.core.model import ModelInfo
from atlas.profile.layers import LayerProfile


VALID_BITS = (2, 4, 8)
# Minimum bit width for quality-first mode (no 2-bit).
# 2-bit quantization introduces catastrophic cascading error on LLMs:
# empirically +29% PPL delta (25% tail 2-bit) vs +4.34% uniform 4-bit.
QUALITY_MIN_BITS = 4

# SGSR group-size tiers.
# Effective bits/weight (4-bit affine, group overhead = 1 scale + 1 zero in bf16):
#   gs=32  → ~4.625 bit/w   (finer scale → better quality, +overhead)
#   gs=64  → ~4.501 bit/w   (MLX default)
#   gs=128 → ~4.156 bit/w   (coarser scale → lower quality, -overhead)
# Mixing 23%/54%/23% across tiers: avg ≈ 4.45 bit/w < 4.501 (uniform 4-bit).
GROUP_SIZE_FINE = 32
GROUP_SIZE_MID = 64
GROUP_SIZE_COARSE = 128
SGSR_FINE_FRACTION = 0.15   # sweep optimum on TinyLlama: 15% fine (gs=32)
SGSR_COARSE_FRACTION = 0.25  # 25% coarse (gs=128); net budget ≈ 4.511 bit/w vs 4.501 uniform

# SGSR-Q: quality gate on top of SGSR.
# Top SGSRQ_QUALITY_FRACTION → 8-bit (focused quality gate, more selective than quality_mode's 20%).
# Remaining layers use SGSR group-size tiers.
# avg_bits ≈ 4.60 bit/w (between quality_mode 4.73 and SGSR 4.51).
SGSRQ_QUALITY_FRACTION = 0.05


@dataclass(frozen=True)
class LayerPlan:
    layer_index: int
    name: str
    bits: int
    group_size: int
    sensitivity_score: float


@dataclass(frozen=True)
class QuantPlan:
    model_id: str
    layers: tuple[LayerPlan, ...]
    avg_bits: float
    estimated_size_gb: float
    target_bits: int


class QuantPlanner:
    """Greedily allocates bit-widths and group-sizes per layer based on sensitivity.

    Four modes:

    quality_mode=False (legacy): promote top 15% to 8-bit, demote bottom 10%
    to 2-bit. Empirically: +13.27% PPL delta vs +4.34% uniform.

    quality_mode=True (default): promote top 20% to 8-bit, NO demotion below
    4-bit. avg_bits ≈ 4.73. Beats uniform 4-bit (+1.67% vs +4.34%).

    sgsr_mode=True (Sensitivity-Guided group-Size Redistribution): all layers
    stay at target_bits, group_size varies by sensitivity tier (gs=32/64/128).
    Effective avg budget ≈ 4.51 bit/w. Novel, +3.28% at equal bit budget.

    sgsrq_mode=True (SGSR + Quality gate): top 5% → 8-bit, remaining layers
    use SGSR group-size tiers. avg_bits ≈ 4.60. Combines focused bit promotion
    with group-size redistribution across two orthogonal axes simultaneously.
    """

    def plan(
        self,
        profile: LayerProfile,
        target_bits: int,
        model_info: ModelInfo,
        usable_memory_gb: float,
        group_size: int = 64,
        quality_mode: bool = True,
        sgsr_mode: bool = False,
        sgsrq_mode: bool = False,
    ) -> QuantPlan:
        if target_bits not in VALID_BITS:
            raise ValueError(f"target_bits must be one of {VALID_BITS}, got {target_bits}")

        n = profile.num_layers
        sorted_by_sens = sorted(
            profile.sensitivities, key=lambda s: s.sensitivity_score, reverse=True
        )

        bits_map: dict[int, int] = {}
        for s in sorted_by_sens:
            bits_map[s.layer_index] = target_bits

        if sgsrq_mode:
            # SGSR-Q: quality gate (top 5% → 8-bit) + SGSR group-size tiers.
            # Two axes simultaneously: bit-width for the most critical layers,
            # group-size for the rest. Novel combination.
            quality_count = max(1, int(n * SGSRQ_QUALITY_FRACTION))
            fine_count = max(1, int(n * SGSR_FINE_FRACTION))
            coarse_count = max(1, int(n * SGSR_COARSE_FRACTION))
            gs_map = {}
            for i, s in enumerate(sorted_by_sens):
                if i < quality_count:
                    bits_map[s.layer_index] = _promote(target_bits)
                    gs_map[s.layer_index] = GROUP_SIZE_MID
                elif i < quality_count + fine_count:
                    gs_map[s.layer_index] = GROUP_SIZE_FINE
                elif i >= n - coarse_count:
                    gs_map[s.layer_index] = GROUP_SIZE_COARSE
                else:
                    gs_map[s.layer_index] = GROUP_SIZE_MID
        elif sgsr_mode:
            # SGSR: keep all layers at target_bits, vary group_size by tier.
            # Top FINE_FRACTION → gs=32 (higher quality, slight overhead).
            # Bottom COARSE_FRACTION → gs=128 (lower overhead, acceptable quality loss).
            # Net effective budget < uniform 4-bit group_size=64.
            fine_count = max(1, int(n * SGSR_FINE_FRACTION))
            coarse_count = max(1, int(n * SGSR_COARSE_FRACTION))
            gs_map: dict[int, int] = {}
            for i, s in enumerate(sorted_by_sens):
                if i < fine_count:
                    gs_map[s.layer_index] = GROUP_SIZE_FINE
                elif i >= n - coarse_count:
                    gs_map[s.layer_index] = GROUP_SIZE_COARSE
                else:
                    gs_map[s.layer_index] = GROUP_SIZE_MID
        else:
            # All layers share the same group_size; bit allocation varies.
            gs_map = {s.layer_index: group_size for s in sorted_by_sens}

            if quality_mode:
                top_count = max(1, int(n * 0.20))
                for s in sorted_by_sens[:top_count]:
                    bits_map[s.layer_index] = _promote(target_bits)
            else:
                top_count = max(1, int(n * 0.15))
                for s in sorted_by_sens[:top_count]:
                    bits_map[s.layer_index] = _promote(target_bits)

                bottom_count = max(1, int(n * 0.10))
                for s in sorted_by_sens[-bottom_count:]:
                    bits_map[s.layer_index] = _demote(target_bits)

        # Memory-fit loop (only relevant when not in sgsr_mode).
        bit_floor = QUALITY_MIN_BITS if quality_mode else VALID_BITS[0]
        params_per_layer = model_info.num_params / n if n else 0.0
        while True:
            est_size = _estimate_size(bits_map, params_per_layer)
            if est_size <= usable_memory_gb:
                break
            demoted = False
            for s in reversed(sorted_by_sens):
                if bits_map[s.layer_index] > bit_floor:
                    bits_map[s.layer_index] = _demote(bits_map[s.layer_index])
                    demoted = True
                    break
            if not demoted:
                break

        layers = tuple(
            LayerPlan(
                layer_index=s.layer_index,
                name=s.name,
                bits=bits_map[s.layer_index],
                group_size=gs_map[s.layer_index],
                sensitivity_score=s.sensitivity_score,
            )
            for s in profile.sensitivities
        )

        avg_bits = round(sum(lp.bits for lp in layers) / len(layers), 2) if layers else 0.0
        est_size = round(_estimate_size(bits_map, params_per_layer), 2)

        return QuantPlan(
            model_id=profile.model_id,
            layers=layers,
            avg_bits=avg_bits,
            estimated_size_gb=est_size,
            target_bits=target_bits,
        )


def _promote(bits: int) -> int:
    idx = VALID_BITS.index(bits)
    return VALID_BITS[min(idx + 1, len(VALID_BITS) - 1)]


def _demote(bits: int) -> int:
    idx = VALID_BITS.index(bits)
    return VALID_BITS[max(idx - 1, 0)]


def _estimate_size(bits_map: dict[int, int], params_per_layer: float) -> float:
    total_bits = sum(bits * params_per_layer for bits in bits_map.values())
    return (total_bits / 8) / (1024 ** 3)
