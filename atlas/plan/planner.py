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
    """Greedily allocates bit-widths per layer based on sensitivity.

    Strategy:
    - Start every layer at `target_bits`.
    - Promote the most sensitive layers (top 15%) to the next higher
      bit-width, since quantization error there hurts output quality most.
    - Demote the least sensitive layers (bottom 25%) to the next lower
      bit-width, recovering memory budget where it's cheap to do so.
    - If the resulting plan doesn't fit in `usable_memory_gb`, greedily
      demote the least-sensitive layer that still has headroom until it
      fits (or no further demotion is possible).
    """

    def plan(
        self,
        profile: LayerProfile,
        target_bits: int,
        model_info: ModelInfo,
        usable_memory_gb: float,
        group_size: int = 64,
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

        # Promote top 15% to next higher bit width
        top_count = max(1, int(n * 0.15))
        for s in sorted_by_sens[:top_count]:
            bits_map[s.layer_index] = _promote(target_bits)

        # Demote bottom 25% to next lower bit width
        bottom_count = max(1, int(n * 0.25))
        for s in sorted_by_sens[-bottom_count:]:
            bits_map[s.layer_index] = _demote(target_bits)

        # Check memory fit, demote more if needed
        params_per_layer = model_info.num_params / n if n else 0.0
        while True:
            est_size = _estimate_size(bits_map, params_per_layer)
            if est_size <= usable_memory_gb:
                break
            # Find lowest-sensitivity layer with bits above the floor and demote
            demoted = False
            for s in reversed(sorted_by_sens):
                if bits_map[s.layer_index] > VALID_BITS[0]:
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
                group_size=group_size,
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
