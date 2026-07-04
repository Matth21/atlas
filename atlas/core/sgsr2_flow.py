"""Flusso prodotto SGSR-2: budget in GB → piano ottimo → modello quantizzato.

Percorso: SmoothQuant → CostProfiler (cache) → allocate(budget) → MixedQuantizer.
Il budget in bit/w è derivato dai GB richiesti sui parametri totali; l'allocatore
lavora sulla media pesata dei blocchi (approssimazione ~5%, per eccesso di
prudenza si sottrae un margine fisso).
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from mlx_lm import load as mlx_lm_load

from atlas.plan.pareto import allocate, sweep, to_quant_plan
from atlas.profile.cost_table import CostProfiler
from atlas.quant.mixed import MixedQuantizer
from atlas.quant.smooth import smooth_model_dir

# Margine prudenziale sul budget (embeddings/lm_head fuori dal denominatore blocchi).
BUDGET_SAFETY_BITS = 0.15


@dataclass(frozen=True)
class Sgsr2Result:
    output_path: Path
    budget_gb: float
    budget_bits: float
    plan_bits: float
    quantized_size_mb: float
    original_size_mb: float
    assignment_summary: str


def budget_gb_to_bits(budget_gb: float, num_params: int) -> float:
    return budget_gb * 8 * 1024**3 / num_params


def compress_to_budget(model_id: str, budget_gb: float, num_params: int,
                       output_dir: Path | None = None) -> Sgsr2Result:
    budget_bits = budget_gb_to_bits(budget_gb, num_params) - BUDGET_SAFETY_BITS
    if output_dir is None:
        safe = model_id.replace("/", "_")
        output_dir = Path.home() / ".cache" / "atlas" / safe / f"sgsr2-{budget_gb:g}gb"

    smooth_dir = smooth_model_dir(model_id)
    try:
        table = CostProfiler().profile(str(smooth_dir), model_id)
        point = allocate(table, budget_bits)
        plan = to_quant_plan(table, point)
        result = MixedQuantizer().quantize(str(smooth_dir), plan, output_dir=output_dir)
    finally:
        shutil.rmtree(smooth_dir, ignore_errors=True)

    counts: dict[str, int] = {}
    for key in point.assignment:
        counts[key] = counts.get(key, 0) + 1
    summary = ", ".join(f"{n}×{k.replace(':', 'b/gs')}" for k, n in sorted(counts.items()))

    return Sgsr2Result(
        output_path=result.output_path,
        budget_gb=budget_gb,
        budget_bits=round(budget_bits, 3),
        plan_bits=round(point.avg_eff_bits, 3),
        quantized_size_mb=result.quantized_size_mb,
        original_size_mb=result.original_size_mb,
        assignment_summary=summary,
    )
