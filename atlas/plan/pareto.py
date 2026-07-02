"""Allocazione Lagrangiana (bits, group-size) da CostTable SGSR-2.

Con costi additivi per blocco, min Σ e_l(c_l) sotto vincolo di budget si
risolve per-blocco: c_l(λ) = argmin_c e_l(c) + λ · params_l · eff_bits(c).
Lo sweep di λ traccia l'intera frontiera Pareto.
"""

from dataclasses import dataclass

import numpy as np

from atlas.plan.planner import LayerPlan, QuantPlan
from atlas.profile.cost_table import CostTable

GROUP_OVERHEAD_BITS = 32.0  # scale bf16 + bias bf16 per gruppo (MLX affine)


def effective_bits(bits: int, group_size: int) -> float:
    return bits + GROUP_OVERHEAD_BITS / group_size


def _parse(key: str) -> tuple[int, int]:
    bits, gs = key.split(":")
    return int(bits), int(gs)


@dataclass(frozen=True)
class ParetoPoint:
    lam: float
    avg_eff_bits: float
    predicted_cost: float
    assignment: tuple[str, ...]


def _solve(table: CostTable, lam: float) -> ParetoPoint:
    total_params = sum(table.block_params)
    assignment = []
    cost = 0.0
    weighted_bits = 0.0
    for costs, params in zip(table.block_costs, table.block_params):
        # Tie-break segue l'ordine di inserimento delle config nel dict
        # (deterministico per una CostTable fissa); nessuna policy di
        # tie-break canonica è richiesta qui.
        best = min(
            costs,
            key=lambda k: costs[k] + lam * params * effective_bits(*_parse(k)),
        )
        assignment.append(best)
        cost += costs[best]
        weighted_bits += params * effective_bits(*_parse(best))
    return ParetoPoint(
        lam=lam,
        avg_eff_bits=weighted_bits / total_params,
        predicted_cost=cost,
        assignment=tuple(assignment),
    )


def _lambda_bounds(table: CostTable) -> tuple[float, float]:
    """Range di λ che copre entrambi gli estremi della frontiera.

    Il punto di svolta per un blocco è λ ≈ Δcost / (params·Δeff): sotto
    domina il costo KL, sopra domina la penalità di size. Prendiamo i
    rapporti min/max sui blocchi con margine 1e3 per coprire tutto.
    """
    ratios = []
    for costs, params in zip(table.block_costs, table.block_params):
        span_cost = max(costs.values()) - min(costs.values())
        effs = [effective_bits(*_parse(k)) for k in costs]
        span_eff = (max(effs) - min(effs)) * params
        if span_cost > 0 and span_eff > 0:
            ratios.append(span_cost / span_eff)
    if not ratios:
        return 1e-8, 1e2
    return min(ratios) / 1e3, max(ratios) * 1e3


def sweep(table: CostTable, num_lambdas: int = 50) -> list[ParetoPoint]:
    lo, hi = _lambda_bounds(table)
    lams = [0.0, *np.logspace(np.log10(lo), np.log10(hi), num_lambdas)]
    points: dict[tuple[str, ...], ParetoPoint] = {}
    for lam in lams:
        p = _solve(table, float(lam))
        # avg_eff_bits e predicted_cost sono funzioni pure di assignment
        # (non di λ), quindi assignment duplicati sono punti numericamente
        # identici; cambia solo il lam memorizzato e nulla a valle lo usa.
        points.setdefault(p.assignment, p)
    return sorted(points.values(), key=lambda p: p.avg_eff_bits)


def allocate(table: CostTable, budget_bits: float) -> ParetoPoint:
    pts = sweep(table)
    feasible = [p for p in pts if p.avg_eff_bits <= budget_bits]
    if not feasible:
        raise ValueError(
            f"budget {budget_bits} irraggiungibile: range valido "
            f"[{pts[0].avg_eff_bits:.2f}, {pts[-1].avg_eff_bits:.2f}]"
        )
    return min(feasible, key=lambda p: p.predicted_cost)


def _fallback_bits(table: CostTable, point: ParetoPoint) -> int:
    """Bits di fallback per moduli fuori piano (embeddings): media pesata
    dei bit dei blocchi, arrotondata ai bit validi. Così il fallback segue
    il budget e il confronto con le curve uniformi resta equo."""
    total = sum(table.block_params)
    mean_bits = sum(
        _parse(key)[0] * params
        for key, params in zip(point.assignment, table.block_params)
    ) / total
    return min((3, 4, 5, 6), key=lambda b: abs(b - mean_bits))


def to_quant_plan(table: CostTable, point: ParetoPoint) -> QuantPlan:
    layers = []
    for i, key in enumerate(point.assignment):
        bits, gs = _parse(key)
        layers.append(
            LayerPlan(
                layer_index=i,
                name=f"model.layers.{i}",
                bits=bits,
                group_size=gs,
                sensitivity_score=float(table.block_costs[i][key]),
            )
        )
    avg_bits = round(sum(lp.bits for lp in layers) / len(layers), 2)
    return QuantPlan(
        model_id=table.model_id,
        layers=tuple(layers),
        avg_bits=avg_bits,
        estimated_size_gb=0.0,
        target_bits=_fallback_bits(table, point),
    )
