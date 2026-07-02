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


def sweep(table: CostTable, num_lambdas: int = 50) -> list[ParetoPoint]:
    points: dict[tuple[str, ...], ParetoPoint] = {}
    for lam in np.logspace(-8, 2, num_lambdas):
        p = _solve(table, float(lam))
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
        target_bits=4,
    )
