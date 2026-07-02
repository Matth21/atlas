import pytest

from atlas.plan.pareto import allocate, effective_bits, sweep, to_quant_plan
from atlas.profile.cost_table import CostTable


def _table():
    # 2 blocchi, 2 config: blocco 0 fragile (costo alto a 3 bit), blocco 1 robusto.
    configs = ("3:128", "6:32")
    return CostTable(
        model_id="toy",
        configs=configs,
        block_costs=(
            {"3:128": 10.0, "6:32": 0.1},
            {"3:128": 0.2, "6:32": 0.1},
        ),
        block_params=(100, 100),
        lmhead_costs=None,
        lmhead_params=0,
        calib_seed=42,
    )


def test_effective_bits():
    assert effective_bits(4, 64) == 4.5
    assert effective_bits(3, 128) == 3.25
    assert effective_bits(6, 32) == 7.0


def test_sweep_extremes():
    pts = sweep(_table(), num_lambdas=30)
    assert pts[0].avg_eff_bits == pytest.approx(3.25)   # λ alto: tutto minimo
    assert pts[-1].avg_eff_bits == pytest.approx(7.0)   # λ→0: tutto massimo
    bits = [p.avg_eff_bits for p in pts]
    assert bits == sorted(bits)


def test_allocate_picks_asymmetric_optimum():
    # budget medio: conviene 6:32 sul blocco fragile, 3:128 sul robusto
    point = allocate(_table(), budget_bits=5.2)
    assert point.assignment == ("6:32", "3:128")
    assert point.avg_eff_bits == pytest.approx((7.0 + 3.25) / 2)


def test_allocate_unreachable_budget_raises():
    with pytest.raises(ValueError, match="3.25"):
        allocate(_table(), budget_bits=3.0)


def test_to_quant_plan():
    point = allocate(_table(), budget_bits=5.2)
    plan = to_quant_plan(_table(), point)
    assert [(lp.bits, lp.group_size) for lp in plan.layers] == [(6, 32), (3, 128)]
    assert plan.layers[0].layer_index == 0
    assert plan.target_bits == 4
