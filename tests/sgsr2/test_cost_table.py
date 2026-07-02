import json
from pathlib import Path

import mlx.core as mx

from atlas.profile.cost_table import CONFIGS, CostProfiler, CostTable
from atlas.profile.kl import kl_vs_snapshot, snapshot_topk
from tests.sgsr2.toy_model import ToyModel


def _profiled(tmp_path=None):
    mx.random.seed(0)
    model = ToyModel(vocab=64, dim=128, n_blocks=3)
    seqs = [[i % 64 for i in range(s, s + 32)] for s in range(4)]
    table = CostProfiler().profile_model(
        model, "toy", seqs,
        checkpoint_path=(tmp_path / "ckpt.json") if tmp_path else None,
    )
    return model, seqs, table


def test_table_shape_and_nonnegative():
    _, _, table = _profiled()
    assert len(table.block_costs) == 3
    assert len(table.configs) == len(CONFIGS)
    for costs in table.block_costs:
        assert set(costs) == set(table.configs)
        assert all(v >= 0.0 for v in costs.values())
    assert all(p > 0 for p in table.block_params)


def test_more_bits_never_much_worse():
    _, _, table = _profiled()
    for costs in table.block_costs:
        for gs in (32, 64, 128):
            assert costs[f"6:{gs}"] <= costs[f"3:{gs}"] + 1e-4


def test_model_restored_after_profiling():
    model, seqs, _ = _profiled()
    logits = model(mx.array(seqs[0])[None, :]).squeeze(0)
    snap = snapshot_topk(logits)
    assert abs(kl_vs_snapshot(snap, logits)) < 1e-4  # sanity
    # ri-profila: se i pesi fossero corrotti, i costi cambierebbero
    table2 = CostProfiler().profile_model(model, "toy", seqs)
    table1 = CostProfiler().profile_model(model, "toy", seqs)
    assert table1.block_costs == table2.block_costs


def test_checkpoint_resume(tmp_path):
    model, seqs, table = _profiled(tmp_path)
    ckpt = tmp_path / "ckpt.json"
    assert ckpt.exists()
    # riprofilare con checkpoint esistente riusa le righe già calcolate
    table2 = CostProfiler().profile_model(model, "toy", seqs, checkpoint_path=ckpt)
    assert table2.block_costs == table.block_costs


def test_json_roundtrip():
    _, _, table = _profiled()
    again = CostTable.from_json(table.to_json())
    assert again == table
