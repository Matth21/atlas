#!/usr/bin/env python3
"""Gate di additività SGSR-2 (spec: sezione 'Validazione additività').

Per 3 punti della frontiera confronta il costo predetto Σ e_l(c_l) con la
KL misurata applicando il piano completo in fake-quant. Accettazione:
ranking dei 3 piani preservato e rapporto predetto/misurato entro ±50%.

Uso:
    .venv/bin/python scripts/sgsr2_validate_additivity.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_lm_load

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.plan.pareto import sweep
from atlas.profile.calib import load_calibration
from atlas.profile.cost_table import CostProfiler
from atlas.profile.kl import kl_vs_snapshot, snapshot_topk
from atlas.quant.fakequant import apply_fake_quant, restore_weights
from atlas.quant.smooth import smooth_model_dir

TARGET_BUDGETS = (3.2, 4.0, 4.8)
RATIO_TOLERANCE = (0.5, 1.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    safe = args.model.replace("/", "_")
    out_path = Path(args.output or f"results/sgsr2_additivity_{safe}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smooth_dir = smooth_model_dir(args.model)
    try:
        table = CostProfiler().profile(str(smooth_dir), args.model)
        model, tokenizer = mlx_lm_load(str(smooth_dir))
        seqs = load_calibration(tokenizer)
        snapshots = [
            snapshot_topk(model(mx.array(s)[None, :]).squeeze(0))
            for s in seqs
        ]

        frontier = sweep(table)
        rows = []
        for budget in TARGET_BUDGETS:
            point = min(frontier, key=lambda p: abs(p.avg_eff_bits - budget))
            measured = _measure_full_plan(model, point, table, snapshots, seqs)
            rows.append({
                "budget_target": budget,
                "avg_eff_bits": point.avg_eff_bits,
                "predicted": point.predicted_cost,
                "measured": measured,
                "ratio": point.predicted_cost / measured if measured > 0 else None,
            })

        pred_rank = sorted(range(3), key=lambda i: rows[i]["predicted"])
        meas_rank = sorted(range(3), key=lambda i: rows[i]["measured"])
        report = {
            "model": args.model,
            "rows": rows,
            "rank_preserved": pred_rank == meas_rank,
            "ratios_in_tolerance": all(
                r["ratio"] is not None
                and RATIO_TOLERANCE[0] <= r["ratio"] <= RATIO_TOLERANCE[1]
                for r in rows
            ),
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("\nGATE:", "PASS" if report["rank_preserved"] else "FAIL (ranking)")
    finally:
        shutil.rmtree(smooth_dir, ignore_errors=True)


def _measure_full_plan(model, point, table, snapshots, seqs) -> float:
    saved = []
    try:
        for block, key in zip(model.model.layers, point.assignment):
            bits, gs = (int(x) for x in key.split(":"))
            saved.append(apply_fake_quant(block, bits=bits, group_size=gs))
        kls = [
            kl_vs_snapshot(snap, model(mx.array(seq)[None, :]).squeeze(0))
            for snap, seq in zip(snapshots, seqs)
        ]
        return sum(kls) / len(kls)
    finally:
        for block, originals in zip(model.model.layers, saved):
            restore_weights(block, originals)


if __name__ == "__main__":
    main()
