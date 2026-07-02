#!/usr/bin/env python3
"""SGSR-2 end-to-end: cost table → piani Pareto → quantizzazione → PPL.

Include la curva uniforme di riferimento (uniform+SQ a vari bits/gs) con lo
stesso protocollo di eval. Bit/w effettivi calcolati dalla size reale su
disco: accounting identico per tutti i metodi.

Uso:
    .venv/bin/python scripts/sgsr2_pareto.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --budgets 3.0 3.25 3.5 4.0 4.5
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from mlx_lm import load as mlx_lm_load

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.eval.sliding_ppl import ppl_with_ci, sliding_nlls, wikitext2_test_tokens
from atlas.plan.pareto import allocate, to_quant_plan
from atlas.plan.planner import LayerPlan, QuantPlan
from atlas.profile.cost_table import CostProfiler
from atlas.quant.mixed import MixedQuantizer
from atlas.quant.smooth import smooth_model_dir

UNIFORM_REFERENCE = [  # (bits, gs) — curva uniforme a pari protocollo
    (3, 32), (3, 64), (3, 128),
    (4, 32), (4, 64), (4, 128),
    (5, 64),
]


def _uniform_plan(model_id: str, n_blocks: int, bits: int, gs: int) -> QuantPlan:
    layers = tuple(
        LayerPlan(
            layer_index=i, name=f"model.layers.{i}",
            bits=bits, group_size=gs, sensitivity_score=0.0,
        )
        for i in range(n_blocks)
    )
    return QuantPlan(
        model_id=model_id, layers=layers, avg_bits=float(bits),
        estimated_size_gb=0.0, target_bits=bits,
    )


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*.safetensors")) / 1024**2


def _eval_model(model_path: str, test_tokens: list[int]) -> dict:
    model, _ = mlx_lm_load(model_path)
    nlls = sliding_nlls(model, test_tokens)
    ppl, lo, hi = ppl_with_ci(nlls)
    return {"ppl": ppl, "ci_low": lo, "ci_high": hi}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[3.0, 3.25, 3.5, 4.0, 4.5])
    parser.add_argument("--skip-uniform", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    safe = args.model.replace("/", "_")
    out_path = Path(args.output or f"results/sgsr2_pareto_{safe}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(f"atlas-output/sgsr2/{safe}")
    work_dir.mkdir(parents=True, exist_ok=True)

    _, tokenizer = mlx_lm_load(args.model)
    test_tokens = wikitext2_test_tokens(tokenizer)

    print("Baseline BF16 PPL (sliding window)...", flush=True)
    baseline = _eval_model(args.model, test_tokens)
    results = {"model": args.model, "baseline": baseline, "runs": []}
    out_path.write_text(json.dumps(results, indent=2))

    smooth_dir = smooth_model_dir(args.model)
    try:
        table = CostProfiler().profile(str(smooth_dir), args.model)
        n_blocks = len(table.block_costs)
        n_params = sum(table.block_params)

        jobs = []
        for b in args.budgets:
            try:
                point = allocate(table, b)
                plan = to_quant_plan(table, point)
                jobs.append((f"sgsr2_{b}", plan))
            except ValueError as exc:
                # Record unreachable budget as error run instead of crashing
                error_run = {"name": f"sgsr2_{b}", "status": "error", "error": str(exc)}
                results["runs"].append(error_run)
                out_path.write_text(json.dumps(results, indent=2))
                print(f"\n=== sgsr2_{b} (ALLOCATION ERROR) ===", flush=True)
                print(json.dumps(error_run, indent=2), flush=True)

        if not args.skip_uniform:
            jobs += [(f"uniform_{bits}b_gs{gs}",
                      _uniform_plan(args.model, n_blocks, bits, gs))
                     for bits, gs in UNIFORM_REFERENCE]

        for name, plan in jobs:
            t0 = time.monotonic()
            out_dir = work_dir / name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            print(f"\n=== {name} ===", flush=True)
            try:
                qr = MixedQuantizer().quantize(str(smooth_dir), plan, output_dir=out_dir)
                ev = _eval_model(str(qr.output_path), test_tokens)
                size_mb = _dir_size_mb(Path(qr.output_path))
                run = {
                    "name": name, "status": "ok",
                    **ev,
                    "ppl_delta_pct": round(
                        (ev["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100, 3
                    ),
                    "size_mb": round(size_mb, 1),
                    "eff_bits_from_size": round(size_mb * 1024**2 * 8 / n_params, 3),
                    "plan": [(lp.bits, lp.group_size) for lp in plan.layers],
                    "elapsed_s": round(time.monotonic() - t0, 1),
                }
            except Exception as exc:
                run = {"name": name, "status": "error", "error": str(exc),
                       "elapsed_s": round(time.monotonic() - t0, 1)}
            results["runs"].append(run)
            out_path.write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in run.items() if k != "plan"},
                             indent=2), flush=True)
    finally:
        shutil.rmtree(smooth_dir, ignore_errors=True)

    print(f"\nRisultati in {out_path}")


if __name__ == "__main__":
    main()
