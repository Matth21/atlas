#!/usr/bin/env python3
"""Atlas — baseline mancanti per il paper SGSR.

Varianti di controllo richieste dalla review:
  uniform_sq       : 4-bit gs=64 uniforme + SmoothQuant (isola il contributo di SQ)
  uniform_gs32_sq  : 4-bit gs=32 uniforme + SmoothQuant (gruppi fini ovunque, niente redistribuzione)
  sgsr_random_sN   : tier SGSR (15/60/25) su ranking casuale seed N + SmoothQuant
  sgsr_inverse     : tier SGSR su ranking entropia invertito + SmoothQuant

Uso:
    python scripts/baseline_ablation.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --seeds 42 43 44 --output results/baseline_ablation_tinyllama.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.eval.perplexity import PerplexityEval
from atlas.plan.planner import (
    QuantPlan,
    LayerPlan,
    GROUP_SIZE_FINE,
    GROUP_SIZE_MID,
    GROUP_SIZE_COARSE,
    SGSR_FINE_FRACTION,
    SGSR_COARSE_FRACTION,
)
from atlas.profile.layers import LayerProfiler
from atlas.quant.manual import ManualLayerQuantizer

# MLX affine: scale bf16 + bias bf16 per gruppo = 32 bit/gruppo.
GROUP_OVERHEAD_BITS = 32.0


def _effective_bits(layers: list[LayerPlan]) -> float:
    """Bit/w effettivi incl. overhead scale/zero, media sui layer."""
    return round(
        sum(lp.bits + GROUP_OVERHEAD_BITS / lp.group_size for lp in layers)
        / len(layers),
        3,
    )


def _uniform_plan(profile, target_bits: int, group_size: int) -> QuantPlan:
    layers = tuple(
        LayerPlan(
            layer_index=s.layer_index,
            name=s.name,
            bits=target_bits,
            group_size=group_size,
            sensitivity_score=s.sensitivity_score,
        )
        for s in profile.sensitivities
    )
    return QuantPlan(
        model_id=profile.model_id,
        layers=layers,
        avg_bits=float(target_bits),
        estimated_size_gb=0.0,
        target_bits=target_bits,
    )


def _sgsr_plan_with_ranking(profile, target_bits: int, ranking) -> QuantPlan:
    """Tier SGSR (fine/mid/coarse) applicati a un ranking arbitrario di layer."""
    n = profile.num_layers
    fine_count = max(1, int(n * SGSR_FINE_FRACTION))
    coarse_count = max(1, int(n * SGSR_COARSE_FRACTION))
    gs_map: dict[int, int] = {}
    for i, s in enumerate(ranking):
        if i < fine_count:
            gs_map[s.layer_index] = GROUP_SIZE_FINE
        elif i >= n - coarse_count:
            gs_map[s.layer_index] = GROUP_SIZE_COARSE
        else:
            gs_map[s.layer_index] = GROUP_SIZE_MID

    layers = tuple(
        LayerPlan(
            layer_index=s.layer_index,
            name=s.name,
            bits=target_bits,
            group_size=gs_map[s.layer_index],
            sensitivity_score=s.sensitivity_score,
        )
        for s in profile.sensitivities
    )
    return QuantPlan(
        model_id=profile.model_id,
        layers=layers,
        avg_bits=float(target_bits),
        estimated_size_gb=0.0,
        target_bits=target_bits,
    )


def build_plans(profile, target_bits: int, seeds: list[int]) -> dict[str, QuantPlan]:
    by_sens_desc = sorted(
        profile.sensitivities, key=lambda s: s.sensitivity_score, reverse=True
    )
    plans = {
        "uniform_sq": _uniform_plan(profile, target_bits, GROUP_SIZE_MID),
        "uniform_gs32_sq": _uniform_plan(profile, target_bits, GROUP_SIZE_FINE),
        "sgsr_inverse": _sgsr_plan_with_ranking(
            profile, target_bits, list(reversed(by_sens_desc))
        ),
    }
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = list(profile.sensitivities)
        rng.shuffle(shuffled)
        plans[f"sgsr_random_s{seed}"] = _sgsr_plan_with_ranking(
            profile, target_bits, shuffled
        )
    return plans


def run_variant(model_id: str, name: str, plan: QuantPlan) -> dict:
    t0 = time.monotonic()
    try:
        quant = ManualLayerQuantizer().quantize(
            model_id, plan, enable_compensation=True, smooth_alpha=0.5
        )
        ev = PerplexityEval().evaluate(quant.output_path, model_id)
        gs_dist = {}
        for lp in plan.layers:
            gs_dist[lp.group_size] = gs_dist.get(lp.group_size, 0) + 1
        return {
            "status": "ok",
            "variant": name,
            "model": model_id,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "ppl_baseline": ev.ppl_baseline,
            "ppl_quantized": ev.ppl_quantized,
            "ppl_delta_pct": ev.ppl_delta_pct,
            "avg_bits": plan.avg_bits,
            "effective_bits": _effective_bits(list(plan.layers)),
            "quantized_size_mb": quant.quantized_size_mb,
            "gs_distribution": gs_dist,
        }
    except Exception as exc:
        return {
            "status": "error",
            "variant": name,
            "model": model_id,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--target-bits", type=int, default=4)
    parser.add_argument("--output", default="results/baseline_ablation.json")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="subset di varianti da eseguire (default: tutte)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = LayerProfiler().profile(args.model, metric="entropy")
    plans = build_plans(profile, args.target_bits, args.seeds)
    if args.variants:
        unknown = set(args.variants) - set(plans)
        if unknown:
            print(f"Varianti sconosciute: {unknown}. Disponibili: {list(plans)}")
            sys.exit(1)
        plans = {k: plans[k] for k in args.variants}

    results = []
    for name, plan in plans.items():
        print(f"\n{'='*60}\nVariante: {name}  |  Modello: {args.model}\n{'='*60}", flush=True)
        r = run_variant(args.model, name, plan)
        results.append(r)
        if r["status"] == "ok":
            print(f"PPL delta: {r['ppl_delta_pct']:.2f}%  |  eff bits: {r['effective_bits']}"
                  f"  |  size: {r['quantized_size_mb']:.0f} MB  |  {r['elapsed_s']:.0f}s", flush=True)
        else:
            print(f"ERRORE: {r['error']}", flush=True)
        output_path.write_text(json.dumps(results, indent=2))

    print(f"\nRisultati in {output_path}")
    print(f"\n{'Variante':<18} {'PPL delta':>10} {'eff bits':>9} {'size MB':>9}")
    print("-" * 50)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['variant']:<18} {r['ppl_delta_pct']:>9.2f}% {r['effective_bits']:>9.3f} "
                  f"{r['quantized_size_mb']:>9.0f}")
        else:
            print(f"{r['variant']:<18} ERROR: {r.get('error', '')[:40]}")


if __name__ == "__main__":
    main()
