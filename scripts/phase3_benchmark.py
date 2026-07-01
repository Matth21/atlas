#!/usr/bin/env python3
"""Atlas Phase 3 — benchmark script per cross-model validation.

Esegue varianti di quantizzazione Atlas su un modello e salva risultati JSON.

Uso:
    python scripts/phase3_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/phase3_benchmark.py --model Qwen/Qwen2.5-7B-Instruct \
        --variants uniform quality sgsr sgsrq qi --output results/phase3_qwen7b.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Assicura che il package atlas sia importabile
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.core.pipeline import Pipeline
from atlas.pack.mlx_packer import MLXPacker

VARIANTS = {
    "uniform": dict(mode="uniform"),
    "quality": dict(mode="mixed", metric="entropy", enable_compensation=True,
                    sgsr_mode=False, sgsrq_mode=False, qi_mode=False, adaptive_alpha=False),
    "sgsr": dict(mode="mixed", metric="entropy", enable_compensation=True,
                 sgsr_mode=True, sgsrq_mode=False, qi_mode=False, adaptive_alpha=False),
    "sgsrq": dict(mode="mixed", metric="entropy", enable_compensation=True,
                  sgsr_mode=False, sgsrq_mode=True, qi_mode=False, adaptive_alpha=False),
    "qi": dict(mode="mixed", metric="entropy", enable_compensation=True,
               sgsr_mode=False, sgsrq_mode=False, qi_mode=True, error_lambda=0.3,
               adaptive_alpha=False),
    "adaptive": dict(mode="mixed", metric="entropy", enable_compensation=True,
                     sgsr_mode=False, sgsrq_mode=False, qi_mode=False, adaptive_alpha=True),
}


def run_variant(model_id: str, variant_name: str, variant_kwargs: dict,
                output_dir: Path) -> dict:
    pipeline = Pipeline()
    pipeline._packer = MLXPacker(output_base=output_dir / variant_name)
    t0 = time.monotonic()
    try:
        result = pipeline.run(
            model_id=model_id,
            target="auto",
            quality=99.0,
            output_format="mlx",
            **variant_kwargs,
        )
        elapsed = time.monotonic() - t0
        er = result.eval_result
        qp = result.quant_plan
        return {
            "status": "ok",
            "variant": variant_name,
            "model": model_id,
            "elapsed_s": round(elapsed, 1),
            "ppl_baseline": er.ppl_baseline if er else None,
            "ppl_quantized": er.ppl_quantized if er else None,
            "ppl_delta_pct": er.ppl_delta_pct if er else None,
            "avg_bits": qp.avg_bits if qp else result.estimated_bits,
            "fits_in_memory": result.fits_in_memory,
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {
            "status": "error",
            "variant": variant_name,
            "model": model_id,
            "elapsed_s": round(elapsed, 1),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--output", default="results/phase3_benchmark.json")
    parser.add_argument("--output-dir", default="atlas-output/phase3")
    args = parser.parse_args()

    unknown = set(args.variants) - set(VARIANTS.keys())
    if unknown:
        print(f"Varianti sconosciute: {unknown}. Disponibili: {list(VARIANTS.keys())}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.variants:
        print(f"\n{'='*60}")
        print(f"Variante: {name}  |  Modello: {args.model}")
        print(f"{'='*60}")
        r = run_variant(args.model, name, VARIANTS[name], output_dir)
        results.append(r)
        if r["status"] == "ok":
            print(f"PPL delta: {r['ppl_delta_pct']:.2f}%  |  avg_bits: {r['avg_bits']:.2f}"
                  f"  |  time: {r['elapsed_s']:.0f}s")
        else:
            print(f"ERRORE: {r['error']}")
        output_path.write_text(json.dumps(results, indent=2))

    print(f"\nRisultati salvati in {output_path}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    print("\n" + "="*60)
    print("RIEPILOGO PARETO")
    print("="*60)
    print(f"{'Variante':<12} {'PPL delta':>10} {'avg bits':>9} {'Status'}")
    print("-"*50)
    for r in results:
        if r["status"] == "ok" and r["ppl_delta_pct"] is not None:
            print(f"{r['variant']:<12} {r['ppl_delta_pct']:>9.2f}% {r['avg_bits']:>9.2f}  ok")
        else:
            err = r.get("error", "")[:30]
            print(f"{r['variant']:<12} {'—':>10} {'—':>9}  ERROR: {err}")


if __name__ == "__main__":
    main()
