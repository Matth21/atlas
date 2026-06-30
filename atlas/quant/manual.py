"""Layer-by-layer quantizer with cross-layer error compensation for Atlas Phase 2.5.

Esegue due passi:
1. Quantizzazione weights via MixedQuantizer (mlx_lm.convert, già testato).
2. Calibration pass su FP16 + quantizzato per calcolare bias corrections per-layer.

Le bias corrections vengono applicate durante la valutazione PPL (PerplexityEval),
non sono baked nei weights — il modello salvato è identico a MixedQuantizer.
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_lm_load

from atlas.plan.planner import QuantPlan
from atlas.quant.compensator import ErrorCompensator
from atlas.quant.mixed import MixedQuantizer


@dataclass(frozen=True)
class ManualQuantResult:
    output_path: Path
    plan: QuantPlan
    quantized_size_mb: float
    original_size_mb: float
    bias_corrections: tuple[mx.array, ...] | None  # None se enable_compensation=False


class ManualLayerQuantizer:
    """Quantizza con MixedQuantizer e calcola bias corrections layer-by-layer."""

    def quantize(
        self,
        model_id: str,
        plan: QuantPlan,
        enable_compensation: bool = True,
    ) -> ManualQuantResult:
        # Step 1: quantizza weights (usa mlx_lm.convert via MixedQuantizer)
        mixed_result = MixedQuantizer().quantize(model_id, plan)

        # Step 2: calibration pass per bias corrections
        bias_corrections = None
        if enable_compensation:
            bias_corrections = _compute_bias_corrections(
                model_id, mixed_result.output_path, plan
            )

        return ManualQuantResult(
            output_path=mixed_result.output_path,
            plan=plan,
            quantized_size_mb=mixed_result.quantized_size_mb,
            original_size_mb=mixed_result.original_size_mb,
            bias_corrections=bias_corrections,
        )


def _compute_bias_corrections(
    model_id: str,
    quantized_path: Path,
    plan: QuantPlan,
    num_samples: int = 5,
) -> tuple[mx.array, ...]:
    """Calibration pass: calcola bias correttivo per-layer da errori di quantizzazione.

    Carica FP16 e quantizzato separatamente; per ogni sample, accumula
    l'errore per-layer (fp16_output - quantized_output) e ne calcola la media.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    samples = [row["text"] for row in ds if row["text"].strip()][:num_samples]

    model_fp16, tok = mlx_lm_load(model_id)
    model_q, _ = mlx_lm_load(str(quantized_path))

    compensator = ErrorCompensator(enabled=True)
    num_layers = len(model_fp16.model.layers)
    bias_accum: list[mx.array | None] = [None] * num_layers
    count = 0

    for text in samples:
        tokens = tok.encode(text)
        if len(tokens) < 2:
            continue
        input_ids = mx.array(tokens)[None, :]

        # Forward pass FP16 layer per layer
        x_fp16 = model_fp16.model.embed_tokens(input_ids)
        fp16_outputs: list[mx.array] = []
        for layer in model_fp16.model.layers:
            x_fp16 = layer(x_fp16, cache=None)
            fp16_outputs.append(x_fp16)

        # Forward pass quantizzato layer per layer
        x_q = model_q.model.embed_tokens(input_ids)
        for i, layer in enumerate(model_q.model.layers):
            x_q = layer(x_q, cache=None)
            bias = compensator.compute_bias(fp16_outputs[i], x_q)
            bias_accum[i] = bias if bias_accum[i] is None else bias_accum[i] + bias

        count += 1

    if count == 0 or any(b is None for b in bias_accum):
        hidden = model_fp16.model.embed_tokens.weight.shape[-1]
        return tuple(mx.zeros((hidden,)) for _ in range(num_layers))

    return tuple(b / count for b in bias_accum)  # type: ignore[union-attr]
