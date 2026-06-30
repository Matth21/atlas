import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_lm_load


@dataclass(frozen=True)
class EvalResult:
    ppl_baseline: float
    ppl_quantized: float
    ppl_delta_pct: float
    num_samples: int
    eval_time_s: float


class PerplexityEval:
    def evaluate(
        self,
        quantized_path: Path,
        model_id: str,
        num_samples: int = 100,
        bias_corrections: "tuple[mx.array, ...] | None" = None,
    ) -> EvalResult:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        samples = _load_wikitext_samples(num_samples)
        start = time.monotonic()

        ppl_baseline = _compute_perplexity(model_id, samples)

        if bias_corrections is not None:
            ppl_quantized = _compute_perplexity_with_corrections(
                str(quantized_path), samples, bias_corrections
            )
        else:
            ppl_quantized = _compute_perplexity(str(quantized_path), samples)

        elapsed = time.monotonic() - start

        if ppl_baseline == 0:
            delta_pct = float("inf")
        else:
            delta_pct = round((ppl_quantized - ppl_baseline) / ppl_baseline * 100, 2)

        return EvalResult(
            ppl_baseline=round(ppl_baseline, 4),
            ppl_quantized=round(ppl_quantized, 4),
            ppl_delta_pct=delta_pct,
            num_samples=len(samples),
            eval_time_s=round(elapsed, 2),
        )


def _load_wikitext_samples(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in ds if row["text"].strip()]
    return texts[:num_samples]


def _compute_perplexity(model_path: str, samples: list[str]) -> float:
    model, tokenizer = mlx_lm_load(model_path)

    total_loss = 0.0
    total_tokens = 0

    for text in samples:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        input_ids = mx.array(tokens[:-1])[None, :]
        targets = mx.array(tokens[1:])

        logits = model(input_ids)
        logits = logits.squeeze(0).astype(mx.float32)

        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def _compute_perplexity_with_corrections(
    model_path: str,
    samples: list[str],
    bias_corrections: tuple[mx.array, ...],
) -> float:
    """Forward pass manuale con bias corrections applicate tra layer consecutivi.

    bias_corrections[i] ha shape [hidden_size]; broadcast a [1, 1, hidden_size].
    Supporta modelli con lm_head separato (es. TinyLlama) e modelli con
    tied embeddings (es. Qwen) dove si usa embed_tokens.as_linear().
    """
    model, tokenizer = mlx_lm_load(model_path)
    total_loss = 0.0
    total_tokens = 0

    for text in samples:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        input_ids = mx.array(tokens[:-1])[None, :]
        targets = mx.array(tokens[1:])

        x = model.model.embed_tokens(input_ids)
        for i, layer in enumerate(model.model.layers):
            x = layer(x, cache=None)
            if i < len(bias_corrections):
                x = x + bias_corrections[i][None, None, :]

        x = model.model.norm(x)

        if hasattr(model, "lm_head") and model.lm_head is not None:
            logits = model.lm_head(x)
        else:
            logits = model.model.embed_tokens.as_linear(x)

        logits = logits.squeeze(0).astype(mx.float32)

        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("inf")

    return math.exp(total_loss / total_tokens)
