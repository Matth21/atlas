"""PPL sliding-window su Wikitext-2 test completo, protocollo standard.

Ogni token è valutato una sola volta, con contesto ≥ (window - stride)
token. Bootstrap CI95 sulle NLL per-token.
"""

import math
import random

import mlx.core as mx
import mlx.nn as nn

WINDOW = 2048
STRIDE = 512


def sliding_nlls(
    model, tokens: list[int], window: int = WINDOW, stride: int = STRIDE
) -> list[float]:
    if not (window > stride > 0):
        raise ValueError(
            f"serve window > stride > 0, ricevuto window={window} stride={stride}"
        )
    nlls: list[float] = []
    scored_until = 1  # il token 0 non è predicibile
    start = 0
    while scored_until < len(tokens):
        end = min(start + window, len(tokens))
        ids = mx.array(tokens[start : end - 1])[None, :]
        targets = mx.array(tokens[start + 1 : end])
        logits = model(ids).squeeze(0).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        offset = scored_until - (start + 1)
        nlls.extend(losses[offset:].tolist())
        scored_until = end
        if end == len(tokens):
            break
        start = end - (window - stride)
    return nlls


def ppl_with_ci(
    nlls: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    n = len(nlls)
    ppl = math.exp(sum(nlls) / n)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [nlls[rng.randrange(n)] for _ in range(n)]
        boots.append(math.exp(sum(sample) / n))
    boots.sort()
    return ppl, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def wikitext2_test_tokens(tokenizer) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    return tokenizer.encode(text)
