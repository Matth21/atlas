"""KL(base ‖ quant) su top-K logits + bucket di coda, streaming.

Memorizzare i logits completi del baseline per 32 seq × 512 token × vocab
150k non entra in memoria; il top-64 + massa residua approssima la KL con
errore trascurabile (la massa oltre il top-64 è < 1% per LLM calibrati).
"""

from dataclasses import dataclass

import mlx.core as mx

TOP_K = 64
_EPS = 1e-10


@dataclass(frozen=True)
class LogitsSnapshot:
    indices: mx.array   # [T, K]
    logprobs: mx.array  # [T, K] float32
    tail_mass: mx.array  # [T] float32


def _log_softmax(logits: mx.array) -> mx.array:
    lp = logits.astype(mx.float32)
    return lp - mx.logsumexp(lp, axis=-1, keepdims=True)


def snapshot_topk(logits: mx.array, k: int = TOP_K) -> LogitsSnapshot:
    lp = _log_softmax(logits)
    idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
    top_lp = mx.take_along_axis(lp, idx, axis=-1)
    tail = mx.maximum(1.0 - mx.exp(top_lp).sum(axis=-1), _EPS)
    return LogitsSnapshot(indices=idx, logprobs=top_lp, tail_mass=tail)


def kl_vs_snapshot(base: LogitsSnapshot, logits: mx.array) -> float:
    lq = _log_softmax(logits)
    lq_k = mx.take_along_axis(lq, base.indices, axis=-1)
    p = mx.exp(base.logprobs)
    kl_top = (p * (base.logprobs - lq_k)).sum(axis=-1)
    q_tail = mx.maximum(1.0 - mx.exp(lq_k).sum(axis=-1), _EPS)
    kl_tail = base.tail_mass * (mx.log(base.tail_mass) - mx.log(q_tail))
    return float(mx.maximum(kl_top + kl_tail, 0.0).mean())
