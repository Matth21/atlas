"""Modello giocattolo con la stessa struttura dei modelli mlx_lm.

dim=128 → compatibile con tutti i group size {32, 64, 128}.
"""

import mlx.core as mx
import mlx.nn as nn


class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache=None):
        return x + self.proj(x)


class _Inner(nn.Module):
    def __init__(self, vocab: int, dim: int, n_blocks: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = [ToyBlock(dim) for _ in range(n_blocks)]
        self.norm = nn.RMSNorm(dim)


class ToyModel(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 128, n_blocks: int = 3):
        super().__init__()
        self.model = _Inner(vocab, dim, n_blocks)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))
