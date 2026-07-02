"""Fake-quantization reversibile per blocco (SGSR-2 CostProfiler).

Sostituisce i pesi 2D di un modulo con dequantize(quantize(w)) mantenendo
il compute in bf16: misura l'errore di rappresentazione senza convertire
il modello. `apply_fake_quant` ritorna gli originali per il ripristino
bit-esatto con `restore_weights`.
"""

import mlx.core as mx
import mlx.utils

MAX_GROUP_SIZE = 128


def quantizable_weights(module) -> list[tuple[str, mx.array]]:
    return [
        (name, w)
        for name, w in mlx.utils.tree_flatten(module.parameters())
        if isinstance(w, mx.array)
        and w.ndim == 2
        and w.shape[-1] % MAX_GROUP_SIZE == 0
    ]


def apply_fake_quant(module, bits: int, group_size: int) -> dict[str, mx.array]:
    originals: dict[str, mx.array] = {}
    updates: list[tuple[str, mx.array]] = []
    for name, w in quantizable_weights(module):
        originals[name] = w
        wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        wd = mx.dequantize(
            wq, scales, biases, group_size=group_size, bits=bits
        ).astype(w.dtype)
        updates.append((name, wd))
    module.update(mlx.utils.tree_unflatten(updates))
    mx.eval(module.parameters())
    return originals


def restore_weights(module, originals: dict[str, mx.array]) -> None:
    module.update(mlx.utils.tree_unflatten(list(originals.items())))
    mx.eval(module.parameters())
