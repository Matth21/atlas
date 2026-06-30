"""Cross-layer activation error compensator for Atlas Phase 2.5.

Dopo quantizzazione layer N, calcola bias correttivo per-canale
(media dell'errore su batch e sequenza) applicato all'input di layer N+1.
"""

import mlx.core as mx


class ErrorCompensator:
    """Calcola e applica correzione additiva per-canale tra layer quantizzati.

    Se enabled=False, compute_bias restituisce sempre un vettore zero —
    usato per disattivare la compensazione nell'ablation study (varianti A e B).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def compute_bias(self, out_fp16: mx.array, out_q: mx.array) -> mx.array:
        """Calcola bias correttivo per-canale dall'errore di quantizzazione.

        Args:
            out_fp16: output del layer FP16, shape [batch, seq, hidden]
            out_q:    output del layer quantizzato, shape [batch, seq, hidden]

        Returns:
            bias: shape [hidden], da sommare all'input del layer successivo.
                  Zero se self.enabled is False.
        """
        if not self.enabled:
            hidden = out_fp16.shape[-1]
            return mx.zeros((hidden,), dtype=mx.float32)

        error = (out_fp16 - out_q).astype(mx.float32)
        # Media su batch e sequenza → [hidden_size]
        return mx.mean(error, axis=(0, 1))
