"""AdaptiveSmooth: per-layer alpha optimization for SmoothQuant.

Standard SmoothQuant uses a fixed alpha=0.5 for all layers. SmoothQuant+
searches a uniform alpha in [0.4, 0.6] across the model. Neither optimizes
alpha independently per layer.

AdaptiveSmooth performs a grid search over alpha candidates for each layer,
selecting the alpha that minimizes activation-weighted quantization error:

  score(alpha, layer) = mean_j( max_out(|W_scaled[:,j] - Q(W_scaled)[:,j]|)
                                × act_max_j )

where W_scaled[:,j] = W[:,j] × s_j and s_j = act_max_j^alpha / w_max_j^(1-alpha).

This is novel: the error metric couples weight reconstruction error with
activation magnitude, so channels that both quantize poorly AND carry large
activations drive alpha selection — not just weight uniformity.
"""

import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np
from mlx_lm import load as mlx_lm_load

from atlas.quant.smooth import (
    _apply_scales_inplace,
    _save_smoothed_model,
)


_DEFAULT_ALPHA_CANDIDATES: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 0.8)


def adaptive_smooth_model_dir(
    model_id: str,
    alpha_candidates: tuple[float, ...] = _DEFAULT_ALPHA_CANDIDATES,
    num_calib: int = 16,
    bits: int = 4,
    group_size: int = 64,
) -> Path:
    """Per-layer alpha-optimized SmoothQuant. Returns temporary smoothed model dir.

    For each layer independently, selects the alpha in alpha_candidates that
    minimizes activation-weighted quantization error on the q_proj (attention)
    and gate_proj (MLP) representative weights.

    Caller is responsible for shutil.rmtree on the returned path.

    Args:
        alpha_candidates: grid of alpha values to search per layer.
            Default (0.2, 0.3, 0.5, 0.7, 0.8) spans the full [0,1] range.
        num_calib: WikiText-2 calibration samples.
        bits: bit-width for error simulation (should match final quantization).
        group_size: group size for error simulation.
    """
    model, tok = mlx_lm_load(model_id)

    # Collect calibration stats (act_max + w_max) for all layers
    calib_stats = _collect_calib_stats(model, tok, num_calib)

    # Per-layer alpha search → optimal scales
    scales = _compute_adaptive_scales(model, calib_stats, alpha_candidates, bits, group_size)

    _apply_scales_inplace(model, scales)
    mx.eval(*[v for v in mlx.utils.tree_flatten(model.parameters())[1]])

    out_dir = Path(tempfile.mkdtemp(prefix="atlas_adaptive_smooth_"))
    _save_smoothed_model(model, model_id, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_calib_stats(
    model,
    tok,
    num_calib: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Collect act_max and w_max per-channel for each layer.

    Returns dict[layer_idx, {
        'act_max_attn': [H],  activation max per input channel (attn path)
        'act_max_mlp':  [H],
        'w_max_attn':   [H],  weight max per input channel (q_proj representative)
        'w_max_mlp':    [H],  weight max per input channel (gate_proj representative)
    }]
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    samples = [r["text"] for r in ds if r["text"].strip()][:num_calib]

    num_layers = len(model.model.layers)
    act_max_attn: list[mx.array | None] = [None] * num_layers
    act_max_mlp:  list[mx.array | None] = [None] * num_layers

    for text in samples:
        tokens = tok.encode(text)
        if len(tokens) < 2:
            continue
        x = model.model.embed_tokens(mx.array(tokens)[None, :])
        for i, layer in enumerate(model.model.layers):
            ln_attn = layer.input_layernorm(x)
            cur_a = mx.max(mx.abs(ln_attn.astype(mx.float32)), axis=(0, 1))
            act_max_attn[i] = cur_a if act_max_attn[i] is None else mx.maximum(act_max_attn[i], cur_a)

            r_attn = layer.self_attn(ln_attn, mask=None, cache=None)
            h = x + r_attn

            ln_mlp = layer.post_attention_layernorm(h)
            cur_m = mx.max(mx.abs(ln_mlp.astype(mx.float32)), axis=(0, 1))
            act_max_mlp[i] = cur_m if act_max_mlp[i] is None else mx.maximum(act_max_mlp[i], cur_m)

            x = h + layer.mlp(ln_mlp)

    all_act = [m for m in act_max_attn + act_max_mlp if m is not None]
    if all_act:
        mx.eval(*all_act)

    stats: dict[int, dict[str, np.ndarray]] = {}
    for i, layer in enumerate(model.model.layers):
        if act_max_attn[i] is None or act_max_mlp[i] is None:
            hidden = layer.input_layernorm.weight.shape[0]
            fallback = np.ones(hidden, dtype=np.float32)
            stats[i] = {
                "act_max_attn": fallback,
                "act_max_mlp":  fallback,
                "w_max_attn":   fallback,
                "w_max_mlp":    fallback,
            }
            continue

        def _w_max(*weights) -> np.ndarray:
            return np.maximum.reduce(
                [np.max(np.abs(np.array(w.astype(mx.float32))), axis=0) for w in weights]
            )

        stats[i] = {
            "act_max_attn": np.array(act_max_attn[i]),
            "act_max_mlp":  np.array(act_max_mlp[i]),
            "w_max_attn":   _w_max(
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
            ),
            "w_max_mlp":    _w_max(
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
            ),
        }

    return stats


def _compute_adaptive_scales(
    model,
    calib_stats: dict[int, dict[str, np.ndarray]],
    alpha_candidates: tuple[float, ...],
    bits: int,
    group_size: int,
) -> dict[int, dict[str, np.ndarray]]:
    """For each layer, grid-search alpha and build smooth scales."""
    scales: dict[int, dict[str, np.ndarray]] = {}

    for i, layer in enumerate(model.model.layers):
        st = calib_stats[i]

        alpha_attn = _find_optimal_alpha(
            weight=layer.self_attn.q_proj.weight,
            act_max=st["act_max_attn"],
            w_max=st["w_max_attn"],
            alpha_candidates=alpha_candidates,
            bits=bits,
            group_size=group_size,
        )
        alpha_mlp = _find_optimal_alpha(
            weight=layer.mlp.gate_proj.weight,
            act_max=st["act_max_mlp"],
            w_max=st["w_max_mlp"],
            alpha_candidates=alpha_candidates,
            bits=bits,
            group_size=group_size,
        )

        s_attn = (
            np.maximum(st["act_max_attn"], 1e-8) ** alpha_attn
            / np.maximum(st["w_max_attn"], 1e-8) ** (1.0 - alpha_attn)
        )
        s_mlp = (
            np.maximum(st["act_max_mlp"], 1e-8) ** alpha_mlp
            / np.maximum(st["w_max_mlp"], 1e-8) ** (1.0 - alpha_mlp)
        )

        scales[i] = {
            "s_attn":     s_attn.astype(np.float32),
            "s_mlp":      s_mlp.astype(np.float32),
            "alpha_attn": alpha_attn,
            "alpha_mlp":  alpha_mlp,
        }

    return scales


def _find_optimal_alpha(
    weight: mx.array,
    act_max: np.ndarray,
    w_max: np.ndarray,
    alpha_candidates: tuple[float, ...],
    bits: int,
    group_size: int,
) -> float:
    """Grid search: returns alpha minimizing activation-weighted quant error.

    score(alpha) = mean_j( max_out(|W_scaled[:,j] - Q(W_scaled)[:,j]|) × act_max_j )

    Coupling weight reconstruction error with activation magnitude ensures
    alpha selection is driven by channels that matter for output quality,
    not just by raw weight uniformity.
    """
    W_np = np.array(weight.astype(mx.float32))  # [out, in]
    best_alpha = alpha_candidates[0]
    best_score = float("inf")

    for alpha in alpha_candidates:
        s = np.maximum(act_max, 1e-8) ** alpha / np.maximum(w_max, 1e-8) ** (1.0 - alpha)
        W_scaled = W_np * s[None, :]  # broadcast over output dim

        W_mx = mx.array(W_scaled.astype(np.float32))
        w_q, scales_q, biases_q = mx.quantize(W_mx, group_size=group_size, bits=bits)
        W_dq = mx.dequantize(w_q, scales_q, biases_q, group_size=group_size, bits=bits)
        mx.eval(W_dq)

        delta = np.array(mx.abs(W_mx - W_dq))    # [out, in]
        delta_per_in = delta.max(axis=0)           # [in] worst output channel per input
        score = float(np.mean(delta_per_in * act_max))

        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha
