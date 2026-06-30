"""QI-SmoothQuant: Quantization-Error-Informed SmoothQuant for Atlas.

Two-pass extension of SmoothQuant (Xiao et al., 2022):

  Pass 1 — standard SmoothQuant with alpha=0.5:
    s_j^(0) = act_max_j^alpha / weight_max_j^(1-alpha)

  Error measurement — simulate 4-bit quantization per-layer, compute
  per-channel error = weight_reconstruction_error × activation_magnitude:
    err_j = max_out(|W[:,j] - Q(W)[:,j]|) × mean_token(|x[:,j]|)

  Scale refinement — channels with high quantization error get their
  smooth scale increased, shifting more load onto the weight side:
    s_j^(1) = s_j^(0) × (1 + λ × rank_norm(err_j))

  Pass 2 — re-apply refined scales s^(1), save smoothed model.

Unlike SmoothQuant (uses calibration statistics only) and AWQ (uses
activation magnitude only), QI-SmoothQuant uses the *actual observed
quantization error* after pass-1 smoothing to guide pass-2 refinement.
This feedback loop is novel: no prior work combines error-informed scale
refinement with SmoothQuant on MLX/Apple Silicon.
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
    _compute_smooth_scales,
    _save_smoothed_model,
)


def qi_smooth_model_dir(
    model_id: str,
    alpha: float = 0.5,
    num_calib: int = 16,
    error_lambda: float = 0.3,
    bits: int = 4,
    group_size: int = 64,
) -> Path:
    """Two-pass QI-SmoothQuant. Returns path to smoothed model dir (caller must rmtree).

    Args:
        alpha: SmoothQuant migration strength (pass-1 base).
        num_calib: calibration samples from WikiText-2.
        error_lambda: scale refinement strength. 0 = pure SmoothQuant (no QI).
            Larger → more aggressive correction of high-error channels.
        bits: bit-width used for error simulation.
        group_size: group size used for error simulation.
    """
    # Pass 1: standard SmoothQuant
    model, tok = mlx_lm_load(model_id)
    scales = _compute_smooth_scales(model, tok, num_calib, alpha)
    _apply_scales_inplace(model, scales)
    mx.eval(*[v for v in mlx.utils.tree_flatten(model.parameters())[1]])

    # Collect calibration activations (reuse same samples as pass-1)
    calib_acts = _collect_calibration_acts(model, tok, num_calib)

    # Measure per-channel quantization error and refine scales
    scales = _refine_scales_from_errors(model, scales, calib_acts, bits, group_size, error_lambda)

    # Pass 2: reload FP16 model, apply refined scales
    model2, _ = mlx_lm_load(model_id)
    _apply_scales_inplace(model2, scales)
    mx.eval(*[v for v in mlx.utils.tree_flatten(model2.parameters())[1]])

    out_dir = Path(tempfile.mkdtemp(prefix="atlas_qi_smooth_"))
    _save_smoothed_model(model2, model_id, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_calibration_acts(
    model,
    tok,
    num_calib: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Forward pass to collect per-layer LN output activations (post-smooth).

    Returns dict[layer_idx, {'act_attn': np.ndarray [T, H], 'act_mlp': np.ndarray [T, H]}]
    aggregated (concatenated) over all calibration samples.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    samples = [r["text"] for r in ds if r["text"].strip()][:num_calib]

    num_layers = len(model.model.layers)
    acts_attn: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    acts_mlp: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

    for text in samples:
        tokens = tok.encode(text)
        if len(tokens) < 2:
            continue
        ids = mx.array(tokens)[None, :]
        x = model.model.embed_tokens(ids)

        for i, layer in enumerate(model.model.layers):
            ln_attn = layer.input_layernorm(x)
            # Store [T, H] — squeeze batch dim
            acts_attn[i].append(np.array(ln_attn.astype(mx.float32).squeeze(0)))

            r_attn = layer.self_attn(ln_attn, mask=None, cache=None)
            h = x + r_attn

            ln_mlp = layer.post_attention_layernorm(h)
            acts_mlp[i].append(np.array(ln_mlp.astype(mx.float32).squeeze(0)))

            r_mlp = layer.mlp(ln_mlp)
            x = h + r_mlp

    result: dict[int, dict[str, np.ndarray]] = {}
    for i in range(num_layers):
        if acts_attn[i]:
            result[i] = {
                "act_attn": np.concatenate(acts_attn[i], axis=0),  # [total_tokens, H]
                "act_mlp": np.concatenate(acts_mlp[i], axis=0),
            }
    return result


def _compute_channel_error(
    weight: mx.array,
    act: np.ndarray,
    bits: int,
    group_size: int,
) -> np.ndarray:
    """Per-input-channel quantization error for a Linear layer weight.

    err_j = max_out(|W[:,j] - Q(W)[:,j]|) × mean_token(|act[:,j]|)

    Channels where quantization error is large AND activations are large
    contribute most to the output error — these need refined smooth scales.

    Args:
        weight: [out_features, in_features] in bfloat16 / float32.
        act: [num_tokens, in_features] float32 activations at layer input.
        bits: quantization bits for error simulation.
        group_size: quantization group size for error simulation.

    Returns:
        err: [in_features] float32 per-channel error.
    """
    W = weight.astype(mx.float32)
    w_q, scales_q, biases_q = mx.quantize(W, group_size=group_size, bits=bits)
    W_dq = mx.dequantize(w_q, scales_q, biases_q, group_size=group_size, bits=bits)
    mx.eval(W_dq)

    # Weight reconstruction error per input channel: max over output dim
    delta_W = mx.abs(W - W_dq)           # [out, in]
    delta_per_in = mx.max(delta_W, axis=0)  # [in]
    mx.eval(delta_per_in)

    delta_np = np.array(delta_per_in)

    # Activation magnitude per input channel: mean over tokens
    act_scale = np.mean(np.abs(act), axis=0)  # [in]

    # Combined: err_j = weight_error_j × act_magnitude_j
    return (delta_np * act_scale).astype(np.float32)


def _refine_scales_from_errors(
    model,
    scales: dict[int, dict[str, np.ndarray]],
    calib_acts: dict[int, dict[str, np.ndarray]],
    bits: int,
    group_size: int,
    error_lambda: float,
) -> dict[int, dict[str, np.ndarray]]:
    """Refine smooth scales using observed per-channel quantization error.

    For each layer, computes err_j = weight_recon_error_j × act_magnitude_j,
    rank-normalizes to [0,1], then:
        s_j_refined = s_j × (1 + λ × rank_norm(err_j))

    High-error channels get a larger smooth scale → more difficulty shifted to
    weights → weights become more uniform for those channels → lower quant error.
    """
    refined: dict[int, dict[str, np.ndarray]] = {}

    for i, layer in enumerate(model.model.layers):
        if i not in scales or i not in calib_acts:
            refined[i] = scales.get(i, {})
            continue

        s_attn = scales[i]["s_attn"]
        s_mlp = scales[i]["s_mlp"]
        act_attn = calib_acts[i]["act_attn"]  # [T, H]
        act_mlp = calib_acts[i]["act_mlp"]

        # Attention path: measure error on q_proj (representative)
        err_attn = _compute_channel_error(
            layer.self_attn.q_proj.weight, act_attn, bits, group_size
        )
        s_attn_ref = _apply_error_refinement(s_attn, err_attn, error_lambda)

        # MLP path: measure error on gate_proj (representative)
        err_mlp = _compute_channel_error(
            layer.mlp.gate_proj.weight, act_mlp, bits, group_size
        )
        s_mlp_ref = _apply_error_refinement(s_mlp, err_mlp, error_lambda)

        refined[i] = {
            "s_attn": s_attn_ref.astype(np.float32),
            "s_mlp": s_mlp_ref.astype(np.float32),
        }

    return refined


def _apply_error_refinement(
    s: np.ndarray,
    err: np.ndarray,
    error_lambda: float,
    threshold_pct: float = 80.0,
) -> np.ndarray:
    """Refine scales only for channels whose error exceeds threshold_pct percentile.

    Two-step:
    1. Only channels with err_j > p{threshold_pct} are refined (rest unchanged).
       This prevents amplifying noise when all channels have similar low error.
    2. For refined channels: s_j *= (1 + λ × rank_norm_above_threshold(err_j)).

    threshold_pct=80 means only top 20% highest-error channels are touched.
    If max(err) / (mean(err) + eps) < 2.0, error distribution is too uniform
    to benefit from refinement → return s unchanged.
    """
    n = len(err)
    if n == 0 or error_lambda == 0.0:
        return s

    # Guard: if error distribution is nearly uniform, skip refinement entirely.
    err_mean = np.mean(err) + 1e-8
    if np.max(err) / err_mean < 2.0:
        return s

    threshold = np.percentile(err, threshold_pct)
    mask = err > threshold

    if not np.any(mask):
        return s

    s_out = s.copy()
    err_above = err[mask]
    n_above = len(err_above)
    if n_above == 1:
        # Single outlier: treat as rank=1 (worst channel gets full lambda correction)
        s_out[mask] = s[mask] * (1.0 + error_lambda)
    else:
        rank_above = np.argsort(np.argsort(err_above)).astype(np.float32) / (n_above - 1)
        s_out[mask] = s[mask] * (1.0 + error_lambda * rank_above)
    return s_out
