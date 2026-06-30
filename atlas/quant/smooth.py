"""SmoothQuant-style pre-quantization weight scaling for Atlas Phase 2.5.

Per ogni coppia (LayerNorm → Linear) nel transformer Llama:
1. Calibration: activation max per-channel su dati di calibrazione
2. Smooth scale: s_j = act_max_j^alpha / weight_max_j^(1-alpha)  (default alpha=0.5)
3. Modifica LN weight: w_ln[j] /= s_j  (riduce magnitude attivazioni)
4. Modifica Linear input channels: W[:, j] *= s_j  (compensa lato pesi)

Net effect: forward pass invariante, pesi più uniformi → quantizzazione migliore.

Reference: "SmoothQuant: Accurate and Efficient Post-Training Quantization for
Large Language Models" (Xiao et al., 2022).
"""

import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np
from mlx_lm import load as mlx_lm_load


def smooth_model_dir(
    model_id: str,
    alpha: float = 0.5,
    num_calib: int = 16,
) -> Path:
    """Applica SmoothQuant e salva il modello smoothed in una directory temporanea.

    La directory è in formato HF-compatibile (config + tokenizer copiati da HF cache,
    pesi salvati con mx.save_safetensors). Il caller deve eliminare la directory
    dopo l'uso con shutil.rmtree.

    Returns:
        Path alla directory con il modello smoothed (non quantizzato).
    """
    model, tok = mlx_lm_load(model_id)
    scales = _compute_smooth_scales(model, tok, num_calib, alpha)
    _apply_scales_inplace(model, scales)

    # Forza materializzazione prima che il modello esca dallo scope.
    mx.eval(*[v for v in mlx.utils.tree_flatten(model.parameters())[1]])

    out_dir = Path(tempfile.mkdtemp(prefix="atlas_smooth_"))
    _save_smoothed_model(model, model_id, out_dir)
    return out_dir


def _compute_smooth_scales(
    model,
    tok,
    num_calib: int,
    alpha: float,
) -> dict[int, dict[str, np.ndarray]]:
    """Calibra smooth scales per ogni layer Llama.

    Attivazioni campionate all'output di input_layernorm (percorso attention)
    e post_attention_layernorm (percorso MLP).
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    samples = [r["text"] for r in ds if r["text"].strip()][:num_calib]

    num_layers = len(model.model.layers)
    act_max_attn: list[mx.array | None] = [None] * num_layers
    act_max_mlp: list[mx.array | None] = [None] * num_layers

    for text in samples:
        tokens = tok.encode(text)
        if len(tokens) < 2:
            continue
        input_ids = mx.array(tokens)[None, :]
        x = model.model.embed_tokens(input_ids)

        for i, layer in enumerate(model.model.layers):
            # Percorso attention: attivazioni a input_layernorm output
            ln_attn = layer.input_layernorm(x)
            cur = mx.max(mx.abs(ln_attn.astype(mx.float32)), axis=(0, 1))
            act_max_attn[i] = cur if act_max_attn[i] is None else mx.maximum(act_max_attn[i], cur)

            # Forward attention per ottenere il residual prima di MLP
            r_attn = layer.self_attn(ln_attn, mask=None, cache=None)
            h = x + r_attn

            # Percorso MLP: attivazioni a post_attention_layernorm output
            ln_mlp = layer.post_attention_layernorm(h)
            cur = mx.max(mx.abs(ln_mlp.astype(mx.float32)), axis=(0, 1))
            act_max_mlp[i] = cur if act_max_mlp[i] is None else mx.maximum(act_max_mlp[i], cur)

            r_mlp = layer.mlp(ln_mlp)
            x = h + r_mlp

    # Materializza act_max prima che il modello esca dallo scope
    all_act = [m for m in act_max_attn + act_max_mlp if m is not None]
    if all_act:
        mx.eval(*all_act)

    scales: dict[int, dict[str, np.ndarray]] = {}
    for i, layer in enumerate(model.model.layers):
        if act_max_attn[i] is None or act_max_mlp[i] is None:
            hidden = layer.input_layernorm.weight.shape[0]
            scales[i] = {
                "s_attn": np.ones(hidden, dtype=np.float32),
                "s_mlp": np.ones(hidden, dtype=np.float32),
            }
            continue

        act_a = np.array(act_max_attn[i])
        act_m = np.array(act_max_mlp[i])

        # Weight max per input channel (axis=1 = input dim in [out, in] weight)
        def _w_max(*weights) -> np.ndarray:
            return np.maximum.reduce(
                [np.max(np.abs(np.array(w.astype(mx.float32))), axis=0) for w in weights]
            )

        w_max_a = _w_max(
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight,
        )
        w_max_m = _w_max(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)

        s_attn = np.maximum(act_a, 1e-8) ** alpha / (np.maximum(w_max_a, 1e-8) ** (1.0 - alpha))
        s_mlp = np.maximum(act_m, 1e-8) ** alpha / (np.maximum(w_max_m, 1e-8) ** (1.0 - alpha))

        scales[i] = {"s_attn": s_attn.astype(np.float32), "s_mlp": s_mlp.astype(np.float32)}

    return scales


def _apply_scales_inplace(model, scales: dict[int, dict[str, np.ndarray]]) -> None:
    """Applica smooth scales ai pesi del modello in-memory.

    Per ogni layer:
    - input_layernorm.weight /= s_attn   (riduce attivazioni attention)
    - q/k/v_proj.weight[:, j] *= s_attn[j]  (compensa nei pesi)
    - post_attention_layernorm.weight /= s_mlp
    - gate/up_proj.weight[:, j] *= s_mlp[j]
    """
    for i, layer in enumerate(model.model.layers):
        if i not in scales:
            continue
        s_attn = mx.array(scales[i]["s_attn"])  # [H]
        s_mlp = mx.array(scales[i]["s_mlp"])    # [H]

        orig_dtype = layer.input_layernorm.weight.dtype

        # Attention path
        ln = layer.input_layernorm
        ln.weight = (ln.weight.astype(mx.float32) / s_attn).astype(orig_dtype)

        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.weight = (proj.weight.astype(mx.float32) * s_attn[None, :]).astype(orig_dtype)

        # MLP path
        post_ln = layer.post_attention_layernorm
        post_ln.weight = (post_ln.weight.astype(mx.float32) / s_mlp).astype(orig_dtype)

        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
            proj.weight = (proj.weight.astype(mx.float32) * s_mlp[None, :]).astype(orig_dtype)


def _save_smoothed_model(model, model_id: str, out_dir: Path) -> None:
    """Salva il modello in formato HF-compatibile (config + tokenizer + pesi)."""
    from huggingface_hub import snapshot_download

    hf_dir = Path(snapshot_download(model_id))

    # Copia config e tokenizer (tutti i file non-safetensors)
    for f in hf_dir.iterdir():
        if f.suffix not in (".safetensors",) and f.name not in ("pytorch_model.bin",):
            shutil.copy2(f, out_dir / f.name)

    # Salva pesi con mx.save_safetensors (supporta bfloat16)
    flat_params = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_dir / "model.safetensors"), flat_params)
