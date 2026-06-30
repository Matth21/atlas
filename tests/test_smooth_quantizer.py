"""Tests per SmoothQuantizer (Phase 2.5).

Proprietà chiave di SmoothQuant: il forward pass è matematicamente invariante
dopo lo smoothing (la scaling si cancella tra LN weight e Linear input channels).
"""

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from unittest.mock import patch, MagicMock

from atlas.quant.smooth import _apply_scales_inplace, _compute_smooth_scales


class MockRMSNorm:
    def __init__(self, hidden: int):
        self.weight = mx.ones((hidden,), dtype=mx.bfloat16)

    def __call__(self, x):
        return x * self.weight[None, None, :]


class MockLinear:
    def __init__(self, out: int, inp: int):
        self.weight = mx.ones((out, inp), dtype=mx.bfloat16)

    def __call__(self, x):
        return x @ self.weight.T


class MockAttn:
    def __init__(self, hidden: int):
        self.q_proj = MockLinear(hidden, hidden)
        self.k_proj = MockLinear(hidden, hidden)
        self.v_proj = MockLinear(hidden, hidden)

    def __call__(self, x, mask=None, cache=None):
        return self.q_proj(x)


class MockMLP:
    def __init__(self, hidden: int, ffn: int):
        self.gate_proj = MockLinear(ffn, hidden)
        self.up_proj = MockLinear(ffn, hidden)

    def __call__(self, x):
        return self.gate_proj(x)


class MockLayer:
    def __init__(self, hidden: int, ffn: int):
        self.input_layernorm = MockRMSNorm(hidden)
        self.post_attention_layernorm = MockRMSNorm(hidden)
        self.self_attn = MockAttn(hidden)
        self.mlp = MockMLP(hidden, ffn)


class MockModel:
    def __init__(self, n_layers: int = 2, hidden: int = 8, ffn: int = 16):
        self.layers = [MockLayer(hidden, ffn) for _ in range(n_layers)]


class MockWrapper:
    def __init__(self):
        self.model = MockModel()


class TestApplyScalesInplace:
    def test_ln_weight_divided_by_s_attn(self):
        model = MockWrapper()
        hidden = 8
        s = np.ones(hidden, dtype=np.float32) * 2.0
        scales = {0: {"s_attn": s, "s_mlp": s}, 1: {"s_attn": s, "s_mlp": s}}

        orig_ln = np.array(model.model.layers[0].input_layernorm.weight.astype(mx.float32))
        _apply_scales_inplace(model, scales)
        new_ln = np.array(model.model.layers[0].input_layernorm.weight.astype(mx.float32))

        np.testing.assert_allclose(new_ln, orig_ln / 2.0, atol=1e-2)

    def test_linear_weight_multiplied_by_s_attn(self):
        model = MockWrapper()
        hidden = 8
        s = np.ones(hidden, dtype=np.float32) * 3.0
        scales = {0: {"s_attn": s, "s_mlp": s}, 1: {"s_attn": s, "s_mlp": s}}

        orig_q = np.array(model.model.layers[0].self_attn.q_proj.weight.astype(mx.float32))
        _apply_scales_inplace(model, scales)
        new_q = np.array(model.model.layers[0].self_attn.q_proj.weight.astype(mx.float32))

        # weight[:, j] *= s[j]; con s uniforme = 3.0, tutti *= 3
        np.testing.assert_allclose(new_q, orig_q * 3.0, atol=1e-2)

    def test_smoothing_invariance(self):
        """Forward pass invariante: LN/s * W*s = LN * W."""
        hidden = 8
        model = MockWrapper()

        # Input casuale
        x = mx.array(np.random.randn(1, 4, hidden).astype(np.float32))

        # Output prima dello smoothing
        layer = model.model.layers[0]
        out_before = layer.self_attn.q_proj(layer.input_layernorm(x))

        # Applica scales non-uniform
        rng = np.random.RandomState(42)
        s = (rng.rand(hidden) * 2 + 0.5).astype(np.float32)
        scales = {0: {"s_attn": s, "s_mlp": s}, 1: {"s_attn": s, "s_mlp": s}}
        _apply_scales_inplace(model, scales)

        # Output dopo lo smoothing (deve essere ~uguale)
        out_after = layer.self_attn.q_proj(layer.input_layernorm(x))

        np.testing.assert_allclose(
            np.array(out_before.astype(mx.float32)),
            np.array(out_after.astype(mx.float32)),
            atol=1e-1,  # bf16 accumulation
        )

    def test_identity_scale_noop(self):
        """Scale = 1.0 non modifica i pesi."""
        model = MockWrapper()
        hidden = 8
        s = np.ones(hidden, dtype=np.float32)
        scales = {0: {"s_attn": s, "s_mlp": s}, 1: {"s_attn": s, "s_mlp": s}}

        orig = np.array(model.model.layers[0].input_layernorm.weight.astype(mx.float32))
        _apply_scales_inplace(model, scales)
        new = np.array(model.model.layers[0].input_layernorm.weight.astype(mx.float32))

        np.testing.assert_allclose(new, orig, atol=1e-3)
