import numpy as np
import pytest
import mlx.core as mx

from atlas.quant.qi_smooth import (
    _apply_error_refinement,
    _compute_channel_error,
)


class TestApplyErrorRefinement:
    def test_zero_lambda_returns_unchanged(self):
        s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # err with clear outlier so uniformity guard doesn't trigger
        err = np.array([0.1, 0.1, 5.0], dtype=np.float32)
        result = _apply_error_refinement(s, err, error_lambda=0.0)
        np.testing.assert_allclose(result, s)

    def test_uniform_error_returns_unchanged(self):
        """Se tutti i canali hanno errore simile, non modificare le scale."""
        s = np.ones(8, dtype=np.float32)
        err = np.ones(8, dtype=np.float32) * 0.5  # uniforme: max/mean = 1.0 < 2.0
        result = _apply_error_refinement(s, err, error_lambda=0.3)
        np.testing.assert_allclose(result, s)

    def test_high_error_channel_gets_larger_scale(self):
        s = np.ones(10, dtype=np.float32)
        # canale 9 ha errore 10× superiore agli altri → outlier chiaro
        err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        result = _apply_error_refinement(s, err, error_lambda=0.3)
        assert result[9] > result[0]

    def test_low_error_channels_unchanged(self):
        """Canali sotto threshold non devono cambiare."""
        s = np.ones(10, dtype=np.float32)
        err = np.array([0.1] * 8 + [0.5, 2.0], dtype=np.float32)
        result = _apply_error_refinement(s, err, error_lambda=0.3, threshold_pct=80.0)
        # i primi 8 canali (sotto threshold) devono restare a 1.0
        np.testing.assert_allclose(result[:8], s[:8], atol=1e-6)

    def test_output_dtype_float32(self):
        s = np.ones(10, dtype=np.float32)
        err = np.array([0.1] * 8 + [0.5, 5.0], dtype=np.float32)
        result = _apply_error_refinement(s, err, error_lambda=0.2)
        assert result.dtype == np.float32

    def test_refined_channels_increase(self):
        s = np.ones(10, dtype=np.float32)
        err = np.array([0.1] * 8 + [0.5, 5.0], dtype=np.float32)
        result = _apply_error_refinement(s, err, error_lambda=0.3, threshold_pct=80.0)
        assert np.all(result >= s - 1e-6)


class TestComputeChannelError:
    def test_returns_per_input_channel_array(self):
        in_features, out_features = 64, 128
        W = mx.random.normal((out_features, in_features)).astype(mx.float32)
        act = np.random.randn(32, in_features).astype(np.float32)
        err = _compute_channel_error(W, act, bits=4, group_size=32)
        assert err.shape == (in_features,)
        assert err.dtype == np.float32

    def test_error_nonnegative(self):
        W = mx.random.normal((64, 32)).astype(mx.float32)
        act = np.random.randn(16, 32).astype(np.float32)
        err = _compute_channel_error(W, act, bits=4, group_size=32)
        assert np.all(err >= 0)

    def test_zero_activation_gives_zero_error(self):
        W = mx.random.normal((64, 32)).astype(mx.float32)
        act = np.zeros((16, 32), dtype=np.float32)
        err = _compute_channel_error(W, act, bits=4, group_size=32)
        np.testing.assert_allclose(err, 0.0, atol=1e-6)

    def test_higher_bits_lower_error(self):
        """8-bit quantization should produce lower per-channel error than 4-bit."""
        mx.random.seed(42)
        W = mx.random.normal((128, 128)).astype(mx.float32)
        act = np.random.randn(32, 128).astype(np.float32)
        err4 = _compute_channel_error(W, act, bits=4, group_size=64)
        err8 = _compute_channel_error(W, act, bits=8, group_size=64)
        assert np.mean(err8) < np.mean(err4)
