import mlx.core as mx
import pytest

from atlas.quant.compensator import ErrorCompensator


class TestErrorCompensator:
    def test_compute_bias_shape(self):
        comp = ErrorCompensator(enabled=True)
        # shape: [batch=1, seq=3, hidden=8]
        out_fp16 = mx.ones((1, 3, 8))
        out_q = mx.zeros((1, 3, 8))
        bias = comp.compute_bias(out_fp16, out_q)
        assert bias.shape == (8,)

    def test_compute_bias_value(self):
        comp = ErrorCompensator(enabled=True)
        out_fp16 = mx.array([[[2.0, 4.0]]])  # shape [1,1,2]
        out_q = mx.array([[[1.0, 1.0]]])
        bias = comp.compute_bias(out_fp16, out_q)
        # error = [[1.0, 3.0]], mean over (0,1) = [1.0, 3.0]
        assert abs(bias[0].item() - 1.0) < 1e-5
        assert abs(bias[1].item() - 3.0) < 1e-5

    def test_disabled_returns_zero_bias(self):
        comp = ErrorCompensator(enabled=False)
        out_fp16 = mx.ones((1, 3, 8)) * 5.0
        out_q = mx.zeros((1, 3, 8))
        bias = comp.compute_bias(out_fp16, out_q)
        assert mx.all(bias == 0).item()

    def test_bias_has_correct_dtype(self):
        comp = ErrorCompensator(enabled=True)
        out_fp16 = mx.ones((1, 2, 4))
        out_q = mx.zeros((1, 2, 4))
        bias = comp.compute_bias(out_fp16, out_q)
        assert bias.dtype == mx.float32
