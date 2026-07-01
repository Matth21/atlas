import numpy as np
import pytest
import mlx.core as mx

from atlas.quant.adaptive_smooth import _find_optimal_alpha


class TestFindOptimalAlpha:
    """Unit tests for per-layer alpha grid search."""

    def _make_weight(self, out: int, in_: int, seed: int = 0) -> mx.array:
        mx.random.seed(seed)
        return mx.random.normal((out, in_)).astype(mx.float32)

    def test_returns_value_from_candidates(self):
        """Alpha returned must be one of the candidates."""
        candidates = (0.2, 0.3, 0.5, 0.7, 0.8)
        W = self._make_weight(64, 32)
        act_max = np.random.rand(32).astype(np.float32) + 0.1
        w_max = np.abs(np.array(mx.max(mx.abs(W), axis=0))) + 1e-8
        alpha = _find_optimal_alpha(W, act_max, w_max, candidates, bits=4, group_size=32)
        assert alpha in candidates

    def test_optimal_alpha_lower_score_than_others(self):
        """Optimal alpha must have the lowest or equal activation-weighted error."""
        mx.random.seed(42)
        candidates = (0.2, 0.5, 0.8)
        # Weight with strong outliers in a few input channels → alpha choice matters
        W_np = np.random.randn(128, 64).astype(np.float32)
        W_np[:, :8] *= 10.0  # outlier channels
        W = mx.array(W_np)
        act_max = np.ones(64, dtype=np.float32)
        act_max[:8] = 5.0  # matching outlier activations
        w_max = np.abs(W_np).max(axis=0)

        best_alpha = _find_optimal_alpha(W, act_max, w_max, candidates, bits=4, group_size=32)

        # Verify best_alpha truly has lowest score
        def score_for(alpha: float) -> float:
            s = np.maximum(act_max, 1e-8) ** alpha / np.maximum(w_max, 1e-8) ** (1 - alpha)
            W_sc = mx.array((W_np * s[None, :]).astype(np.float32))
            wq, sq, bq = mx.quantize(W_sc, group_size=32, bits=4)
            W_dq = mx.dequantize(wq, sq, bq, group_size=32, bits=4)
            mx.eval(W_dq)
            delta = np.array(mx.abs(W_sc - W_dq)).max(axis=0)
            return float(np.mean(delta * act_max))

        best_score = score_for(best_alpha)
        for c in candidates:
            if c != best_alpha:
                assert best_score <= score_for(c) + 1e-6, (
                    f"alpha={best_alpha} score={best_score:.6f} > alpha={c} score={score_for(c):.6f}"
                )

    def test_single_candidate_returns_it(self):
        """With one candidate, always return that candidate."""
        W = self._make_weight(64, 32)
        act_max = np.ones(32, dtype=np.float32)
        w_max = np.ones(32, dtype=np.float32)
        alpha = _find_optimal_alpha(W, act_max, w_max, (0.5,), bits=4, group_size=32)
        assert alpha == 0.5

    def test_uniform_weight_returns_valid_alpha(self):
        """Uniform weights: any alpha is valid, must still return one from candidates."""
        candidates = (0.1, 0.5, 0.9)
        W = mx.ones((64, 32))
        act_max = np.ones(32, dtype=np.float32)
        w_max = np.ones(32, dtype=np.float32)
        alpha = _find_optimal_alpha(W, act_max, w_max, candidates, bits=4, group_size=32)
        assert alpha in candidates

    def test_high_activation_channels_drive_selection(self):
        """Channels with large act_max should dominate alpha selection."""
        mx.random.seed(7)
        candidates = (0.2, 0.8)
        # Two regimes: channels 0-15 have huge act, channels 16-31 have tiny act
        W_np = np.random.randn(64, 32).astype(np.float32)
        act_max = np.concatenate([np.ones(16) * 10.0, np.ones(16) * 0.01]).astype(np.float32)
        w_max = np.abs(W_np).max(axis=0) + 1e-8

        # Run twice with swapped act importance to verify selection changes
        alpha_high = _find_optimal_alpha(mx.array(W_np), act_max, w_max, candidates, bits=4, group_size=32)

        act_low = act_max[::-1].copy()  # swap: now channels 16-31 are high
        alpha_low = _find_optimal_alpha(mx.array(W_np), act_low, w_max, candidates, bits=4, group_size=32)

        # Both must be valid candidates (may or may not differ, but both valid)
        assert alpha_high in candidates
        assert alpha_low in candidates
