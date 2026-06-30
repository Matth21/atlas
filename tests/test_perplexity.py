import pytest
import math
from unittest.mock import patch, MagicMock
from pathlib import Path

from atlas.eval.perplexity import PerplexityEval, EvalResult


class TestEvalResult:
    def test_eval_result_fields(self):
        result = EvalResult(
            ppl_baseline=12.5,
            ppl_quantized=13.0,
            ppl_delta_pct=4.0,
            num_samples=100,
            eval_time_s=25.3,
        )
        assert result.ppl_baseline == 12.5
        assert result.ppl_quantized == 13.0
        assert result.ppl_delta_pct == 4.0
        assert result.num_samples == 100

    def test_eval_result_is_frozen(self):
        result = EvalResult(
            ppl_baseline=12.5,
            ppl_quantized=13.0,
            ppl_delta_pct=4.0,
            num_samples=100,
            eval_time_s=25.3,
        )
        with pytest.raises(AttributeError):
            result.ppl_baseline = 20.0


class TestPerplexityEval:
    def test_invalid_num_samples_raises(self):
        evaluator = PerplexityEval()
        with pytest.raises(ValueError, match="num_samples must be"):
            evaluator.evaluate(Path("/fake"), "model", num_samples=0)

    @patch("atlas.eval.perplexity._load_wikitext_samples")
    @patch("atlas.eval.perplexity._compute_perplexity")
    def test_evaluate_computes_delta(self, mock_ppl, mock_wiki):
        mock_wiki.return_value = ["sample text"] * 10
        # First call = baseline (FP16), second call = quantized
        mock_ppl.side_effect = [10.0, 11.0]

        evaluator = PerplexityEval()
        result = evaluator.evaluate(
            Path("/fake/quantized"), "some/model", num_samples=10
        )

        assert result.ppl_baseline == 10.0
        assert result.ppl_quantized == 11.0
        assert result.ppl_delta_pct == pytest.approx(10.0)
        assert result.num_samples == 10
        assert result.eval_time_s >= 0

    @patch("atlas.eval.perplexity._load_wikitext_samples")
    @patch("atlas.eval.perplexity._compute_perplexity")
    def test_evaluate_zero_baseline_handled(self, mock_ppl, mock_wiki):
        mock_wiki.return_value = ["text"] * 5
        mock_ppl.side_effect = [0.0, 5.0]

        evaluator = PerplexityEval()
        result = evaluator.evaluate(Path("/fake"), "model", num_samples=5)
        assert math.isinf(result.ppl_delta_pct)


class TestPerplexityEvalWithCompensation:
    def test_evaluate_accepts_bias_corrections_param(self):
        """evaluate() deve accettare bias_corrections senza errori."""
        import mlx.core as mx
        from unittest.mock import patch

        evaluator = PerplexityEval()
        fake_biases = (mx.zeros((64,)), mx.zeros((64,)))

        with patch("atlas.eval.perplexity._load_wikitext_samples", return_value=["hello world test"]), \
             patch("atlas.eval.perplexity._compute_perplexity", return_value=10.0), \
             patch("atlas.eval.perplexity._compute_perplexity_with_corrections", return_value=11.0) as mock_comp:
            result = evaluator.evaluate(Path("/tmp/q"), "test/model",
                                        num_samples=1, bias_corrections=fake_biases)

        mock_comp.assert_called_once()
        assert result.ppl_quantized == 11.0

    def test_evaluate_without_bias_uses_standard_path(self):
        from unittest.mock import patch

        evaluator = PerplexityEval()
        with patch("atlas.eval.perplexity._load_wikitext_samples", return_value=["hello world"]), \
             patch("atlas.eval.perplexity._compute_perplexity", return_value=10.0) as mock_std, \
             patch("atlas.eval.perplexity._compute_perplexity_with_corrections") as mock_comp:
            evaluator.evaluate(Path("/tmp/q"), "test/model", num_samples=1)

        mock_comp.assert_not_called()
        assert mock_std.call_count == 2  # baseline + quantized
