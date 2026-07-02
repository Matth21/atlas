import math

import mlx.core as mx
import pytest

from atlas.eval.sliding_ppl import ppl_with_ci, sliding_nlls
from tests.sgsr2.toy_model import ToyModel


def test_every_token_scored_once():
    mx.random.seed(0)
    model = ToyModel(vocab=64, dim=128, n_blocks=2)
    tokens = [i % 64 for i in range(300)]
    nlls = sliding_nlls(model, tokens, window=128, stride=32)
    assert len(nlls) == len(tokens) - 1


def test_matches_full_context_when_window_covers_all():
    mx.random.seed(0)
    model = ToyModel(vocab=64, dim=128, n_blocks=2)
    tokens = [i % 64 for i in range(100)]
    a = sliding_nlls(model, tokens, window=128, stride=32)
    b = sliding_nlls(model, tokens, window=4096, stride=512)
    assert math.isclose(sum(a) / len(a), sum(b) / len(b), rel_tol=0.05)


def test_ppl_with_ci_brackets_point_estimate():
    nlls = [1.0, 1.2, 0.8, 1.1, 0.9] * 40
    ppl, lo, hi = ppl_with_ci(nlls, n_boot=500, seed=0)
    assert lo <= ppl <= hi
    assert math.isclose(ppl, math.exp(sum(nlls) / len(nlls)), rel_tol=1e-9)


def test_window_not_greater_than_stride_raises():
    mx.random.seed(0)
    model = ToyModel(vocab=64, dim=128, n_blocks=1)
    tokens = [i % 64 for i in range(300)]
    with pytest.raises(ValueError, match="window > stride"):
        sliding_nlls(model, tokens, window=128, stride=128)
    with pytest.raises(ValueError, match="window > stride"):
        sliding_nlls(model, tokens, window=64, stride=128)
