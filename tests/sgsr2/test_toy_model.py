import mlx.core as mx

from tests.sgsr2.toy_model import ToyModel


def test_toy_model_shape_and_structure():
    mx.random.seed(0)
    model = ToyModel(vocab=64, dim=128, n_blocks=3)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 64)
    assert len(model.model.layers) == 3
    assert model.lm_head is not None
