import mlx.core as mx
import mlx.nn as nn

from atlas.quant.fakequant import apply_fake_quant, quantizable_weights, restore_weights


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64, bias=False)
        self.norm = nn.RMSNorm(128)

    def __call__(self, x):
        return self.proj(self.norm(x))


def test_quantizable_weights_selects_2d_only():
    toy = Toy()
    names = [n for n, _ in quantizable_weights(toy)]
    assert names == ["proj.weight"]  # norm.weight è 1D → esclusa


def test_fake_quant_changes_weights_and_restore_is_exact():
    toy = Toy()
    original = mx.array(toy.proj.weight)
    saved = apply_fake_quant(toy, bits=4, group_size=32)
    assert not mx.array_equal(toy.proj.weight, original)
    restore_weights(toy, saved)
    assert mx.array_equal(toy.proj.weight, original)


def test_more_bits_less_error():
    def err(bits):
        toy = Toy()
        w0 = mx.array(toy.proj.weight).astype(mx.float32)
        apply_fake_quant(toy, bits=bits, group_size=32)
        return mx.abs(toy.proj.weight.astype(mx.float32) - w0).mean().item()

    mx.random.seed(0)
    assert err(6) < err(3)


def test_all_group_sizes_roundtrip():
    for gs in (32, 64, 128):
        toy = Toy()
        original = mx.array(toy.proj.weight)
        saved = apply_fake_quant(toy, bits=4, group_size=gs)
        assert not mx.array_equal(toy.proj.weight, original), gs
        restore_weights(toy, saved)
        assert mx.array_equal(toy.proj.weight, original), gs
