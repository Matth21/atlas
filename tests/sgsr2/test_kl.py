import mlx.core as mx

from atlas.profile.kl import kl_vs_snapshot, snapshot_topk


def _logits(T=16, V=256, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((T, V))


def test_kl_self_is_zero():
    logits = _logits()
    snap = snapshot_topk(logits, k=64)
    assert abs(kl_vs_snapshot(snap, logits)) < 1e-4


def test_kl_nonnegative_and_grows_with_perturbation():
    logits = _logits()
    snap = snapshot_topk(logits, k=64)
    mx.random.seed(1)
    noise = mx.random.normal(logits.shape)
    kl_small = kl_vs_snapshot(snap, logits + 0.1 * noise)
    kl_big = kl_vs_snapshot(snap, logits + 1.0 * noise)
    assert kl_small >= 0.0
    assert kl_big > kl_small


def test_snapshot_shapes():
    snap = snapshot_topk(_logits(T=8, V=128), k=32)
    assert snap.indices.shape == (8, 32)
    assert snap.logprobs.shape == (8, 32)
    assert snap.tail_mass.shape == (8,)
