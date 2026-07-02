import pytest

from atlas.profile.calib import CALIB_NUM_SEQS, CALIB_SEQ_LEN, CALIB_SEED, chunk_tokens


def test_chunk_shapes_and_determinism():
    tokens = list(range(10_000))
    a = chunk_tokens(tokens, seq_len=512, num_seqs=8, seed=42)
    b = chunk_tokens(tokens, seq_len=512, num_seqs=8, seed=42)
    assert len(a) == 8
    assert all(len(seq) == 512 for seq in a)
    assert a == b


def test_chunks_disjoint():
    tokens = list(range(10_000))
    chunks = chunk_tokens(tokens, seq_len=512, num_seqs=8, seed=42)
    seen = set()
    for seq in chunks:
        assert not (set(seq) & seen)
        seen |= set(seq)


def test_different_seed_different_selection():
    tokens = list(range(100_000))
    assert chunk_tokens(tokens, 512, 8, 42) != chunk_tokens(tokens, 512, 8, 7)


def test_short_input_raises():
    with pytest.raises(ValueError, match="disponibili solo 0"):
        chunk_tokens(list(range(100)), seq_len=512, num_seqs=8, seed=42)


def test_not_enough_chunks_raises():
    with pytest.raises(ValueError, match="disponibili solo 2"):
        chunk_tokens(list(range(1024)), seq_len=512, num_seqs=8, seed=42)


def test_exact_multiple_single_chunk():
    chunks = chunk_tokens(list(range(512)), seq_len=512, num_seqs=1, seed=42)
    assert chunks == [list(range(512))]


def test_production_constants_shape():
    tokens = list(range(CALIB_NUM_SEQS * CALIB_SEQ_LEN))
    chunks = chunk_tokens(tokens, CALIB_SEQ_LEN, CALIB_NUM_SEQS, CALIB_SEED)
    assert len(chunks) == 32
    assert all(len(c) == 512 for c in chunks)
