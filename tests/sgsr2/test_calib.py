from atlas.profile.calib import chunk_tokens


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
