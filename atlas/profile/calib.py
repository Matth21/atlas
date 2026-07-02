"""Calibration set fisso e riproducibile per il CostProfiler SGSR-2.

La riproducibilità assume anche dataset Wikitext-2 e tokenizer invariati tra run (versioni pinnate), non solo il seed.
"""

import random

CALIB_SEED = 42
CALIB_NUM_SEQS = 32
CALIB_SEQ_LEN = 512
_TEXT_CAP_CHARS = 2_000_000  # ~500k token: margine ampio su 32×512


def chunk_tokens(
    tokens: list[int], seq_len: int, num_seqs: int, seed: int
) -> list[list[int]]:
    chunks = [
        tokens[i : i + seq_len]
        for i in range(0, len(tokens) - seq_len + 1, seq_len)
    ]
    if len(chunks) < num_seqs:
        raise ValueError(
            f"servono {num_seqs} chunk da {seq_len} token, "
            f"disponibili solo {len(chunks)} ({len(tokens)} token)"
        )
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks[:num_seqs]


def load_calibration(
    tokenizer,
    num_seqs: int = CALIB_NUM_SEQS,
    seq_len: int = CALIB_SEQ_LEN,
    seed: int = CALIB_SEED,
) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text[:_TEXT_CAP_CHARS])
    return chunk_tokens(tokens, seq_len, num_seqs, seed)
