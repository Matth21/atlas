#!/usr/bin/env python3
"""Dump del test set Wikitext-2 identico a wikitext2_test_tokens (stesso testo)."""
import sys
from pathlib import Path

from datasets import load_dataset

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/wiki2test.txt")
out.parent.mkdir(parents=True, exist_ok=True)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
out.write_text("\n\n".join(row["text"] for row in ds if row["text"].strip()))
print(out, out.stat().st_size, "bytes")
