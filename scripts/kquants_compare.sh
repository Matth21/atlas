#!/usr/bin/env bash
# scripts/kquants_compare.sh — K-quants baseline via llama.cpp
# Uso: scripts/kquants_compare.sh <hf-model-id>
# Prerequisiti: brew install llama.cpp ; .venv con gguf installato
set -euo pipefail

MODEL_ID="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SAFE="${MODEL_ID//\//_}"
WORK="build/kquants/$SAFE"
VENV=".venv/bin/python"
mkdir -p "$WORK" results

# 0. testo eval identico alla pipeline MLX
$VENV scripts/dump_wikitext2_test.py build/wiki2test.txt

# 1. convert HF → gguf f16 (serve lo script del repo llama.cpp)
if [ ! -d build/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp build/llama.cpp
fi
HF_DIR=$($VENV -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID'))")
F16="$WORK/model-f16.gguf"
[ -f "$F16" ] || $VENV build/llama.cpp/convert_hf_to_gguf.py "$HF_DIR" --outfile "$F16" --outtype f16

# 2. quantizza e misura
RESULTS="results/kquants_${SAFE}.json"
[ -f "$RESULTS" ] || $VENV -c "import json,sys; json.dump({'model': sys.argv[1], 'runs': []}, open(sys.argv[2],'w'), indent=2)" "$MODEL_ID" "$RESULTS"
for Q in f16 Q3_K_M Q4_K_M Q5_K_M; do
  if $VENV -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if any(r['quant']==sys.argv[2] for r in d['runs']) else 1)" "$RESULTS" "$Q"; then
    echo "=== $Q già misurato, skip ==="
    continue
  fi
  if [ "$Q" = "f16" ]; then GGUF="$F16";
  else
    GGUF="$WORK/model-$Q.gguf"
    [ -f "$GGUF" ] || llama-quantize "$F16" "$GGUF" "$Q"
  fi
  echo "=== $Q ==="
  # llama-perplexity stampa 'Final estimate: PPL = <val> +/- <err>'
  if ! PPL_LINE=$(llama-perplexity -m "$GGUF" -f build/wiki2test.txt --ctx-size 2048 2>&1 | grep "Final estimate" | tail -1) || [ -z "$PPL_LINE" ]; then
    echo "FAILED: llama-perplexity su $Q ($GGUF)" >&2
    exit 1
  fi
  SIZE=$(stat -f%z "$GGUF")
  $VENV - "$RESULTS" "$Q" "$SIZE" "$PPL_LINE" <<'EOF'
import json, re, sys
path, q, size, line = sys.argv[1:5]
m = re.search(r"PPL = ([\d.]+) \+/- ([\d.]+)", line)
if m is None:
    sys.exit(f"parse fallito per {q}: {line!r}")
d = json.load(open(path))
d["runs"].append({"quant": q, "size_bytes": int(size),
                  "ppl": float(m.group(1)), "ppl_err": float(m.group(2))})
json.dump(d, open(path, "w"), indent=2)
EOF
done
cat "$RESULTS"
