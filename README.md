# Atlas — MLX-Native LLM Compression for Apple Silicon

**Atlas** compresses large language models for Apple Silicon using mixed-precision quantization guided by per-layer activation sensitivity. It runs entirely on-device — no GPU server, no cloud API.

```
atlas compress TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode mixed
```

> **SGSR-2:** per-block joint (bit-width, group-size) allocation driven by *measured* KL cost and an exact Lagrangian solver — the full 3–5 bit/w Pareto frontier from a single overnight on-device profiling run. See `atlas/profile/cost_table.py`, `atlas/plan/pareto.py`, and tests under `tests/sgsr2/`. Controlled experiments showed that the entropy proxy used in SGSR v1 is statistically indistinguishable from random ranking; SGSR-2 replaces it with direct measurement (see the paper's negative-result section).

**Research:** Raviotta, M. (2026). *SGSR-2: Measured-Cost Pareto Allocation of Bit-Width and Group-Size for On-Device LLM Quantization.* Zenodo. https://doi.org/10.5281/zenodo.21190586 — paper source and PDF under [`paper/`](paper/). (v1, superseded: https://doi.org/10.5281/zenodo.21110556)

---

## Why Atlas?

Standard 4-bit quantization (`mlx_lm.convert`) treats every layer equally. Atlas doesn't. Not all transformer layers are equally sensitive to precision loss — quantizing them uniformly wastes bit budget on robust layers and under-protects fragile ones.

Atlas introduces **two orthogonal axes of adaptation**:

1. **Bit-width per layer** — sensitive layers get more bits
2. **Group-size per layer** (SGSR, novel) — sensitive layers get finer quantization scales

---

## Results on TinyLlama 1.1B

PPL delta = `(ppl_quantized - ppl_baseline) / ppl_baseline × 100%`. Lower is better.

| Method | Avg bit/w | PPL delta | Notes |
|---|---|---|---|
| Uniform 4-bit (mlx_lm baseline) | 4.50 | +4.34% | standard `convert` |
| Atlas Phase 2.5 (SmoothQuant) | 4.50 | +3.63% | SmoothQuant only |
| **Atlas SGSR + SmoothQuant** | **4.51** | **+3.28%** | same budget, better quality |
| Atlas quality_mode (4+8-bit) | 5.38 | +1.67% | +18% bit budget |
| Uniform 8-bit | 8.50 | +0.04% | upper bound |

**SGSR beats uniform 4-bit at the same bit budget** — 4.51 vs 4.50 bit/w effective, -1.06pp PPL delta.

---

## Innovations

### 1 — Entropy-based Layer Sensitivity

Atlas profiles each transformer layer by running calibration data (16 samples from WikiText-2) and computing the Shannon entropy of output activations:

```
H(layer) = -Σ p·log₂(p)    over 256-bin histogram of activations
```

High-entropy layers have richer, more unpredictable activation distributions — they contribute more signal and are more sensitive to quantization error. This replaces the simpler `|out_norm − in_norm| / in_norm` metric used in Phase 2.

Results are cached to `~/.cache/atlas/<model>/layer_profile_entropy.json` with version key to invalidate stale cache automatically.

### 2 — SmoothQuant (pre-quantization channel scaling)

For each LayerNorm → Linear pair, Atlas computes per-channel smooth scales:

```
s_j = max(|act_j|)^α / max(|W[:,j]|)^(1−α)    α = 0.5
```

Applied as:
- `ln_weight[j] /= s_j` — reduces activation outliers
- `W[:, j] *= s_j` — compensates on the weight side

Forward pass is mathematically invariant. Weights become more uniform → quantization error decreases.

### 3 — SGSR: Sensitivity-Guided Group-Size Redistribution *(novel)*

Standard quantization tools (AWQ, GPTQ, JANG) vary **bit-width** per layer. Atlas SGSR varies **group-size** — a completely orthogonal axis.

In affine quantization, each group of `g` consecutive weights shares one scale + one zero-point. Smaller groups → finer scales → lower quantization error, but higher storage overhead.

Atlas SGSR assigns group-size by sensitivity tier:

```
Top 15% most sensitive layers  →  group_size = 32   (+overhead, better quality)
Middle 70%                     →  group_size = 64   (MLX default)
Bottom 25% least sensitive     →  group_size = 128  (-overhead, acceptable quality)
```

Effective bit/w per tier (4-bit affine, bf16 scales):
- gs=32  → ~4.63 bit/w
- gs=64  → ~4.50 bit/w  ← uniform baseline
- gs=128 → ~4.16 bit/w

Weighted average (15%/70%/25% split): **~4.51 bit/w** — nearly identical to uniform 4-bit, with quality gains on sensitive layers that outweigh losses on insensitive ones.

**Sweep results** (TinyLlama 1.1B, 22 layers):

| fine% | coarse% | bit/w | PPL delta |
|---|---|---|---|
| 30% | 15% | 4.59 | +3.92% |
| 23% | 23% | 4.55 | +3.63% |
| **15%** | **25%** | **4.51** | **+3.28%** ← optimum |
| 10% | 30% | 4.48 | +3.34% |

**Literature comparison:** JANG (jangq.ai, 2025) mixes bit-widths per layer type on MLX. SFMP (arXiv 2602.01027) uses block-wise fractional bits. Entropy-guided mixed precision (Scientific Reports 2025) allocates bit-widths via entropy. None combine sensitivity-guided group-size redistribution with SmoothQuant on MLX native.

---

## Architecture

```
atlas/
├── atlas/
│   ├── core/
│   │   ├── model.py          # ModelInfo: fetch HF metadata
│   │   └── pipeline.py       # Pipeline.run() — orchestrates all phases
│   ├── profile/
│   │   ├── hardware.py       # Apple Silicon chip / RAM detection
│   │   └── layers.py         # LayerProfiler: entropy sensitivity (ALGO_VERSION=3)
│   ├── plan/
│   │   └── planner.py        # QuantPlanner: bit-width + SGSR group-size allocation
│   ├── quant/
│   │   ├── mlx_quantizer.py  # Uniform quantization (mlx_lm.convert wrapper)
│   │   ├── mixed.py          # MixedQuantizer: per-layer quant_predicate
│   │   ├── manual.py         # ManualLayerQuantizer: SmoothQuant → MixedQuantizer
│   │   └── smooth.py         # SmoothQuant: channel scaling, saves smoothed model
│   ├── eval/
│   │   └── perplexity.py     # PPL on WikiText-2 test split (float32 accumulation)
│   └── pack/
│       └── mlx_packer.py     # Output: quantized model + metadata.json
├── cli/
│   └── main.py               # typer CLI: atlas compress <model_id>
└── tests/                    # 93 unit + 6 slow E2E ablation tests
```

---

## Ablation Study

6 variants tested on TinyLlama/TinyLlama-1.1B-Chat-v1.0 (100 WikiText-2 samples):

| Variant | Metric | SmoothQuant | SGSR | PPL delta |
|---|---|---|---|---|
| A — baseline (Phase 2.1) | relative_growth | ✗ | ✗ | +13.27% |
| B — entropy only | entropy | ✗ | ✗ | ~+13.27% |
| C — smooth only | relative_growth | ✓ | ✗ | <+13% |
| D — entropy + smooth | entropy | ✓ | ✗ | +12.35% |
| **E — SGSR + smooth** | **entropy** | **✓** | **✓** | **+3.28%** |
| quality_mode (4+8-bit) | entropy | ✓ | ✗ | +1.67% (+18% bit) |

All variants: `pytest tests/test_e2e.py -v -s` (requires HuggingFace access, ~3 min each).

---

## Installation

```bash
git clone <repo>
cd atlas
uv venv && uv pip install -e ".[dev]"

# or
pip install -e ".[dev]"
```

Requires: Python ≥ 3.11, Apple Silicon (M1+), MLX ≥ 0.18.

---

## Usage

```bash
# Uniform 4-bit
atlas compress TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode uniform

# Mixed 4-bit with SGSR + SmoothQuant (best quality at same bit budget)
atlas compress TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode mixed

# Dry run (no quantization, just hardware + model check)
atlas compress TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode mixed --dry-run
```

Output: `./atlas-output/<model>-mlx-smooth-avg4bit/` containing the quantized model + `metadata.json`:

```json
{
  "atlas_version": "0.1.0",
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "bits": 4,
  "ppl_baseline": 16.08,
  "ppl_quantized": 16.61,
  "ppl_delta_pct": 3.28,
  "mixed_quant": {
    "avg_bits": 4.0,
    "layers": [
      {"layer_index": 0, "bits": 4, "group_size": 128},
      {"layer_index": 17, "bits": 4, "group_size": 32},
      ...
    ]
  }
}
```

---

## Tests

```bash
# Unit tests (fast, no model download)
pytest tests/ -v -m "not slow"

# Full ablation (requires TinyLlama, ~15 min)
pytest tests/test_e2e.py -v -s
```

Current: **93 unit tests + 6 E2E ablation tests**, all green.

---

## Technical Notes

- **MLX lazy evaluation**: all arrays must be `mx.eval()`'d before the model goes out of scope — otherwise the computation graph references freed memory and produces garbage output
- **bfloat16 → numpy**: use `np.array(x.astype(mx.float32).flatten())` — numpy does not implement the PEP 3118 buffer protocol for bf16
- **PPL accumulation**: must be in `float32` — bf16 accumulation silently inflates PPL delta by 3–10x
- **Cache invalidation**: layer profiles are cached with `ALGO_VERSION` key — bump on any metric change
- **SmoothQuant cleanup**: smoothed model saved to tempdir, cleaned up even on quantizer exception (`try/finally` with `shutil.rmtree`)
- **2-bit quantization**: available in `{2, 4, 8}` VALID_BITS but catastrophic for quality — +29% PPL delta at 25% tail, +13% at 10% tail. Disabled in all production modes.

---

## Roadmap

- [ ] **QI-SmoothQuant**: two-pass per-layer smooth scale optimization using observed quantization error (not just activation statistics) — novel, not in literature
- [ ] **SGSR + quality_mode combined**: fine group_size for sensitive layers AND 8-bit for top 5%
- [ ] **Larger models**: Qwen2.5-7B, Mistral-7B validation
- [ ] **GGUF export**: parallel output format for llama.cpp / Ollama
- [ ] **Streaming quantization**: process layers sequentially without full model in RAM

---

## Pre-computed cost tables

Profiling a model takes hours (one-time, then cached). To skip it, this repo ships pre-computed cost tables under [`cost_tables/`](cost_tables/). Install them with:

```bash
mkdir -p ~/.cache/atlas/Qwen_Qwen2.5-7B-Instruct
cp cost_tables/Qwen_Qwen2.5-7B-Instruct.json ~/.cache/atlas/Qwen_Qwen2.5-7B-Instruct/cost_table_v1.json
```

Then `atlas Qwen/Qwen2.5-7B-Instruct --budget-gb 4` produces a plan in seconds and a quantized model in minutes. Tables available: TinyLlama-1.1B-Chat, Qwen2.5-7B-Instruct (more coming; contributions welcome).

---

## License

Atlas is available under a tri-license model (Redis 8 style). Users may choose any one of:

- **RSALv2** — Redis Source Available License 2.0
- **SSPLv1** — Server Side Public License v1
- **AGPLv3** — GNU Affero General Public License v3

The source code is fully available and may be used, modified, and redistributed under any of the licenses above. Offering Atlas to third parties as a managed or hosted service requires compliance with the copyleft terms of SSPLv1/AGPLv3, or a separate commercial license.

Full license texts: [LICENSE.txt](LICENSE.txt). Commercial licensing inquiries: raviottamatthias@gmail.com
