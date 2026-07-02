#!/usr/bin/env python3
# scripts/plot_pareto.py
"""Plot Pareto ΔPPL% vs bit/w da size reale. Uso: plot_pareto.py <model-safe-name>"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

safe = sys.argv[1]  # es. TinyLlama_TinyLlama-1.1B-Chat-v1.0
pareto = json.load(open(f"results/sgsr2_pareto_{safe}.json"))
base_ppl = pareto["baseline"]["ppl"]

fig, ax = plt.subplots(figsize=(7, 5))

for prefix, style, label in (("sgsr2_", "o-", "SGSR-2"),
                             ("uniform_", "s", "Uniform+SQ")):
    runs = [r for r in pareto["runs"]
            if r["status"] == "ok" and r["name"].startswith(prefix)]
    runs.sort(key=lambda r: r["eff_bits_from_size"])
    x = [r["eff_bits_from_size"] for r in runs]
    y = [r["ppl_delta_pct"] for r in runs]
    yerr = [[(r["ppl"] - r["ci_low"]) / base_ppl * 100 for r in runs],
            [(r["ci_high"] - r["ppl"]) / base_ppl * 100 for r in runs]]
    ax.errorbar(x, y, yerr=yerr, fmt=style, capsize=3, label=label)

kq_path = Path(f"results/kquants_{safe}.json")
if kq_path.exists():
    kq = json.load(open(kq_path))
    f16 = next(r for r in kq["runs"] if r["quant"] == "f16")
    n_params_est = f16["size_bytes"] / 2  # f16 = 2 byte/param
    pts = [r for r in kq["runs"] if r["quant"] != "f16"]
    x = [r["size_bytes"] * 8 / n_params_est for r in pts]
    y = [(r["ppl"] - f16["ppl"]) / f16["ppl"] * 100 for r in pts]
    ax.plot(x, y, "^", label="llama.cpp K-quants")
    for r, xi, yi in zip(pts, x, y):
        ax.annotate(r["quant"], (xi, yi), fontsize=8)

ax.set_xlabel("bit per weight (da size su disco)")
ax.set_ylabel("ΔPPL % vs baseline")
ax.set_title(safe)
ax.legend()
ax.grid(alpha=0.3)
out = Path(f"paper/figures/pareto_{safe}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
