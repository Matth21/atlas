"""Per-layer activation sensitivity profiling for Atlas.

Runs calibration data through a model and records the L2 norm of each
transformer layer's output activations. Layers with higher activation
norms are treated as more sensitive to quantization error, and their
normalized sensitivity score feeds into the QuantPlanner's bit allocation
decisions (Phase 2 mixed-bit quantization).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_lm_load


CACHE_DIR = Path.home() / ".cache" / "atlas"


@dataclass(frozen=True)
class LayerSensitivity:
    layer_index: int
    name: str
    activation_norm: float
    sensitivity_score: float


@dataclass(frozen=True)
class LayerProfile:
    model_id: str
    num_layers: int
    sensitivities: tuple[LayerSensitivity, ...]
    calibration_samples: int


class LayerProfiler:
    """Measures per-layer activation sensitivity via calibration data."""

    def profile(self, model_id: str, num_samples: int = 64) -> LayerProfile:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        cached = self._load_cache(model_id, num_samples)
        if cached is not None:
            return cached

        samples = _load_calibration_samples(num_samples)
        layer_norms = _compute_layer_norms(model_id, samples)

        if not layer_norms:
            raise RuntimeError(f"No layers found in model {model_id}")

        norms = [n for _, n in layer_norms]
        min_norm = min(norms)
        max_norm = max(norms)
        norm_range = max_norm - min_norm if max_norm > min_norm else 1.0

        sensitivities = tuple(
            LayerSensitivity(
                layer_index=i,
                name=name,
                activation_norm=round(norm, 4),
                sensitivity_score=round((norm - min_norm) / norm_range, 4),
            )
            for i, (name, norm) in enumerate(layer_norms)
        )

        result = LayerProfile(
            model_id=model_id,
            num_layers=len(sensitivities),
            sensitivities=sensitivities,
            calibration_samples=len(samples),
        )

        self._save_cache(result)
        return result

    def _cache_path(self, model_id: str) -> Path:
        safe_name = model_id.replace("/", "_")
        return CACHE_DIR / safe_name / "layer_profile.json"

    def _load_cache(self, model_id: str, num_samples: int) -> LayerProfile | None:
        path = self._cache_path(model_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("calibration_samples", 0) >= num_samples:
                sensitivities = tuple(
                    LayerSensitivity(**s) for s in data["sensitivities"]
                )
                return LayerProfile(
                    model_id=data["model_id"],
                    num_layers=data["num_layers"],
                    sensitivities=sensitivities,
                    calibration_samples=data["calibration_samples"],
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

    def _save_cache(self, profile: LayerProfile) -> None:
        path = self._cache_path(profile.model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": profile.model_id,
            "num_layers": profile.num_layers,
            "calibration_samples": profile.calibration_samples,
            "sensitivities": [
                {
                    "layer_index": s.layer_index,
                    "name": s.name,
                    "activation_norm": s.activation_norm,
                    "sensitivity_score": s.sensitivity_score,
                }
                for s in profile.sensitivities
            ],
        }
        path.write_text(json.dumps(data, indent=2))


def _load_calibration_samples(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in ds if row["text"].strip()]
    return texts[:num_samples]


def _compute_layer_norms(
    model_id: str, samples: list[str]
) -> list[tuple[str, float]]:
    model, tokenizer = mlx_lm_load(model_id)

    num_layers = len(model.model.layers)
    layer_norms = [0.0] * num_layers
    total_tokens = 0

    for text in samples:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        input_ids = mx.array(tokens)[None, :]

        # Capture activations by running through layers manually
        x = model.model.embed_tokens(input_ids)
        for i, layer in enumerate(model.model.layers):
            x = layer(x, cache=None)
            norm = mx.sqrt(mx.sum(x * x)).item()
            layer_norms[i] += norm

        total_tokens += 1

    if total_tokens == 0:
        return []

    return [
        (f"model.layers.{i}", layer_norms[i] / total_tokens)
        for i in range(num_layers)
    ]
