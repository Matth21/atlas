FREE_TIER_MODELS: frozenset[str] = frozenset(
    {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-3B-Instruct",
    }
)

FREE_TIER_CONFIG: dict = {
    "target": "quality",
    "quality": 99.0,
    "output_format": "mlx",
    "mode": "mixed",
    "sgsrq_mode": True,
}


class FreeTierError(ValueError):
    pass


def validate_free_tier_request(model_id: str, config: dict) -> None:
    if model_id not in FREE_TIER_MODELS:
        raise FreeTierError(
            f"'{model_id}' non è nella lista modelli free tier — richiede Pro tier"
        )
    if config != FREE_TIER_CONFIG:
        raise FreeTierError(
            "config custom non è disponibile su free tier — richiede Pro tier"
        )
