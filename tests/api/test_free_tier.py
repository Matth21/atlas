import pytest

from atlas.api.free_tier import (
    FREE_TIER_CONFIG,
    FREE_TIER_MODELS,
    FreeTierError,
    validate_free_tier_request,
)


def test_curated_model_with_default_config_passes():
    model_id = next(iter(FREE_TIER_MODELS))
    validate_free_tier_request(model_id, dict(FREE_TIER_CONFIG))


def test_non_curated_model_raises():
    with pytest.raises(FreeTierError, match="richiede Pro tier"):
        validate_free_tier_request("meta-llama/Llama-3.1-70B", dict(FREE_TIER_CONFIG))


def test_custom_config_raises():
    model_id = next(iter(FREE_TIER_MODELS))
    custom_config = {**FREE_TIER_CONFIG, "quality": 0.99}
    with pytest.raises(FreeTierError, match="richiede Pro tier"):
        validate_free_tier_request(model_id, custom_config)
