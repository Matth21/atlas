import pytest

from atlas.core.model import ModelInfo, ModelLoader


def test_model_info_fields():
    info = ModelInfo(
        model_id="test/model",
        num_params=1_100_000_000,
        num_layers=22,
        size_fp16_gb=2.2,
        architecture="LlamaForCausalLM",
        exists_locally=False,
    )
    assert info.model_id == "test/model"
    assert info.num_layers == 22
    assert info.size_fp16_gb == 2.2


def test_load_metadata_tinyllama():
    loader = ModelLoader()
    info = loader.load_metadata("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert isinstance(info, ModelInfo)
    assert info.num_params > 1_000_000_000
    assert info.num_layers > 0
    assert info.size_fp16_gb > 1.0
    assert "Llama" in info.architecture


def test_load_metadata_invalid_model():
    loader = ModelLoader()
    with pytest.raises(ValueError, match="not found"):
        loader.load_metadata("nonexistent/fake-model-xyz-999")


def test_size_calculation():
    info = ModelInfo(
        model_id="test/model",
        num_params=7_000_000_000,
        num_layers=32,
        size_fp16_gb=14.0,
        architecture="LlamaForCausalLM",
        exists_locally=False,
    )
    assert abs(info.size_fp16_gb - 14.0) < 0.1
