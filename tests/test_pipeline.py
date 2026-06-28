from unittest.mock import patch, MagicMock
from atlas.core.pipeline import Pipeline, CompressionResult
from atlas.profile.hardware import HardwareSpec
from atlas.core.model import ModelInfo


def _mock_hw():
    return HardwareSpec(
        platform="darwin", chip="Apple M1", ram_total_gb=16.0,
        ram_available_gb=12.0, gpu_vendor="apple", gpu_cores=8, cpu_cores=8,
    )


def _mock_model():
    return ModelInfo(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_params=1_100_000_000, num_layers=22,
        size_fp16_gb=2.2, architecture="LlamaForCausalLM",
        exists_locally=False,
    )


def test_pipeline_run_returns_result():
    pipeline = Pipeline()
    with patch.object(pipeline._profiler, "detect", return_value=_mock_hw()), \
         patch.object(pipeline._profiler, "usable_memory_gb", return_value=11.2), \
         patch.object(pipeline._loader, "load_metadata", return_value=_mock_model()):
        result = pipeline.run("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", 99.0, "mlx")
    assert isinstance(result, CompressionResult)
    assert result.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert result.fits_in_memory is True
    assert result.estimated_bits > 0


def test_pipeline_model_too_large():
    pipeline = Pipeline()
    big_model = ModelInfo(
        model_id="big/model", num_params=70_000_000_000, num_layers=80,
        size_fp16_gb=140.0, architecture="LlamaForCausalLM", exists_locally=False,
    )
    small_hw = HardwareSpec(
        platform="darwin", chip="Apple M1", ram_total_gb=8.0,
        ram_available_gb=6.0, gpu_vendor="apple", gpu_cores=8, cpu_cores=8,
    )
    with patch.object(pipeline._profiler, "detect", return_value=small_hw), \
         patch.object(pipeline._profiler, "usable_memory_gb", return_value=5.6), \
         patch.object(pipeline._loader, "load_metadata", return_value=big_model):
        result = pipeline.run("big/model", "auto", 99.0, "mlx")
    assert result.fits_in_memory is False


def test_estimated_size_smaller_than_fp16():
    pipeline = Pipeline()
    with patch.object(pipeline._profiler, "detect", return_value=_mock_hw()), \
         patch.object(pipeline._profiler, "usable_memory_gb", return_value=11.2), \
         patch.object(pipeline._loader, "load_metadata", return_value=_mock_model()):
        result = pipeline.run("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", 99.0, "mlx")
    assert result.estimated_size_gb < result.model_info.size_fp16_gb
