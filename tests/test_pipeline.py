from pathlib import Path
from unittest.mock import patch, MagicMock
from atlas.core.pipeline import Pipeline, CompressionResult
from atlas.profile.hardware import HardwareSpec
from atlas.core.model import ModelInfo
from atlas.quant.mlx_quantizer import QuantResult
from atlas.eval.perplexity import EvalResult
from atlas.pack.mlx_packer import PackageInfo


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
         patch.object(pipeline._loader, "load_metadata", return_value=_mock_model()), \
         patch.object(pipeline._quantizer, "quantize", return_value=QuantResult(
             output_path=Path("/tmp/quant"), bits=4, group_size=64,
             original_size_mb=2048.0, quantized_size_mb=600.0,
         )), \
         patch.object(pipeline._evaluator, "evaluate", return_value=EvalResult(
             ppl_baseline=12.5, ppl_quantized=13.0, ppl_delta_pct=4.0,
             num_samples=100, eval_time_s=30.0,
         )), \
         patch.object(pipeline._packer, "package", return_value=PackageInfo(
             output_path=Path("/tmp/output"), total_size_mb=600.0, metadata={},
         )):
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
         patch.object(pipeline._loader, "load_metadata", return_value=_mock_model()), \
         patch.object(pipeline._quantizer, "quantize", return_value=QuantResult(
             output_path=Path("/tmp/quant"), bits=4, group_size=64,
             original_size_mb=2048.0, quantized_size_mb=600.0,
         )), \
         patch.object(pipeline._evaluator, "evaluate", return_value=EvalResult(
             ppl_baseline=12.5, ppl_quantized=13.0, ppl_delta_pct=4.0,
             num_samples=100, eval_time_s=30.0,
         )), \
         patch.object(pipeline._packer, "package", return_value=PackageInfo(
             output_path=Path("/tmp/output"), total_size_mb=600.0, metadata={},
         )):
        result = pipeline.run("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", 99.0, "mlx")
    assert result.estimated_size_gb < result.model_info.size_fp16_gb


def _mock_hardware():
    return HardwareSpec(
        platform="darwin",
        chip="Apple M1 Pro",
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpu_vendor="apple",
        gpu_cores=16,
        cpu_cores=10,
    )


def _mock_model_info():
    return ModelInfo(
        model_id="test/tiny-1B",
        num_params=1_100_000_000,
        num_layers=22,
        size_fp16_gb=2.05,
        architecture="LlamaForCausalLM",
        exists_locally=False,
    )


def _mock_quant_result():
    return QuantResult(
        output_path=Path("/tmp/quant"),
        bits=4,
        group_size=64,
        original_size_mb=2048.0,
        quantized_size_mb=600.0,
    )


def _mock_eval_result():
    return EvalResult(
        ppl_baseline=12.5,
        ppl_quantized=13.0,
        ppl_delta_pct=4.0,
        num_samples=100,
        eval_time_s=30.0,
    )


def _mock_package_info():
    return PackageInfo(
        output_path=Path("/tmp/output/tiny-1B-4bit-atlas"),
        total_size_mb=600.0,
        metadata={"bits": 4},
    )


class TestCompressionResult:
    def test_new_fields_default_none(self):
        result = CompressionResult(
            model_id="test",
            hardware=_mock_hardware(),
            model_info=_mock_model_info(),
            fits_in_memory=True,
            estimated_bits=4.0,
            estimated_size_gb=0.6,
        )
        assert result.quant_result is None
        assert result.eval_result is None
        assert result.package_info is None

    def test_new_fields_populated(self):
        result = CompressionResult(
            model_id="test",
            hardware=_mock_hardware(),
            model_info=_mock_model_info(),
            fits_in_memory=True,
            estimated_bits=4.0,
            estimated_size_gb=0.6,
            quant_result=_mock_quant_result(),
            eval_result=_mock_eval_result(),
            package_info=_mock_package_info(),
        )
        assert result.quant_result.bits == 4
        assert result.eval_result.ppl_delta_pct == 4.0
        assert result.package_info.total_size_mb == 600.0


class TestPipelineRun:
    def test_dry_run_skips_quantization(self):
        pipeline = Pipeline()
        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()):

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", dry_run=True)

        assert result.quant_result is None
        assert result.eval_result is None
        assert result.package_info is None
        assert result.fits_in_memory is True

    def test_run_calls_quantize_eval_pack(self):
        pipeline = Pipeline()
        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._quantizer, "quantize", return_value=_mock_quant_result()), \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()), \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx")

        assert result.quant_result is not None
        assert result.eval_result is not None
        assert result.package_info is not None

    def test_run_skips_quant_if_doesnt_fit(self):
        pipeline = Pipeline()
        big_model = ModelInfo(
            model_id="test/huge-70B",
            num_params=70_000_000_000,
            num_layers=80,
            size_fp16_gb=140.0,
            architecture="LlamaForCausalLM",
            exists_locally=False,
        )
        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=big_model):

            result = pipeline.run("test/huge-70B", "auto", 99.0, "mlx")

        assert result.fits_in_memory is False
        assert result.quant_result is None
