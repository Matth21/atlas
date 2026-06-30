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
        result = pipeline.run("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", 99.0, "mlx", mode="uniform")
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
        result = pipeline.run("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", 99.0, "mlx", mode="uniform")
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

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="uniform")

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


from atlas.plan.planner import QuantPlan, LayerPlan
from atlas.quant.manual import ManualQuantResult


def _mock_quant_plan():
    layers = tuple(
        LayerPlan(i, f"model.layers.{i}", 4, 64, round(i/9, 2))
        for i in range(10)
    )
    return QuantPlan(
        model_id="test/tiny-1B", layers=layers,
        avg_bits=4.0, estimated_size_gb=0.5, target_bits=4,
    )


class TestPipelineMixedMode:
    def test_mixed_mode_calls_profiler_planner_manual_quantizer(self):
        """Phase 2.5: mode='mixed' chiama profiler, planner e ManualLayerQuantizer."""
        pipeline = Pipeline()
        manual_result = ManualQuantResult(
            output_path=Path("/tmp/manual"),
            plan=_mock_quant_plan(),
            quantized_size_mb=500.0,
            original_size_mb=2000.0,
            bias_corrections=None,
        )

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._layer_profiler, "profile") as mock_profile, \
             patch.object(pipeline._planner, "plan", return_value=_mock_quant_plan()) as mock_plan, \
             patch.object(pipeline._manual_quantizer, "quantize", return_value=manual_result), \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()), \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            from atlas.profile.layers import LayerProfile, LayerSensitivity
            mock_profile.return_value = LayerProfile(
                model_id="test/tiny-1B", num_layers=10,
                sensitivities=tuple(
                    LayerSensitivity(i, f"model.layers.{i}", float(i), round(i/9, 4))
                    for i in range(10)
                ),
                calibration_samples=64,
            )

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="mixed")

        mock_profile.assert_called_once()
        mock_plan.assert_called_once()
        assert result.quant_plan is not None

    def test_mixed_mode_calls_manual_quantizer_not_mixed(self):
        """Phase 2.5: mode='mixed' deve usare ManualLayerQuantizer, non MixedQuantizer."""
        pipeline = Pipeline()
        manual_result = ManualQuantResult(
            output_path=Path("/tmp/manual"),
            plan=_mock_quant_plan(),
            quantized_size_mb=500.0,
            original_size_mb=2000.0,
            bias_corrections=None,
        )

        from atlas.profile.layers import LayerProfile, LayerSensitivity
        mock_layer_profile = LayerProfile(
            model_id="test/tiny-1B", num_layers=10,
            sensitivities=tuple(
                LayerSensitivity(i, f"model.layers.{i}", float(i), round(i/9, 4))
                for i in range(10)
            ),
            calibration_samples=64,
        )

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._layer_profiler, "profile", return_value=mock_layer_profile), \
             patch.object(pipeline._planner, "plan", return_value=_mock_quant_plan()), \
             patch.object(pipeline._manual_quantizer, "quantize", return_value=manual_result) as mock_manual, \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()), \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="mixed")

        mock_manual.assert_called_once()
        assert result.quant_plan is not None

    def test_uniform_mode_skips_profiler_planner(self):
        pipeline = Pipeline()
        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._quantizer, "quantize", return_value=_mock_quant_result()), \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()), \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            result = pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="uniform")

        assert result.quant_plan is None
        assert result.quant_result is not None


class TestPipelinePhase25Params:
    def test_run_accepts_metric_and_compensation_params(self):
        pipeline = Pipeline()

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=100.0), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()):
            result = pipeline.run(
                model_id="test/model",
                target="auto", quality=99.0, output_format="mlx",
                mode="mixed", dry_run=True,
                metric="entropy", enable_compensation=True,
            )

        assert result.metric == "entropy"
        assert result.enable_compensation is True

    def test_run_metric_default_is_entropy(self):
        pipeline = Pipeline()

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=100.0), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()):
            result = pipeline.run(
                model_id="test/model",
                target="auto", quality=99.0, output_format="mlx",
                mode="mixed", dry_run=True,
            )

        assert result.metric == "entropy"

    def test_compression_result_metric_and_compensation_defaults(self):
        """CompressionResult deve avere metric e enable_compensation con defaults corretti."""
        result = CompressionResult(
            model_id="test",
            hardware=_mock_hardware(),
            model_info=_mock_model_info(),
            fits_in_memory=True,
            estimated_bits=4.0,
            estimated_size_gb=0.6,
        )
        assert result.metric == "relative_growth"
        assert result.enable_compensation is False

    def test_metric_passed_to_layer_profiler(self):
        """Pipeline.run() deve passare metric= a layer_profiler.profile()."""
        pipeline = Pipeline()
        manual_result = ManualQuantResult(
            output_path=Path("/tmp/manual"),
            plan=_mock_quant_plan(),
            quantized_size_mb=500.0,
            original_size_mb=2000.0,
            bias_corrections=None,
        )

        from atlas.profile.layers import LayerProfile, LayerSensitivity
        mock_layer_profile = LayerProfile(
            model_id="test/tiny-1B", num_layers=10,
            sensitivities=tuple(
                LayerSensitivity(i, f"model.layers.{i}", float(i), round(i/9, 4))
                for i in range(10)
            ),
            calibration_samples=64,
        )

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._layer_profiler, "profile", return_value=mock_layer_profile) as mock_profile, \
             patch.object(pipeline._planner, "plan", return_value=_mock_quant_plan()), \
             patch.object(pipeline._manual_quantizer, "quantize", return_value=manual_result), \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()), \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="mixed", metric="entropy")

        mock_profile.assert_called_once_with("test/tiny-1B", metric="entropy")

    def test_bias_corrections_none_passed_to_evaluator(self):
        """Pipeline.run() passa bias_corrections=None al valutatore (SmoothQuant bakes corrections)."""
        pipeline = Pipeline()
        manual_result = ManualQuantResult(
            output_path=Path("/tmp/manual"),
            plan=_mock_quant_plan(),
            quantized_size_mb=500.0,
            original_size_mb=2000.0,
            bias_corrections=None,
        )

        from atlas.profile.layers import LayerProfile, LayerSensitivity
        mock_layer_profile = LayerProfile(
            model_id="test/tiny-1B", num_layers=10,
            sensitivities=tuple(
                LayerSensitivity(i, f"model.layers.{i}", float(i), round(i/9, 4))
                for i in range(10)
            ),
            calibration_samples=64,
        )

        with patch.object(pipeline._profiler, "detect", return_value=_mock_hardware()), \
             patch.object(pipeline._profiler, "usable_memory_gb", return_value=22.4), \
             patch.object(pipeline._loader, "load_metadata", return_value=_mock_model_info()), \
             patch.object(pipeline._layer_profiler, "profile", return_value=mock_layer_profile), \
             patch.object(pipeline._planner, "plan", return_value=_mock_quant_plan()), \
             patch.object(pipeline._manual_quantizer, "quantize", return_value=manual_result), \
             patch.object(pipeline._evaluator, "evaluate", return_value=_mock_eval_result()) as mock_eval, \
             patch.object(pipeline._packer, "package", return_value=_mock_package_info()):

            pipeline.run("test/tiny-1B", "auto", 99.0, "mlx", mode="mixed")

        call_kwargs = mock_eval.call_args
        assert call_kwargs.kwargs.get("bias_corrections") is None
