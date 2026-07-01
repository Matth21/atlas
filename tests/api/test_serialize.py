import json
from pathlib import Path

from atlas.api.serialize import serialize_compression_result
from atlas.core.model import ModelInfo
from atlas.core.pipeline import CompressionResult
from atlas.eval.perplexity import EvalResult
from atlas.pack.mlx_packer import PackageInfo
from atlas.profile.hardware import HardwareSpec
from atlas.quant.mlx_quantizer import QuantResult


def test_serialize_compression_result_is_json_safe():
    result = CompressionResult(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hardware=HardwareSpec(
            platform="darwin", chip="Apple M1", ram_total_gb=16.0,
            ram_available_gb=12.0, gpu_vendor="apple", gpu_cores=8, cpu_cores=8,
        ),
        model_info=ModelInfo(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            num_params=1_100_000_000, num_layers=22,
            size_fp16_gb=2.2, architecture="LlamaForCausalLM",
            exists_locally=False,
        ),
        fits_in_memory=True,
        estimated_bits=4.0,
        estimated_size_gb=0.6,
        quant_result=QuantResult(
            output_path=Path("/tmp/quant"), bits=4, group_size=64,
            original_size_mb=2048.0, quantized_size_mb=600.0,
        ),
        eval_result=EvalResult(
            ppl_baseline=12.5, ppl_quantized=13.0, ppl_delta_pct=4.0,
            num_samples=100, eval_time_s=30.0,
        ),
        package_info=PackageInfo(
            output_path=Path("/tmp/pack"), total_size_mb=600.0, metadata={"bits": 4},
        ),
    )

    serialized = serialize_compression_result(result)

    json.dumps(serialized)  # raises if not JSON-safe
    assert serialized["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert serialized["quant_result"]["output_path"] == "/tmp/quant"
    assert serialized["package_info"]["output_path"] == "/tmp/pack"
    assert serialized["package_info"]["total_size_mb"] == 600.0
    assert serialized["eval_result"]["ppl_delta_pct"] == 4.0
