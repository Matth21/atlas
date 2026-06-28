import json
import pytest
from pathlib import Path

from atlas.pack.mlx_packer import MLXPacker, PackageInfo
from atlas.quant.mlx_quantizer import QuantResult
from atlas.eval.perplexity import EvalResult
from atlas.profile.hardware import HardwareSpec


def _make_quant_result(tmp_path: Path) -> QuantResult:
    return QuantResult(
        output_path=tmp_path / "quantized",
        bits=4,
        group_size=64,
        original_size_mb=2048.0,
        quantized_size_mb=512.0,
    )


def _make_eval_result() -> EvalResult:
    return EvalResult(
        ppl_baseline=12.5,
        ppl_quantized=13.1,
        ppl_delta_pct=4.8,
        num_samples=100,
        eval_time_s=30.0,
    )


def _make_hardware_spec() -> HardwareSpec:
    return HardwareSpec(
        platform="darwin",
        chip="Apple M1 Pro",
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpu_vendor="apple",
        gpu_cores=16,
        cpu_cores=10,
    )


class TestPackageInfo:
    def test_package_info_fields(self):
        info = PackageInfo(
            output_path=Path("/tmp/out"),
            total_size_mb=512.0,
            metadata={"bits": 4},
        )
        assert info.output_path == Path("/tmp/out")
        assert info.total_size_mb == 512.0
        assert info.metadata["bits"] == 4

    def test_package_info_is_frozen(self):
        info = PackageInfo(
            output_path=Path("/tmp/out"),
            total_size_mb=512.0,
            metadata={},
        )
        with pytest.raises(AttributeError):
            info.total_size_mb = 1.0


class TestMLXPacker:
    def test_package_creates_output_dir(self, tmp_path):
        quant_path = tmp_path / "quantized"
        quant_path.mkdir()
        (quant_path / "model.safetensors").write_bytes(b"\x00" * 2048)
        (quant_path / "config.json").write_text('{"model_type": "llama"}')
        (quant_path / "tokenizer.json").write_text('{}')

        output_base = tmp_path / "atlas-output"
        packer = MLXPacker(output_base=output_base)
        result = packer.package(
            quantized_path=quant_path,
            model_id="test/model-1B",
            quant_result=_make_quant_result(tmp_path),
            eval_result=_make_eval_result(),
            hardware=_make_hardware_spec(),
        )

        assert result.output_path.exists()
        assert (result.output_path / "metadata.json").exists()
        assert (result.output_path / "model.safetensors").exists()

    def test_package_metadata_contents(self, tmp_path):
        quant_path = tmp_path / "quantized"
        quant_path.mkdir()
        (quant_path / "model.safetensors").write_bytes(b"\x00" * 2048)
        (quant_path / "config.json").write_text('{"model_type": "llama"}')

        output_base = tmp_path / "atlas-output"
        packer = MLXPacker(output_base=output_base)
        result = packer.package(
            quantized_path=quant_path,
            model_id="test/model-1B",
            quant_result=_make_quant_result(tmp_path),
            eval_result=_make_eval_result(),
            hardware=_make_hardware_spec(),
        )

        meta = json.loads((result.output_path / "metadata.json").read_text())
        assert meta["atlas_version"] == "0.1.0"
        assert meta["model_id"] == "test/model-1B"
        assert meta["bits"] == 4
        assert meta["group_size"] == 64
        assert meta["ppl_delta_pct"] == 4.8
        assert meta["hardware"]["chip"] == "Apple M1 Pro"
        assert "timestamp" in meta

    def test_package_copies_all_files(self, tmp_path):
        quant_path = tmp_path / "quantized"
        quant_path.mkdir()
        (quant_path / "model.safetensors").write_bytes(b"\x00" * 1024)
        (quant_path / "config.json").write_text("{}")
        (quant_path / "tokenizer.json").write_text("{}")
        (quant_path / "tokenizer_config.json").write_text("{}")

        output_base = tmp_path / "atlas-output"
        packer = MLXPacker(output_base=output_base)
        result = packer.package(
            quantized_path=quant_path,
            model_id="org/model",
            quant_result=_make_quant_result(tmp_path),
            eval_result=_make_eval_result(),
            hardware=_make_hardware_spec(),
        )

        for fname in ["model.safetensors", "config.json", "tokenizer.json",
                       "tokenizer_config.json", "metadata.json"]:
            assert (result.output_path / fname).exists(), f"Missing {fname}"
