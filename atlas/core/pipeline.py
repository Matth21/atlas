from dataclasses import dataclass

from atlas.profile.hardware import HardwareProfiler, HardwareSpec
from atlas.core.model import ModelLoader, ModelInfo


@dataclass(frozen=True)
class CompressionResult:
    model_id: str
    hardware: HardwareSpec
    model_info: ModelInfo
    fits_in_memory: bool
    estimated_bits: float
    estimated_size_gb: float


class Pipeline:
    def __init__(self):
        self._profiler = HardwareProfiler()
        self._loader = ModelLoader()

    def run(
        self, model_id: str, target: str, quality: float, output_format: str
    ) -> CompressionResult:
        hardware = self._profiler.detect()
        usable_gb = self._profiler.usable_memory_gb()
        model_info = self._loader.load_metadata(model_id)

        target_bits = self._estimate_bits(quality)
        estimated_size_gb = (model_info.num_params * target_bits / 8) / (1024 ** 3)
        fits = estimated_size_gb <= usable_gb

        return CompressionResult(
            model_id=model_id,
            hardware=hardware,
            model_info=model_info,
            fits_in_memory=fits,
            estimated_bits=target_bits,
            estimated_size_gb=round(estimated_size_gb, 2),
        )

    def _estimate_bits(self, quality: float) -> float:
        if quality >= 99.5:
            return 5.0
        elif quality >= 99.0:
            return 4.0
        elif quality >= 98.0:
            return 3.5
        elif quality >= 96.0:
            return 3.0
        else:
            return 2.5
