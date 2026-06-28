import json
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from huggingface_hub import model_info as hf_model_info
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

LAYER_KEYS = ("num_hidden_layers", "n_layer", "n_layers", "num_layers")


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    num_params: int
    num_layers: int
    size_fp16_gb: float
    architecture: str
    exists_locally: bool


class ModelLoader:
    """Fetches model metadata from the HuggingFace Hub."""

    def load_metadata(self, model_id: str) -> ModelInfo:
        try:
            info = hf_model_info(model_id)
        except RepositoryNotFoundError:
            raise ValueError(f"Model '{model_id}' not found on HuggingFace Hub")
        except HfHubHTTPError as exc:
            raise ValueError(f"Model '{model_id}' not found on HuggingFace Hub") from exc

        num_params = self._extract_num_params(info)
        config = self._fetch_config(model_id)
        num_layers = self._extract_num_layers(config)
        architecture = self._extract_architecture(config)
        size_fp16_gb = round((num_params * 2) / (1024 ** 3), 2)
        exists_locally = self._check_local_cache(model_id)

        return ModelInfo(
            model_id=model_id,
            num_params=num_params,
            num_layers=num_layers,
            size_fp16_gb=size_fp16_gb,
            architecture=architecture,
            exists_locally=exists_locally,
        )

    @staticmethod
    def _extract_num_params(info) -> int:
        if getattr(info, "safetensors", None) and info.safetensors.parameters:
            return sum(info.safetensors.parameters.values())
        if getattr(info, "config", None):
            num_params = info.config.get("num_parameters")
            if num_params:
                return int(num_params)
        return 0

    @staticmethod
    def _fetch_config(model_id: str) -> dict:
        try:
            path = hf_hub_download(model_id, "config.json")
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _extract_num_layers(config: dict) -> int:
        for key in LAYER_KEYS:
            if key in config:
                return int(config[key])
        return 0

    @staticmethod
    def _extract_architecture(config: dict) -> str:
        architectures = config.get("architectures") or []
        if architectures:
            return architectures[0]
        return "Unknown"

    @staticmethod
    def _check_local_cache(model_id: str) -> bool:
        from huggingface_hub import scan_cache_dir

        try:
            cache = scan_cache_dir()
            return any(repo.repo_id == model_id for repo in cache.repos)
        except Exception:
            return False
