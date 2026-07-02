"""CostProfiler SGSR-2: tabella costi KL misurati per (blocco, bits, gs).

Per ogni blocco e config fa fake-quant del solo blocco, misura la KL sui
logits rispetto al baseline, ripristina i pesi. Checkpoint incrementale
per riprendere run notturne interrotte.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_lm_load

from atlas.profile.calib import CALIB_SEED, load_calibration
from atlas.profile.kl import LogitsSnapshot, kl_vs_snapshot, snapshot_topk
from atlas.quant.fakequant import apply_fake_quant, quantizable_weights, restore_weights

CACHE_DIR = Path.home() / ".cache" / "atlas"
ALGO_VERSION = 1
CONFIGS: list[tuple[int, int]] = [
    (bits, gs) for bits in (3, 4, 5, 6) for gs in (32, 64, 128)
]
LMHEAD_CONFIGS: list[tuple[int, int]] = [(bits, 64) for bits in (4, 5, 6, 8)]


def _key(bits: int, gs: int) -> str:
    return f"{bits}:{gs}"


@dataclass(frozen=True)
class CostTable:
    model_id: str
    configs: tuple[str, ...]
    block_costs: tuple[dict[str, float], ...]
    block_params: tuple[int, ...]
    lmhead_costs: dict[str, float] | None
    lmhead_params: int
    calib_seed: int
    algo_version: int = ALGO_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CostTable":
        d = json.loads(s)
        return cls(
            model_id=d["model_id"],
            configs=tuple(d["configs"]),
            block_costs=tuple(d["block_costs"]),
            block_params=tuple(d["block_params"]),
            lmhead_costs=d["lmhead_costs"],
            lmhead_params=d["lmhead_params"],
            calib_seed=d["calib_seed"],
            algo_version=d["algo_version"],
        )


class CostProfiler:
    def profile(self, model_dir: str, model_id: str) -> CostTable:
        cache = CACHE_DIR / model_id.replace("/", "_") / "cost_table_v1.json"
        if cache.exists():
            table = CostTable.from_json(cache.read_text())
            if table.algo_version == ALGO_VERSION and table.calib_seed == CALIB_SEED:
                return table
        model, tokenizer = mlx_lm_load(model_dir)
        seqs = load_calibration(tokenizer)
        ckpt = cache.with_suffix(".ckpt.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        table = self.profile_model(model, model_id, seqs, checkpoint_path=ckpt)
        cache.write_text(table.to_json())
        ckpt.unlink(missing_ok=True)
        return table

    def profile_model(
        self,
        model,
        model_id: str,
        token_seqs: list[list[int]],
        checkpoint_path: Path | None = None,
    ) -> CostTable:
        snapshots = [
            snapshot_topk(self._forward(model, seq)) for seq in token_seqs
        ]
        done: dict[str, dict[str, float]] = {}
        if checkpoint_path is not None and checkpoint_path.exists():
            done = json.loads(checkpoint_path.read_text())

        blocks = model.model.layers
        block_costs: list[dict[str, float]] = []
        block_params: list[int] = []
        for l, block in enumerate(blocks):
            block_params.append(
                sum(w.size for _, w in quantizable_weights(block))
            )
            if str(l) in done:
                block_costs.append(done[str(l)])
                continue
            costs = {
                _key(bits, gs): self._measure(model, block, bits, gs, snapshots, token_seqs)
                for bits, gs in CONFIGS
            }
            block_costs.append(costs)
            done[str(l)] = costs
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(json.dumps(done))

        lmhead_costs = None
        lmhead_params = 0
        head = getattr(model, "lm_head", None)
        if head is not None:
            lmhead_params = sum(w.size for _, w in quantizable_weights(head))
            if "lm_head" in done:
                lmhead_costs = done["lm_head"]
            else:
                lmhead_costs = {
                    _key(bits, gs): self._measure(model, head, bits, gs, snapshots, token_seqs)
                    for bits, gs in LMHEAD_CONFIGS
                }
                done["lm_head"] = lmhead_costs
                if checkpoint_path is not None:
                    checkpoint_path.write_text(json.dumps(done))

        return CostTable(
            model_id=model_id,
            configs=tuple(_key(b, g) for b, g in CONFIGS),
            block_costs=tuple(block_costs),
            block_params=tuple(block_params),
            lmhead_costs=lmhead_costs,
            lmhead_params=lmhead_params,
            calib_seed=CALIB_SEED,
        )

    def _measure(self, model, module, bits, gs, snapshots, token_seqs) -> float:
        originals = apply_fake_quant(module, bits=bits, group_size=gs)
        try:
            kls = [
                kl_vs_snapshot(snap, self._forward(model, seq))
                for snap, seq in zip(snapshots, token_seqs)
            ]
        finally:
            restore_weights(module, originals)
        return sum(kls) / len(kls)

    @staticmethod
    def _forward(model, seq: list[int]) -> mx.array:
        return model(mx.array(seq)[None, :]).squeeze(0)
