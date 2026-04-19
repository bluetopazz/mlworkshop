from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class StageConfig:
    name: str
    stage_type: str
    train_tokens_path: str | None = None
    val_tokens_path: str | None = None
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr_scale: float = 0.1
    warmup_steps: int = 100
    grad_accum_steps: int = 8
    micro_batch_size: int = 2
    seq_len: int = 2048
    eval_interval: int = 50
    save_interval: int = 100
    notes: str = ""


@dataclass
class TrainConfig:
    seed: int = 42
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    use_bf16: bool = True
    activation_checkpointing: bool = True
    log_interval: int = 1
    eval_batches: int = 10
    run_name: str = "small_frontier_dense"
    stage_name: str = "base"
    extra_metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FsdpLaunchConfig:
    module: str = "transformer_lab.fsdp_worker"
    nproc_per_node: int = 1
    standalone: bool = True

    def as_command(self, bundle_path: Path | str) -> list[str]:
        command = ["torchrun"]
        if self.standalone:
            command.append("--standalone")
        command.extend(["--nproc_per_node", str(self.nproc_per_node), "-m", self.module, "--bundle", str(bundle_path)])
        return command


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(step: int, max_steps: int, learning_rate: float, min_lr_scale: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(max_steps - warmup_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    min_lr = learning_rate * min_lr_scale
    return float(min_lr + cosine * (learning_rate - min_lr))


def load_token_array(path: Path | str) -> np.ndarray:
    array = np.load(Path(path), mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"Expected flat token array, got shape {array.shape}")
    return array


def sample_lm_batch(tokens: np.ndarray, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[start : start + seq_len] for start in starts]).astype(np.int64)
    y = np.stack([tokens[start + 1 : start + seq_len + 1] for start in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def build_stage_plan(
    base_train_path: str,
    base_val_path: str,
    adapt_train_path: str,
    adapt_val_path: str,
) -> list[StageConfig]:
    return [
        StageConfig(
            name="base_pretrain",
            stage_type="pretrain",
            train_tokens_path=base_train_path,
            val_tokens_path=base_val_path,
            max_steps=3000,
            learning_rate=3e-4,
            notes="Broad pretraining with repo-local structure kept in the mixture.",
        ),
        StageConfig(
            name="domain_adapt",
            stage_type="adapt",
            train_tokens_path=adapt_train_path,
            val_tokens_path=adapt_val_path,
            max_steps=1200,
            learning_rate=1.5e-4,
            notes="Domain adaptation on reasoning/science/systems-heavy text.",
        ),
        StageConfig(
            name="instruction_sft",
            stage_type="sft",
            max_steps=600,
            learning_rate=8e-5,
            notes="Supervised finetuning on curated instruction data.",
        ),
        StageConfig(
            name="reward_refine",
            stage_type="reward",
            max_steps=300,
            learning_rate=5e-5,
            notes="Lightweight verifier-guided refinement.",
        ),
    ]


def write_launch_bundle(
    path: Path | str,
    model_cfg,
    train_cfg: TrainConfig,
    stage_cfg: StageConfig,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "stage_cfg": asdict(stage_cfg),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
