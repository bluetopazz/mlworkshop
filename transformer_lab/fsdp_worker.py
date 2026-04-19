from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .modeling import ModelConfig, ResearchTransformer, TransformerBlock
from .training import TrainConfig, StageConfig, cosine_with_warmup, load_token_array, sample_lm_batch, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notebook-friendly FSDP worker for the refactored transformer stack.")
    parser.add_argument("--bundle", required=True, help="Path to a launch bundle JSON written by the notebook.")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    model_cfg = ModelConfig(**bundle["model_cfg"])
    train_cfg = TrainConfig(**bundle["train_cfg"])
    stage_cfg = StageConfig(**bundle["stage_cfg"])

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(train_cfg.seed + rank)

    if stage_cfg.stage_type not in {"pretrain", "adapt"}:
        if rank == 0:
            print(f"Stage type {stage_cfg.stage_type!r} is not distributed in this worker yet; notebook-local path should be used instead.")
        cleanup()
        return

    train_tokens = load_token_array(stage_cfg.train_tokens_path)
    val_tokens = load_token_array(stage_cfg.val_tokens_path)

    model = ResearchTransformer(model_cfg)
    if train_cfg.activation_checkpointing:
        wrapper = lambda module: checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=lambda submodule: isinstance(submodule, TransformerBlock))

    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
        module, recurse, nonwrapped_numel, transformer_layer_cls={TransformerBlock}
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16 if train_cfg.use_bf16 and torch.cuda.is_available() else torch.float32,
        reduce_dtype=torch.bfloat16 if train_cfg.use_bf16 and torch.cuda.is_available() else torch.float32,
        buffer_dtype=torch.bfloat16 if train_cfg.use_bf16 and torch.cuda.is_available() else torch.float32,
    )
    model = FSDP(
        model.to(device),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device if device.type == "cuda" else None,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    run_dir = Path("artifacts_transformer") / train_cfg.run_name / stage_cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    ckpt_path = run_dir / "checkpoint_latest.pt"

    if rank == 0:
        print(f"Starting stage={stage_cfg.name} on world_size={world_size}")

    def append_metric(record: dict) -> None:
        if rank != 0:
            return
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def evaluate() -> float:
        losses = []
        model.eval()
        with torch.no_grad():
            for _ in range(train_cfg.eval_batches):
                x, y = sample_lm_batch(val_tokens, stage_cfg.micro_batch_size, stage_cfg.seq_len, device)
                out = model(x, targets=y, capture_diagnostics=True)
                losses.append(out["loss"].detach())
        loss_tensor = torch.stack(losses).mean()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        model.train()
        return float((loss_tensor / world_size).item())

    start_time = time.time()
    for step in range(stage_cfg.max_steps):
        lr = cosine_with_warmup(step, stage_cfg.max_steps, stage_cfg.learning_rate, stage_cfg.min_lr_scale, stage_cfg.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_diag = {}
        for _ in range(stage_cfg.grad_accum_steps):
            x, y = sample_lm_batch(train_tokens, stage_cfg.micro_batch_size, stage_cfg.seq_len, device)
            out = model(x, targets=y, capture_diagnostics=True)
            loss = (out["loss"] + out["aux_loss"]) / stage_cfg.grad_accum_steps
            loss.backward()
            total_loss += float(loss.detach().item())
            last_diag = out["diagnostics"][-1] if out["diagnostics"] else {}

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip).item())
        optimizer.step()

        val_loss = None
        if step % stage_cfg.eval_interval == 0:
            val_loss = evaluate()

        if step % train_cfg.log_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_seen = (step + 1) * stage_cfg.micro_batch_size * stage_cfg.seq_len * stage_cfg.grad_accum_steps * world_size
            record = {
                "step": step,
                "stage": stage_cfg.name,
                "train_loss": total_loss,
                "val_loss": val_loss,
                "grad_norm": grad_norm,
                "lr": lr,
                "tokens_per_sec": tokens_seen / elapsed,
                "max_memory_gib": (torch.cuda.max_memory_reserved(device) / 1024**3) if device.type == "cuda" else 0.0,
                **{key: value for key, value in last_diag.items() if isinstance(value, float)},
            }
            append_metric(record)
            if rank == 0:
                print(record)

        if step > 0 and step % stage_cfg.save_interval == 0:
            dist.barrier()
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state = model.state_dict()
            if rank == 0:
                torch.save(
                    {
                        "model_state_dict": state,
                        "model_cfg": bundle["model_cfg"],
                        "train_cfg": bundle["train_cfg"],
                        "stage_cfg": bundle["stage_cfg"],
                        "step": step,
                    },
                    ckpt_path,
                )
            dist.barrier()

    cleanup()


if __name__ == "__main__":
    main()
