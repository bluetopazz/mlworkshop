from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class EvalCase:
    group: str
    prompt: str
    target: str | None = None
    verifier: str = "keyword"


def build_default_eval_suite() -> list[EvalCase]:
    return [
        EvalCase("reasoning", "Explain why RoPE helps extrapolation better than learned absolute position embeddings."),
        EvalCase("systems", "Describe the tradeoff between dense MLP blocks and sparse MoE blocks in a student-scale LM."),
        EvalCase("science", "Explain why notebook-first experimentation matters in a scientific ML research workflow."),
        EvalCase("instruction", "Write a short checklist for safely resuming distributed training after a crash."),
    ]


def token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="mean")


def perplexity_from_losses(losses: list[float]) -> float:
    if not losses:
        return float("nan")
    return float(math.exp(sum(losses) / len(losses)))


def calibration_bins(confidences: list[float], correct: list[float], num_bins: int = 10) -> list[dict[str, float]]:
    if not confidences:
        return []
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    conf_arr = np.asarray(confidences)
    corr_arr = np.asarray(correct)
    records = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= left) & (conf_arr < right if right < 1.0 else conf_arr <= right)
        if not mask.any():
            continue
        records.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "mean_confidence": float(conf_arr[mask].mean()),
                "mean_accuracy": float(corr_arr[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return records


def summarize_eval_results(records: list[dict]) -> dict[str, object]:
    grouped = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record["score"])
    return {
        "group_scores": {group: float(np.mean(scores)) for group, scores in grouped.items()},
        "overall_score": float(np.mean([record["score"] for record in records])) if records else float("nan"),
        "records": records,
    }
