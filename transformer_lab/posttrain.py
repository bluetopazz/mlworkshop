from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class InstructionExample:
    prompt: str
    response: str
    source: str


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    source: str


@dataclass
class RewardExample:
    prompt: str
    candidate: str
    reward: float
    source: str


def build_repo_instruction_set(root: Path | str) -> list[InstructionExample]:
    root = Path(root)
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    agents = (root / "agents.md").read_text(encoding="utf-8") if (root / "agents.md").exists() else ""
    examples = [
        InstructionExample(
            prompt="Summarize the purpose of the mlworkshop repository in 4 sentences.",
            response=readme.split("\n", 8)[0:8] and " ".join(line.strip() for line in readme.splitlines()[0:8] if line.strip()),
            source="README.md",
        ),
        InstructionExample(
            prompt="Explain the notebook-first rule for this repository.",
            response="Notebooks remain the primary experimentation surface, while helper modules exist only to keep iteration, debugging, and diagnostics cleaner.",
            source="agents.md",
        ),
        InstructionExample(
            prompt="Describe how a strong small transformer should differ from a toy decoder-only model.",
            response="A strong student-scale model should improve data quality, backbone design, stability, diagnostics, and post-training rather than only increasing parameters.",
            source="transformer_refactor",
        ),
        InstructionExample(
            prompt="Give a checklist for resuming a distributed run safely.",
            response="Verify checkpoint integrity, config compatibility, optimizer state, data pointers, random seeds, and the last completed step before relaunching.",
            source="transformer_refactor",
        ),
    ]
    if agents:
        examples.append(
            InstructionExample(
                prompt="What broader mission should the repository support?",
                response="The work should support long-horizon mastery in scientific ML, transformer systems, and high-impact AI capacity building.",
                source="agents.md",
            )
        )
    return [example for example in examples if example.response]


def pack_instruction_batch(prompt_ids: list[list[int]], response_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(prompt) + len(response) for prompt, response in zip(prompt_ids, response_ids))
    input_rows = []
    target_rows = []
    for prompt, response in zip(prompt_ids, response_ids):
        sequence = prompt + response
        targets = [-100] * len(prompt) + response
        pad_len = max_len - len(sequence)
        input_rows.append(sequence + [pad_id] * pad_len)
        target_rows.append(targets + [-100] * pad_len)
    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(target_rows, dtype=torch.long)


def sequence_logprob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logits = beta * ((policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp))
    return -F.logsigmoid(logits).mean()


def outcome_reward(candidate: str, reference: str | None = None, required_keywords: list[str] | None = None) -> float:
    score = 0.0
    text = candidate.lower()
    if reference:
        ref = reference.lower()
        overlap = len(set(text.split()) & set(ref.split())) / max(len(set(ref.split())), 1)
        score += 0.6 * overlap
    if required_keywords:
        hits = sum(1 for keyword in required_keywords if keyword.lower() in text)
        score += 0.4 * hits / max(len(required_keywords), 1)
    return float(min(score, 1.0))
