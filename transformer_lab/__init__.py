"""Notebook-friendly helper modules for the transformer research workflow."""

from .data import (
    CorpusRecipe,
    EvalSlice,
    PhaseRecipe,
    TokenizerRecipe,
    build_repo_local_corpus,
    contamination_report,
)
from .dashboards import (
    build_attention_probe_figure,
    build_dashboard_bundle,
    build_moe_dashboard,
    build_next_token_figure,
    build_training_overview,
    load_metrics_rows,
)
from .evals import (
    EvalCase,
    build_default_eval_suite,
    calibration_bins,
    summarize_eval_results,
)
from .modeling import (
    ModelConfig,
    ResearchTransformer,
    active_parameter_count,
    count_parameters,
    estimate_kv_cache_bytes,
)
from .posttrain import (
    InstructionExample,
    PreferenceExample,
    RewardExample,
    build_repo_instruction_set,
    dpo_loss,
    outcome_reward,
    pack_instruction_batch,
)
from .training import (
    FsdpLaunchConfig,
    StageConfig,
    TrainConfig,
    build_stage_plan,
    cosine_with_warmup,
    set_seed,
    write_launch_bundle,
)

__all__ = [
    "CorpusRecipe",
    "EvalCase",
    "EvalSlice",
    "FsdpLaunchConfig",
    "InstructionExample",
    "ModelConfig",
    "PhaseRecipe",
    "PreferenceExample",
    "ResearchTransformer",
    "RewardExample",
    "StageConfig",
    "TokenizerRecipe",
    "TrainConfig",
    "active_parameter_count",
    "build_attention_probe_figure",
    "build_dashboard_bundle",
    "build_default_eval_suite",
    "build_moe_dashboard",
    "build_next_token_figure",
    "build_repo_instruction_set",
    "build_repo_local_corpus",
    "build_stage_plan",
    "build_training_overview",
    "calibration_bins",
    "contamination_report",
    "cosine_with_warmup",
    "count_parameters",
    "dpo_loss",
    "estimate_kv_cache_bytes",
    "load_metrics_rows",
    "outcome_reward",
    "pack_instruction_batch",
    "set_seed",
    "summarize_eval_results",
    "write_launch_bundle",
]
