from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d


def load_metrics_rows(path: Path | str) -> list[dict]:
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _series(rows: list[dict], key: str, default: float | None = None) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, default)
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=float)


def build_training_overview(rows: list[dict]) -> go.Figure:
    steps = _series(rows, "step")
    train = _series(rows, "train_loss")
    val = _series(rows, "val_loss")
    grad = _series(rows, "grad_norm")
    lr = _series(rows, "lr")
    toks = _series(rows, "tokens_per_sec")
    mem = _series(rows, "max_memory_gib", default=None)
    if np.isnan(mem).all():
        mem = _series(rows, "rank0_max_reserved_gib")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Train / Val Loss",
            "Learning Rate",
            "Gradient Norm",
            "Tokens / sec",
            "Memory (GiB)",
            "Instability Gap",
        ),
    )
    fig.add_trace(go.Scatter(x=steps, y=train, mode="lines+markers", name="train_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=val, mode="lines+markers", name="val_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=lr, mode="lines", name="lr"), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=grad, mode="lines", name="grad_norm"), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=toks, mode="lines", name="tokens_per_sec"), row=2, col=2)
    fig.add_trace(go.Scatter(x=steps, y=mem, mode="lines", name="memory_gib"), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=val - train, mode="lines", name="val_minus_train"), row=3, col=2)
    fig.update_layout(height=900, width=1200, title="Transformer Training Overview", template="plotly_white")
    return fig


def build_moe_dashboard(rows: list[dict]) -> go.Figure:
    steps = _series(rows, "step")
    router_entropy = _series(rows, "router_entropy")
    load_max = _series(rows, "expert_usage_max")
    load_min = _series(rows, "expert_usage_min")
    aux = _series(rows, "router_aux_loss")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Router Entropy", "Expert Usage Spread", "Router Aux Loss", "Collapse Gap"))
    fig.add_trace(go.Scatter(x=steps, y=router_entropy, mode="lines", name="router_entropy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=load_max, mode="lines", name="usage_max"), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=load_min, mode="lines", name="usage_min"), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=aux, mode="lines", name="aux_loss"), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=load_max - load_min, mode="lines", name="collapse_gap"), row=2, col=2)
    fig.update_layout(height=700, width=1100, title="MoE Routing Dashboard", template="plotly_white")
    return fig


def build_attention_probe_figure(diagnostics: list[dict]) -> go.Figure:
    layers = np.arange(1, len(diagnostics) + 1)
    entropy = np.asarray([layer.get("attn_entropy", np.nan) for layer in diagnostics], dtype=float)
    q_norm = np.asarray([layer.get("q_norm", np.nan) for layer in diagnostics], dtype=float)
    k_norm = np.asarray([layer.get("k_norm", np.nan) for layer in diagnostics], dtype=float)
    residual = np.asarray([layer.get("residual_norm_post_ffn", np.nan) for layer in diagnostics], dtype=float)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Attention Entropy", "Q/K Norms", "Residual Norm", "Entropy Trend"))
    fig.add_trace(go.Scatter(x=layers, y=entropy, mode="lines+markers", name="attn_entropy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=layers, y=q_norm, mode="lines+markers", name="q_norm"), row=1, col=2)
    fig.add_trace(go.Scatter(x=layers, y=k_norm, mode="lines+markers", name="k_norm"), row=1, col=2)
    fig.add_trace(go.Scatter(x=layers, y=residual, mode="lines+markers", name="residual_norm"), row=2, col=1)
    if len(entropy) > 2 and not np.isnan(entropy).all():
        smoothed = gaussian_filter1d(np.nan_to_num(entropy, nan=np.nanmean(entropy)), sigma=1)
        fig.add_trace(go.Scatter(x=layers, y=smoothed, mode="lines", name="entropy_smooth"), row=2, col=2)
    fig.update_layout(height=700, width=1100, title="Attention / Residual Probe", template="plotly_white")
    return fig


def build_next_token_figure(token_labels: list[str], probs: list[float]) -> go.Figure:
    probs_arr = np.asarray(probs, dtype=float)
    cumulative = np.cumsum(probs_arr)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Top-k Probabilities", "Cumulative Mass", "Distribution Histogram"))
    fig.add_trace(go.Bar(x=token_labels, y=probs_arr, name="top_k"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(probs_arr) + 1), y=cumulative, mode="lines+markers", name="cumulative"), row=1, col=2)
    fig.add_trace(go.Histogram(x=probs_arr, nbinsx=20, name="hist"), row=1, col=3)
    fig.update_layout(height=420, width=1200, title="Next-Token Probability Dashboard", template="plotly_white")
    return fig


def build_dashboard_bundle(rows: list[dict]) -> dict[str, go.Figure]:
    bundle = {"training": build_training_overview(rows)}
    if any("router_entropy" in row for row in rows):
        bundle["moe"] = build_moe_dashboard(rows)
    return bundle
