from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 2048
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    qk_norm: bool = False
    sliding_window: int | None = None
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_every_n_layers: int = 0
    moe_aux_loss_coef: float = 0.01
    init_std: float = 0.02


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def active_parameter_count(cfg: ModelConfig) -> int:
    dense_params = cfg.d_model * cfg.vocab_size * (2 if not cfg.tie_embeddings else 1)
    mlp_hidden = int(cfg.d_model * cfg.mlp_ratio)
    attn_params = cfg.n_layers * (
        cfg.d_model * cfg.d_model * 2
        + cfg.d_model * (cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)) * 2
    )
    mlp_params = cfg.n_layers * (3 * cfg.d_model * mlp_hidden)
    if cfg.moe_num_experts > 0:
        moe_layers = cfg.n_layers // max(cfg.moe_every_n_layers, 1)
        mlp_params += moe_layers * (cfg.moe_num_experts - 1) * (3 * cfg.d_model * mlp_hidden) // max(cfg.moe_num_experts, 1)
    return dense_params + attn_params + mlp_params


def estimate_kv_cache_bytes(cfg: ModelConfig, batch_size: int, seq_len: int, dtype_bytes: int = 2) -> int:
    head_dim = cfg.d_model // cfg.n_heads
    return batch_size * seq_len * cfg.n_layers * cfg.n_kv_heads * head_dim * 2 * dtype_bytes


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cached_cos = emb.cos().to(dtype=dtype)
        self._cached_sin = emb.sin().to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_cos is None or seq_len > self._cached_seq_len or self._cached_cos.device != device or self._cached_cos.dtype != dtype:
            self._build_cache(seq_len, device, dtype)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        hidden = int(math.ceil(hidden / 256) * 256)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.dropout(self.down_proj(gated))
        diag = {
            "mlp_hidden_norm": float(gated.detach().norm(dim=-1).mean().item()),
            "mlp_out_norm": float(out.detach().norm(dim=-1).mean().item()),
        }
        return out, diag


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_experts = cfg.moe_num_experts
        self.top_k = cfg.moe_top_k
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(self.num_experts)])
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        self.aux_loss_coef = cfg.moe_aux_loss_coef

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        bsz, seq_len, dim = x.shape
        flat = x.reshape(bsz * seq_len, dim)
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        expert_usage = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)

        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            expert_input = flat[token_ids]
            expert_out, _ = expert(expert_input.unsqueeze(1))
            weights = topk_vals[token_ids, slot_ids].unsqueeze(-1)
            output[token_ids] += expert_out.squeeze(1) * weights
            expert_usage[expert_id] = mask.sum()

        router_mean = router_probs.mean(dim=0)
        usage_mean = expert_usage / max(float(expert_usage.sum().item()), 1.0)
        aux_loss = float((self.num_experts * torch.sum(router_mean * usage_mean)).item())
        entropy = float((-(router_probs.clamp_min(1e-9) * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()).item())

        diag = {
            "router_aux_loss": aux_loss,
            "router_entropy": entropy,
            "expert_usage_max": float(usage_mean.max().item()) if expert_usage.sum() > 0 else 0.0,
            "expert_usage_min": float(usage_mean.min().item()) if expert_usage.sum() > 0 else 0.0,
            "expert_usage": usage_mean.detach().cpu().tolist() if expert_usage.sum() > 0 else [0.0] * self.num_experts,
        }
        return output.reshape(bsz, seq_len, dim), diag


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_base)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_heads == self.n_kv_heads:
            return x
        repeat = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
        capture_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor] | None]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cache_len = 0 if kv_cache is None else kv_cache["k"].shape[-2]
        cos, sin = self.rope(cache_len + seq_len, x.device, q.dtype)
        q = apply_rope(q, cos[cache_len:], sin[cache_len:])
        k = apply_rope(k, cos[cache_len:], sin[cache_len:])

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2)
            v = torch.cat([kv_cache["v"], v], dim=-2)
        next_cache = {"k": k, "v": v} if use_cache else None

        q_full = q
        k_full = self._expand_kv(k)
        v_full = self._expand_kv(v)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_full, k_full.transpose(-1, -2)) * scale

        causal_offset = k_full.shape[-2] - q_full.shape[-2]
        causal_mask = torch.ones(q_full.shape[-2], k_full.shape[-2], device=x.device, dtype=torch.bool).tril(diagonal=causal_offset)
        if self.cfg.sliding_window is not None:
            window_mask = torch.zeros_like(causal_mask)
            for row in range(q_full.shape[-2]):
                left = max(0, k_full.shape[-2] - q_full.shape[-2] + row - self.cfg.sliding_window + 1)
                right = k_full.shape[-2] - q_full.shape[-2] + row + 1
                window_mask[row, left:right] = True
            causal_mask &= window_mask
        scores = scores.masked_fill(~causal_mask.view(1, 1, *causal_mask.shape), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v_full)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.d_model)
        out = self.resid_dropout(self.out_proj(out))

        diag = {
            "attn_entropy": float((-(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()).item()),
            "q_norm": float(q.detach().norm(dim=-1).mean().item()),
            "k_norm": float(k.detach().norm(dim=-1).mean().item()),
            "v_norm": float(v.detach().norm(dim=-1).mean().item()),
        }
        if capture_diagnostics:
            diag["head_entropy_std"] = float((-(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(dim=-1).mean(dim=(0, 2)).std()).item())
        return out, diag, next_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        use_moe = cfg.moe_num_experts > 0 and cfg.moe_every_n_layers > 0 and ((layer_idx + 1) % cfg.moe_every_n_layers == 0)
        self.ffn = MoEFeedForward(cfg) if use_moe else SwiGLU(cfg)
        self.uses_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
        capture_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor] | None]:
        resid = x
        attn_out, attn_diag, next_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, use_cache=use_cache, capture_diagnostics=capture_diagnostics)
        x = resid + attn_out
        resid_after_attn = float(x.detach().norm(dim=-1).mean().item())
        ffn_out, ffn_diag = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        diag = {
            "residual_norm_post_attn": resid_after_attn,
            "residual_norm_post_ffn": float(x.detach().norm(dim=-1).mean().item()),
            **attn_diag,
            **ffn_diag,
        }
        return x, diag, next_cache


class ResearchTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=self.cfg.init_std * scale)
            if isinstance(block.ffn, SwiGLU):
                nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=self.cfg.init_std * scale)
            else:
                for expert in block.ffn.experts:
                    nn.init.normal_(expert.down_proj.weight, mean=0.0, std=self.cfg.init_std * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: list[dict[str, torch.Tensor] | None] | None = None,
        capture_diagnostics: bool = False,
    ) -> dict[str, object]:
        x = self.dropout(self.embed_tokens(input_ids))
        next_cache: list[dict[str, torch.Tensor] | None] = []
        diagnostics: list[dict[str, float]] = []

        for idx, block in enumerate(self.blocks):
            layer_cache = None if past_key_values is None else past_key_values[idx]
            x, block_diag, cache = block(x, kv_cache=layer_cache, use_cache=use_cache, capture_diagnostics=capture_diagnostics)
            diagnostics.append(block_diag)
            next_cache.append(cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        aux_loss = 0.0
        if diagnostics:
            aux_loss = sum(diag.get("router_aux_loss", 0.0) for diag in diagnostics) * self.cfg.moe_aux_loss_coef
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": loss.new_tensor(aux_loss) if loss is not None else torch.tensor(aux_loss, device=logits.device),
            "past_key_values": next_cache if use_cache else None,
            "diagnostics": diagnostics,
            "hidden_states": x,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        generated = input_ids
        past = None
        for _ in range(max_new_tokens):
            step_input = generated[:, -1:] if past is not None else generated[:, -self.cfg.max_seq_len :]
            out = self(step_input, use_cache=True, past_key_values=past)
            past = out["past_key_values"]
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated
