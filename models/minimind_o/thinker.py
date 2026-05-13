"""
MiniMind Thinker — transformer backbone, ported from PyTorch to MLX.

Components: RMSNorm, GQA Attention with QK-norm + RoPE + KV cache,
SwiGLU FeedForward, sparse MoE, MiniMindBlock, MiniMindModel.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.minimind_o.config import MiniMindConfig


# ---------------------------------------------------------------------------
# RoPE helpers (stored as numpy to avoid MLX parameter tracking)
# ---------------------------------------------------------------------------

def precompute_freqs_cis_np(dim: int, end: int, rope_base: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute full-rotation RoPE tables as numpy arrays (not tracked by MLX)."""
    freqs = 1.0 / (rope_base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)              # (end, dim/2)
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)  # (end, dim)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """
    q, k: (batch, seq, heads, head_dim)
    cos, sin: (seq, head_dim)
    """
    def rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    # (seq, head_dim) → (seq, 1, head_dim) for broadcasting
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    q_rot = (q * cos + rotate_half(q) * sin).astype(q.dtype)
    k_rot = (k * cos + rotate_half(k) * sin).astype(k.dtype)
    return q_rot, k_rot


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """GQA: expand kv heads to match query heads.

    x: (batch, seq, n_kv_heads, head_dim)
    returns: (batch, seq, n_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv, hd = x.shape
    x = mx.expand_dims(x, axis=3)                                   # (..., n_kv, 1, hd)
    x = mx.broadcast_to(x, (bs, slen, n_kv, n_rep, hd))
    return x.reshape(bs, slen, n_kv * n_rep, hd)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.resid_dropout = nn.Dropout(p=config.dropout)
        self.flash = config.flash_attn

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        bsz, seq_len, _ = x.shape

        xq = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # QK-norm (per head)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV cache
        if past_key_value is not None:
            xk = mx.concatenate([past_key_value[0], xk], axis=1)
            xv = mx.concatenate([past_key_value[1], xv], axis=1)
        present = (xk, xv) if use_cache else None
        full_seq = xk.shape[1]

        # GQA: expand kv heads
        xk_rep = repeat_kv(xk, self.n_rep)  # (b, full_seq, n_heads, hd)
        xv_rep = repeat_kv(xv, self.n_rep)

        # Transpose to (batch, heads, seq, head_dim) for attention
        xq_t = xq.transpose(0, 2, 1, 3)       # (b, h, sq, hd)
        xk_t = xk_rep.transpose(0, 2, 1, 3)   # (b, h, full, hd)
        xv_t = xv_rep.transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)

        if self.flash and seq_len > 1 and past_key_value is None and attention_mask is None:
            # Full-sequence prefill with flash attention (no KV cache)
            causal_mask = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.float32), k=1) * -1e9
            output = mx.fast.scaled_dot_product_attention(
                xq_t, xk_t, xv_t, scale=scale, mask=causal_mask
            )
        else:
            scores = (xq_t @ xk_t.transpose(0, 1, 3, 2)) * scale  # (b, h, sq, full)

            # Causal mask on the new tokens against all keys
            if seq_len > 1:
                past_len = full_seq - seq_len
                causal_block = mx.triu(
                    mx.ones((seq_len, seq_len), dtype=mx.float32), k=1
                ) * -1e9
                if past_len > 0:
                    causal_mask = mx.concatenate(
                        [mx.zeros((seq_len, past_len)), causal_block], axis=-1
                    )
                else:
                    causal_mask = causal_block
                scores = scores + causal_mask[None, None, :, :]

            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask[:, None, None, :]) * -1e9

            output = (
                self.attn_dropout(mx.softmax(scores.astype(mx.float32), axis=-1)).astype(xq.dtype)
                @ xv_t
            )

        # (b, h, sq, hd) → (b, sq, h*hd)
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE FeedForward
# ---------------------------------------------------------------------------

class MoEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = [
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ]
        self.aux_loss = mx.array(0.0)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*T, D)

        scores = mx.softmax(self.gate(x_flat), axis=-1)  # (N, num_experts)
        k = self.config.num_experts_per_tok

        if k == 1:
            topk_idx = mx.argmax(scores, axis=-1, keepdims=True)  # (N, 1)
            topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)  # (N, 1)
        else:
            sorted_idx = mx.argsort(-scores, axis=-1)
            topk_idx = sorted_idx[:, :k]
            topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)

        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)

        y = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            if k == 1:
                mask = (topk_idx[:, 0] == i).astype(x_flat.dtype)[:, None]
                w = topk_weight[:, 0:1] * mask
            else:
                expert_mask = (topk_idx == i)  # (N, k)
                w = (topk_weight * expert_mask.astype(topk_weight.dtype)).sum(axis=-1, keepdims=True)
                mask = (w > 0).astype(x_flat.dtype)
            y = y + expert(x_flat) * w

        # Aux loss (load balancing)
        if self.training:
            load = mx.zeros(self.config.num_experts)
            for i in range(self.config.num_experts):
                load = load.at[i].add(
                    (mx.argmax(scores, axis=-1) == i).astype(mx.float32).mean()
                )
            self.aux_loss = (
                (load / self.config.num_experts * scores.mean(axis=0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = mx.array(0.0)

        return y.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp: nn.Module = (
            MoEFeedForward(config) if config.use_moe else FeedForward(config)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple]]:
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


# ---------------------------------------------------------------------------
# Full transformer model (Thinker)
# ---------------------------------------------------------------------------

class MiniMindModel(nn.Module):
    """8-layer causal transformer (the Thinker / semantic reasoning module)."""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layers = [MiniMindBlock(i, config) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE tables stored as numpy to avoid MLX tracking them as parameters
        cos_np, sin_np = precompute_freqs_cis_np(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self._freqs_cos: np.ndarray = cos_np
        self._freqs_sin: np.ndarray = sin_np

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, List, mx.array]:
        _, seq_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        cos = mx.array(self._freqs_cos[start_pos: start_pos + seq_len])
        sin = mx.array(self._freqs_sin[start_pos: start_pos + seq_len])

        presents: List = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, cos, sin, past_kv, use_cache, attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux = mx.array(0.0)
        if self.config.use_moe:
            for layer in self.layers:
                if isinstance(layer.mlp, MoEFeedForward):
                    aux = aux + layer.mlp.aux_loss

        return hidden_states, presents, aux
