"""
NemotronDense LM backbone (MLX) — the text+audio-token decoder used by Audex
(nvidia/Nemotron-Labs-Audex-2B, `model_type: nemotron_dense`).

Architecture (from the released modeling_nemotron_dense.py):
  - 28 decoder layers, hidden 2048
  - RMSNorm (F.rms_norm style: x * rsqrt(mean(x^2) + eps) * w)
  - MLP: down(relu(up(x))^2)  — squared-ReLU, NO gate
  - GQA: 16 query heads / 8 KV heads, head_dim 128
  - full RoPE (partial_rotary_factor=1.0), theta=1e8
  - vocab 205312 (text + 65536 speechcodec + 8192 audiocodec tokens)
  - untied lm_head

This is a plain autoregressive LM; the only Audex-specific thing is the
extended vocabulary. Audio *understanding* (Whisper encoder + projector) is a
separate module and is not needed for text generation or TTS.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LMConfig:
    vocab_size: int = 205312
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-5
    rope_theta: float = 100000000.0
    max_position_embeddings: int = 131072

    @classmethod
    def from_hf(cls, cfg: dict) -> "LMConfig":
        rope = cfg.get("rope_parameters") or {}
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            norm_eps=cfg.get("norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", 100000000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 131072),
        )


class MLP(nn.Module):
    """Squared-ReLU MLP, no gate: down(relu(up(x))**2)."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.relu(self.up_proj(x)) ** 2)


class Attention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NemotronDenseModel(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            if cache is not None and cache[0] is not None and cache[0].offset > 0:
                # prefill after some cached context: extend mask over cached keys
                offset = cache[0].offset
                mask = mx.concatenate(
                    [mx.zeros((h.shape[1], offset), dtype=mask.dtype), mask], axis=-1
                )
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class NemotronDenseForCausalLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.model = NemotronDenseModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.model(inputs, cache=cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(self.cfg.num_hidden_layers)]
