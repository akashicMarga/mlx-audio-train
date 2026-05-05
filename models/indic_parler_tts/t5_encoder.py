"""
Pure MLX T5 encoder (Flan-T5-Large variant).

Differences from standard transformer:
- RMSNorm (no bias) before each sub-layer
- Relative position bias (bidirectional, 32 buckets) — computed once in block 0, reused
- Gated-GELU FFN: output = wo(gelu(wi_0(x)) * wi_1(x))
- No scaling in attention (T5 omits the 1/sqrt(d_kv) factor)
- No absolute position embeddings
"""

import math
import mlx.core as mx
import mlx.nn as nn

from .config import T5Config


class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms) * self.weight


class T5Attention(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_bias: bool = False):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_kv = cfg.d_kv
        inner = cfg.num_heads * cfg.d_kv

        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)

        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(
                cfg.relative_attention_num_buckets, cfg.num_heads
            )
        else:
            self.relative_attention_bias = None

        self.num_buckets = cfg.relative_attention_num_buckets
        self.max_distance = cfg.relative_attention_max_distance

    def _bucket(self, rel_pos: mx.array) -> mx.array:
        # Bidirectional: map signed relative_position to [0, num_buckets)
        n_buckets = self.num_buckets
        half = n_buckets // 2

        # Sign bit selects the upper/lower half of the table
        ret = (rel_pos > 0).astype(mx.int32) * half
        n = mx.abs(rel_pos)

        max_exact = half // 2
        is_small = n < max_exact

        # Log-spaced buckets for large distances
        log_ratio = math.log(self.max_distance / max_exact)
        val_large = (
            mx.log(mx.maximum(n.astype(mx.float32), 1.0) / max_exact) / log_ratio
            * (half - max_exact)
        ).astype(mx.int32) + max_exact
        val_large = mx.minimum(val_large, half - 1)

        ret = ret + mx.where(is_small, n.astype(mx.int32), val_large)
        return ret

    def compute_bias(self, qlen: int, klen: int) -> mx.array:
        q_pos = mx.arange(qlen, dtype=mx.int32)[:, None]  # [Q, 1]
        k_pos = mx.arange(klen, dtype=mx.int32)[None, :]  # [1, K]
        rel_pos = k_pos - q_pos                            # [Q, K]
        buckets = self._bucket(rel_pos)                    # [Q, K]
        # relative_attention_bias.weight: [num_buckets, num_heads]
        bias = self.relative_attention_bias.weight[buckets]  # [Q, K, H]
        return bias.transpose(2, 0, 1)[None]               # [1, H, Q, K]

    def __call__(
        self,
        x: mx.array,
        position_bias: mx.array = None,
    ):
        B, T, _ = x.shape
        H, D = self.num_heads, self.d_kv

        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # [B, H, T, D]
        k = self.k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # T5 uses no explicit scale; relative bias handles stability
        scores = q @ k.transpose(0, 1, 3, 2)  # [B, H, T, T]

        if position_bias is None and self.relative_attention_bias is not None:
            position_bias = self.compute_bias(T, T)
        if position_bias is not None:
            scores = scores + position_bias

        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.o(out), position_bias


class T5FFN(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(_gelu_new(self.wi_0(x)) * self.wi_1(x))


def _gelu_new(x: mx.array) -> mx.array:
    """T5 gated-gelu uses the Hugging Face gelu_new activation."""
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class T5Block(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_bias: bool = False):
        super().__init__()
        self.attn = T5Attention(cfg, has_relative_bias=has_relative_bias)
        self.attn_ln = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)
        self.ffn = T5FFN(cfg)
        self.ffn_ln = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array, position_bias: mx.array = None):
        h, position_bias = self.attn(self.attn_ln(x), position_bias=position_bias)
        x = x + h
        x = x + self.ffn(self.ffn_ln(x))
        return x, position_bias


class T5Encoder(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.shared = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [
            T5Block(cfg, has_relative_bias=(i == 0))
            for i in range(cfg.num_layers)
        ]
        self.final_ln = T5LayerNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.shared(input_ids)  # [B, T, d_model]
        # position_bias computed once in block 0, reused for all blocks
        position_bias = None
        for block in self.blocks:
            x, position_bias = block(x, position_bias=position_bias)
        return self.final_ln(x)
