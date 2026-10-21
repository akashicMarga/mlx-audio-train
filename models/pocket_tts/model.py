"""
Pocket TTS FlowLM in MLX — trainable port of kyutai-labs/pocket-tts.

Parameter names mirror the PyTorch state dict (without the "flow_lm." prefix),
so released `model.safetensors` files load directly and the checkpoints written
here load back into the official `pocket-tts` package.

Architecture (CALM, arXiv:2509.06926):

    [bos_before_voice | voice latents @ speaker_proj | text LUT | BOS, audio latents...]
        -> causal transformer (RoPE, optional GQA / attn_dim)
        -> out_norm -> (optional z_proj bottleneck) = z          (one per audio frame)
        -> sampler head  flow_net(z, noise) -> next latent      (LSD / drifting / pointwise)
        -> EOS head      out_eos(z)

Distillation only ever touches the backbone: the heads are copied from the
teacher and stay frozen, so the student just has to reproduce the teacher's z.

Supported options (all of the Oct-2026 small-model additions):
    z_dim, attn_dim, num_kv_heads, dim_feedforward, low-rank text table,
    sampler heads with 0/1/2 time conditions.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.pocket_tts.config import FlowLMArgs


def gelu_tanh(x: mx.array) -> mx.array:
    # Same tanh approximation as torch's F.gelu(approximate="tanh").
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


# ── Backbone ──────────────────────────────────────────────────────────────────

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dim: int | None,
                 num_kv_heads: int | None, max_period: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim or d_model
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.attn_dim % num_heads == 0 and num_heads % self.num_kv_heads == 0
        self.head_dim = self.attn_dim // num_heads
        self.max_period = max_period
        kv_dim = self.head_dim * self.num_kv_heads
        self.in_proj = nn.Linear(d_model, self.attn_dim + 2 * kv_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, d_model, bias=False)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        b, t, _ = x.shape
        hd = self.head_dim
        p = self.in_proj(x)
        # Rows of in_proj are [q | (k, v) interleaved per kv head], like the torch view().
        q = p[..., : self.attn_dim].reshape(b, t, self.num_heads, hd).transpose(0, 2, 1, 3)
        kv = p[..., self.attn_dim:].reshape(b, t, 2, self.num_kv_heads, hd)
        k = kv[:, :, 0].transpose(0, 2, 1, 3)
        v = kv[:, :, 1].transpose(0, 2, 1, 3)

        offset = 0 if cache is None else cache.offset
        q = mx.fast.rope(q, hd, traditional=True, base=self.max_period, scale=1.0, offset=offset)
        k = mx.fast.rope(k, hd, traditional=True, base=self.max_period, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)

        if t == 1:
            mask = None
        elif k.shape[2] == t:
            mask = "causal"
        else:
            pos_q = mx.arange(offset, offset + t)[:, None]
            mask = pos_q >= mx.arange(k.shape[2])[None, :]
        # GQA: sdpa shares kv head j across query heads [j*g, (j+1)*g), like repeat_interleave.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.attn_dim)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, a: FlowLMArgs):
        super().__init__()
        self.self_attn = Attention(a.d_model, a.num_heads, a.attn_dim, a.num_kv_heads, a.max_period)
        self.norm1 = nn.LayerNorm(a.d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(a.d_model, eps=1e-5)
        self.linear1 = nn.Linear(a.d_model, a.dim_feedforward, bias=False)
        self.linear2 = nn.Linear(a.dim_feedforward, a.d_model, bias=False)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        x = x + self.self_attn(self.norm1(x), cache)
        return x + self.linear2(gelu_tanh(self.linear1(self.norm2(x))))


class Transformer(nn.Module):
    def __init__(self, a: FlowLMArgs):
        super().__init__()
        self.layers = [TransformerLayer(a) for _ in range(a.num_layers)]

    def __call__(self, x: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[i])
        return x


class LUTConditioner(nn.Module):
    """Text lookup table; low-rank (rank-wide table + bias-free projection) when rank is set."""

    def __init__(self, n_bins: int, dim: int, rank: int | None):
        super().__init__()
        self.embed = nn.Embedding(n_bins + 1, rank or dim)  # last bin = padding
        if rank is not None:
            self.proj = nn.Linear(rank, dim, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.embed(tokens)
        return self.proj(x) if "proj" in self else x


# ── Sampler head (SimpleMLPAdaLN, from MAR / DiT) ─────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((dim,))

    def __call__(self, x):
        # torch's x.var() is unbiased (ddof=1).
        var = self.eps + mx.var(x, axis=-1, keepdims=True, ddof=1)
        return x * (self.alpha * mx.rsqrt(var))


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight + self.bias if "weight" in self else x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.mlp = [nn.Linear(freq_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden), RMSNorm(hidden)]
        half = freq_size // 2
        self.freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)

    def __call__(self, t):
        args = t * self.freqs
        x = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        for layer in self.mlp:
            x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = [nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)]
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(channels, 3 * channels)]

    def __call__(self, x, y):
        shift, scale, gate = mx.split(self.adaLN_modulation[1](nn.silu(y)), 3, axis=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp[2](nn.silu(self.mlp[0](h)))
        return x + gate * h


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(channels, eps=1e-6, affine=False)
        self.linear = nn.Linear(channels, out_channels)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(channels, 2 * channels)]

    def __call__(self, x, c):
        shift, scale = mx.split(self.adaLN_modulation[1](nn.silu(c)), 2, axis=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_ch: int, ch: int, out_ch: int, cond_ch: int, depth: int, num_time_conds: int):
        super().__init__()
        self.num_time_conds = num_time_conds
        self.time_embed = [TimestepEmbedder(ch) for _ in range(num_time_conds)]
        self.cond_embed = nn.Linear(cond_ch, ch)
        self.input_proj = nn.Linear(in_ch, ch)
        self.res_blocks = [ResBlock(ch) for _ in range(depth)]
        self.final_layer = FinalLayer(ch, out_ch)

    def __call__(self, c, *args):
        ts, x = args[:-1], args[-1]
        assert len(ts) == self.num_time_conds
        x = self.input_proj(x)
        y = self.cond_embed(c)
        if self.num_time_conds:
            y = y + sum(emb(t) for emb, t in zip(self.time_embed, ts)) / self.num_time_conds
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


def sample_head(flow_type: str, head, z: mx.array, noise: mx.array, n_steps: int = 1) -> mx.array:
    """One latent per row of z, from the head's decode rule for its training objective."""
    if flow_type == "drifting":
        return head(z, noise)
    if flow_type == "pointwise":
        return noise + head(z, noise)
    x = noise
    ones = mx.ones(noise.shape[:-1] + (1,), dtype=noise.dtype)
    for i in range(n_steps):
        if flow_type == "lsd":
            x = x + head(z, ones * (i / n_steps), ones * ((i + 1) / n_steps), x) / n_steps
        else:  # flow_matching (OT Euler)
            x = x + head(z, ones * (i / n_steps), x) / n_steps
    return x


# ── FlowLM ────────────────────────────────────────────────────────────────────

# Tensors that are statistics / fixed tables, never trained here.
BUFFER_KEYS = ("emb_mean", "emb_std")


class FlowLM(nn.Module):
    def __init__(self, a: FlowLMArgs):
        super().__init__()
        self.args = a
        self.conditioner = LUTConditioner(a.n_bins, a.d_model, a.lut_rank)
        self.flow_net = SimpleMLPAdaLN(a.ldim, a.flow_dim, a.ldim, a.z_width, a.flow_depth,
                                       a.num_time_conds)
        self.emb_std = mx.ones((a.ldim,))
        self.emb_mean = mx.zeros((a.ldim,))
        self.bos_emb = mx.random.normal((a.ldim,))
        if a.insert_bos_before_voice:
            self.bos_before_voice = mx.random.normal((1, 1, a.d_model))
        self.input_linear = nn.Linear(a.ldim, a.d_model, bias=False)
        self.transformer = Transformer(a)
        self.out_norm = nn.LayerNorm(a.d_model, eps=1e-5)
        if a.z_dim is not None:
            self.z_proj = nn.Linear(a.d_model, a.z_dim, bias=False)
        self.out_eos = nn.Linear(a.z_width, 1)
        self.speaker_proj_weight = mx.zeros((a.d_model, a.voice_dim))

    # The parts of the state that are not the backbone: copied from the teacher
    # and frozen during distillation (see distill.py).
    HEAD_PREFIXES = ("flow_net.", "out_eos.")

    def freeze_buffers(self):
        self.freeze(keys=list(BUFFER_KEYS), recurse=False)
        for emb in self.flow_net.time_embed:
            emb.freeze(keys=["freqs"], recurse=False)

    def backbone(self, x: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        out = self.out_norm(self.transformer(x, cache))
        return self.z_proj(out) if "z_proj" in self else out

    def build_inputs(
        self,
        normalized_latents: mx.array,           # [B, T, ldim]
        text_tokens: list[np.ndarray],          # per-row int token ids
        voice_latents: mx.array | None,         # [B, Tv, voice_dim] raw latents
        voice_lens: np.ndarray | None,          # [B] valid voice frames
        keep_voice: np.ndarray,                 # [B] bool
        keep_text: np.ndarray,                  # [B] bool
    ) -> tuple[mx.array, np.ndarray]:
        """Right-padded [bos_before_voice, voice, text, BOS, audio[:-1]] rows + prefix lengths.

        Padding sits after the audio, so under causal attention it never reaches a
        valid position. A dropped segment (CFG null branch) is removed entirely.
        """
        B, T, _ = normalized_latents.shape
        bos = mx.broadcast_to(self.bos_emb[None, None, :], (B, 1, self.args.ldim))
        audio_emb = self.input_linear(mx.concatenate([bos, normalized_latents[:, :-1]], axis=1))
        dtype = audio_emb.dtype

        lens = np.array([len(t) for t in text_tokens])
        tokens = np.full((B, max(int(lens.max()), 1)), self.args.n_bins, dtype=np.int32)
        for b, t in enumerate(text_tokens):
            tokens[b, : len(t)] = t
        text_emb = self.conditioner(mx.array(tokens)).astype(dtype)
        t_len = lens * keep_text

        if voice_latents is not None and "speaker_proj_weight" in self:
            voice_emb = (voice_latents.astype(dtype) @ self.speaker_proj_weight.T.astype(dtype))
            vl = voice_lens if voice_lens is not None else np.full(B, voice_latents.shape[1])
            v_len = np.minimum(vl, voice_latents.shape[1]) * keep_voice
        else:
            voice_emb, v_len = None, np.zeros(B, dtype=np.int64)

        has_bos_prefix = "bos_before_voice" in self
        prefix = int(has_bos_prefix) + v_len + t_len
        width = int((prefix + T).max())
        rows = []
        for b in range(B):
            parts = []
            if has_bos_prefix:
                parts.append(self.bos_before_voice[0].astype(dtype))
            if v_len[b]:
                parts.append(voice_emb[b, : int(v_len[b])])
            if t_len[b]:
                parts.append(text_emb[b, : int(t_len[b])])
            parts.append(audio_emb[b])
            pad = width - int(prefix[b]) - T
            if pad:
                parts.append(mx.zeros((pad, audio_emb.shape[-1]), dtype=dtype))
            rows.append(mx.concatenate(parts, axis=0))
        return mx.stack(rows), prefix

    def backbone_z(self, normalized_latents, text_tokens, voice_latents, voice_lens,
                   keep_voice, keep_text) -> mx.array:
        """Teacher-forced z [B, T, z_width]: z[:, t] conditions the heads for frame t."""
        x, prefix = self.build_inputs(normalized_latents, text_tokens, voice_latents, voice_lens,
                                      keep_voice, keep_text)
        out = self.backbone(x)
        T = normalized_latents.shape[1]
        idx = mx.array(prefix[:, None] + np.arange(T)[None, :], dtype=mx.int32)
        idx = mx.broadcast_to(idx[:, :, None], (idx.shape[0], T, out.shape[-1]))
        return mx.take_along_axis(out, idx, axis=1)

    def generate_latents(
        self,
        text_tokens: np.ndarray,
        voice_latents: mx.array | None = None,   # [Tv, voice_dim]
        max_frames: int = 375,
        temp: float = 0.3,
        n_steps: int = 1,
        cfg_coef: float = 1.0,
        eos_threshold: float = -1.0,
        min_frames: int = 6,
    ) -> mx.array:
        """Autoregressive sampling with a KV cache (batch 1). Returns raw latents [T, ldim]."""
        a = self.args
        prefix = []
        if "bos_before_voice" in self:
            prefix.append(self.bos_before_voice[0])
        if voice_latents is not None and "speaker_proj_weight" in self:
            prefix.append(voice_latents @ self.speaker_proj_weight.T)
        prefix.append(self.conditioner(mx.array(text_tokens, dtype=mx.int32)))
        prefixes = [mx.concatenate(prefix, axis=0)[None]]
        if cfg_coef != 1.0:
            prefixes.append(self.bos_before_voice if "bos_before_voice" in self
                            else mx.zeros((1, 0, a.d_model)))
        caches = [[KVCache() for _ in self.transformer.layers] for _ in prefixes]

        x_lat = self.bos_emb[None, None, :]
        frames = []
        for step in range(max_frames):
            zs = []
            for p, cache in zip(prefixes, caches):
                inp = self.input_linear(x_lat)
                if step == 0:
                    inp = mx.concatenate([p, inp], axis=1)
                zs.append(self.backbone(inp, cache)[:, -1])
            z = zs[0] if len(zs) == 1 else zs[1] + cfg_coef * (zs[0] - zs[1])
            noise = (temp**0.5) * mx.random.normal((1, a.ldim))
            latent = sample_head(a.flow_type, self.flow_net, z, noise, n_steps)
            eos = self.out_eos(z)[0, 0].item() > eos_threshold
            if eos and step >= min_frames:
                break
            frames.append(latent)
            x_lat = latent[:, None, :]
            mx.eval(x_lat)
        if not frames:
            return mx.zeros((0, a.ldim))
        return mx.concatenate(frames, axis=0) * self.emb_std + self.emb_mean
