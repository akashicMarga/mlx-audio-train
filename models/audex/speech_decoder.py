"""
Audex causal speech decoder (MLX) — FSQ speech tokens -> 16 kHz waveform.

Ported from the released modeling_audex_causal_speech_decoder.py. We implement
the full-sequence `forward()` path (no streaming cache): for a complete
utterance this is mathematically equivalent to the chunked streaming session
the reference uses (causal attention == incremental KV-cache decode; the
look-ahead conv sees the same real future frames, zero-padded only at the true
end), and it is far simpler.

Pipeline:
  speech token id (single int per frame, 0..65535)
    -> FSQ decompose into 8 base-4 levels -> codes in {-1,-1/3,1/3,1}
    -> project_out: Linear(8 -> 2048) [+bias]                     (vq_emb)
    -> fc_post_a:   Linear(2048 -> 2048)
    -> look-ahead: depthwise causal conv(k=5) -> SiLU -> 1x1 conv  (residual)
    -> 12x Vocos transformer block (RMSNorm, fused qkv, interleaved RoPE dim64)
    -> final RMSNorm
    -> head: tanh(Linear(2048 -> 320)) -> reshape to waveform (320 samples/frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SpeechDecoderConfig:
    hidden_dim: int = 2048
    depth: int = 12
    heads: int = 32
    pos_meb_dim: int = 64
    hop_length: int = 320
    vq_dim: int = 2048
    lookahead_steps: int = 4
    sample_rate: int = 16000
    token_embed_dim: int = 8
    codebook_levels: list = field(default_factory=lambda: [4, 4, 4, 4, 4, 4, 4, 4])

    @classmethod
    def from_hf(cls, cfg: dict) -> "SpeechDecoderConfig":
        return cls(
            hidden_dim=cfg.get("hidden_dim", 2048),
            depth=cfg.get("depth", 12),
            heads=cfg.get("heads", 32),
            pos_meb_dim=cfg.get("pos_meb_dim", 64),
            hop_length=cfg.get("hop_length", 320),
            vq_dim=cfg.get("vq_dim", 2048),
            lookahead_steps=cfg.get("lookahead_steps", 4),
            sample_rate=cfg.get("sample_rate", 16000),
            token_embed_dim=cfg.get("token_embed_dim", 8),
            codebook_levels=cfg.get("codebook_levels", [4] * 8),
        )


class SpeechTokenEmbedder(nn.Module):
    """FSQ index -> continuous codes -> Linear projection."""

    def __init__(self, cfg: SpeechDecoderConfig):
        super().__init__()
        # Stored as plain Python lists (not mx.array attributes) so they are NOT
        # registered as module parameters — they are FSQ constants, absent from
        # the checkpoint, and would otherwise break strict weight loading.
        self._levels = list(cfg.codebook_levels)
        basis = [1]
        for lv in cfg.codebook_levels[:-1]:
            basis.append(basis[-1] * lv)
        self._basis = basis
        self.project_out = nn.Linear(cfg.token_embed_dim, cfg.vq_dim, bias=True)

    def __call__(self, indices: mx.array) -> mx.array:
        # indices: [B, T] int -> [B, T, 8] level indices
        basis = mx.array(self._basis, dtype=mx.int32)
        levels_i = mx.array(self._levels, dtype=mx.int32)
        idx = indices[..., None].astype(mx.int32)
        level_indices = (idx // basis) % levels_i  # [B, T, 8]
        levels = levels_i.astype(mx.float32)
        codes = level_indices.astype(mx.float32)
        codes = codes * (2.0 / (levels - 1.0)) - 1.0
        return self.project_out(codes)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(rope_dim, traditional=True, base=10000)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        qkv = self.c_attn(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B, H, L, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q)
        k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal"
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_dim: int):
        super().__init__()
        self.att_norm = nn.RMSNorm(dim, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.att = Attention(dim, n_heads, rope_dim)
        self.mlp = MLP(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.att(self.att_norm(x))
        return x + self.mlp(self.ffn_norm(x))


class Backbone(nn.Module):
    def __init__(self, cfg: SpeechDecoderConfig):
        super().__init__()
        self.transformers = [
            TransformerBlock(cfg.hidden_dim, cfg.heads, cfg.pos_meb_dim)
            for _ in range(cfg.depth)
        ]
        self.final_layer_norm = nn.RMSNorm(cfg.hidden_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.transformers:
            x = block(x)
        return self.final_layer_norm(x)


class Vocos(nn.Module):
    """CausalCodecDecoderVocos.forward path."""

    def __init__(self, cfg: SpeechDecoderConfig):
        super().__init__()
        self.lookahead_steps = cfg.lookahead_steps
        self.hop_length = cfg.hop_length
        self.fc_post_a = nn.Linear(cfg.vq_dim, cfg.hidden_dim, bias=False)
        # wav_proj exists in the checkpoint (training-time conditioning); unused here.
        self.wav_proj = nn.Linear(cfg.hop_length, cfg.hidden_dim, bias=False)
        if cfg.lookahead_steps > 0:
            self.lookahead_conv = nn.Conv1d(
                cfg.hidden_dim, cfg.hidden_dim,
                kernel_size=cfg.lookahead_steps + 1, groups=cfg.hidden_dim, bias=False,
            )
            self.lookahead_proj = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1, bias=False)
        self.backbone = Backbone(cfg)
        self.head_proj = nn.Linear(cfg.hidden_dim, cfg.hop_length, bias=False)

    def _apply_lookahead(self, x: mx.array) -> mx.array:
        if self.lookahead_steps <= 0:
            return x
        # x: [B, L, C]; pad L on the right with lookahead_steps zeros, depthwise conv
        h = mx.pad(x, [(0, 0), (0, self.lookahead_steps), (0, 0)])
        h = self.lookahead_conv(h)          # [B, L, C]
        h = nn.silu(h)
        h = self.lookahead_proj(h)          # [B, L, C]
        return x + h

    def __call__(self, vq_emb: mx.array) -> mx.array:
        x = self.fc_post_a(vq_emb)
        x = self._apply_lookahead(x)
        x = self.backbone(x)
        x = mx.tanh(self.head_proj(x))      # [B, L, hop_length]
        B = x.shape[0]
        return x.reshape(B, 1, -1)          # [B, 1, L*hop_length]


class AudexSpeechDecoder(nn.Module):
    def __init__(self, cfg: SpeechDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.audex_speech_token_embedder = SpeechTokenEmbedder(cfg)
        self.module = Vocos(cfg)

    def __call__(self, indices: mx.array) -> mx.array:
        vq_emb = self.audex_speech_token_embedder(indices)
        return self.module(vq_emb)

    def decode(self, codes) -> mx.array:
        """codes: list[int] or 1-D array of speech token ids -> waveform [samples]."""
        idx = mx.array(codes, dtype=mx.int32)[None, :]
        wav = self(idx)
        return wav[0, 0]
