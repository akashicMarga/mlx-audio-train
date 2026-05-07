"""
Pure MLX Parler-TTS audio decoder.

Architecture:
- 24 decoder layers, hidden=1024, heads=16, ffn=4096
- Causal self-attention + cross-attention to T5 encoder output
- Standard LayerNorm WITH bias (not RMSNorm)
- Learned absolute position embeddings (no RoPE)
- 9 separate codebook embedding tables
- Fused LM head: (9 * lm_vocab_size, hidden) → split to [9, lm_vocab_size]
- Delay pattern: codebook k uses BOS for the first k generation steps

KV cache format per layer: [self_k, self_v, cross_k, cross_v]
Cross-attn KV is computed once from enc_hidden and reused.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import DecoderConfig


def gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / 1.4142135623730951))


class ParlerSelfAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: list = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        if cache is not None:
            if len(cache) == 2:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
                cache[0] = k
                cache[1] = v
            else:
                cache.append(k)
                cache.append(v)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # [B, H, T, S]
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.out(out)


class ParlerCrossAttention(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        enc_hidden: mx.array,
        cache: list = None,
        encoder_mask: mx.array = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # Compute enc KV once, then cache
        if cache is not None and len(cache) == 2:
            k, v = cache[0], cache[1]
        else:
            Ts = enc_hidden.shape[1]
            k = self.k(enc_hidden).reshape(B, Ts, H, D).transpose(0, 2, 1, 3)
            v = self.v(enc_hidden).reshape(B, Ts, H, D).transpose(0, 2, 1, 3)
            if cache is not None:
                cache.append(k)
                cache.append(v)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if encoder_mask is not None:
            scores = scores + encoder_mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.out(out)


class ParlerDecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.self_attn = ParlerSelfAttention(cfg)
        self.self_attn_ln = nn.LayerNorm(cfg.hidden_size)
        self.cross_attn = ParlerCrossAttention(cfg)
        self.cross_attn_ln = nn.LayerNorm(cfg.hidden_size)
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_size, bias=False)
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size)

    def __call__(
        self,
        x: mx.array,
        enc_hidden: mx.array,
        mask: mx.array = None,
        self_cache: list = None,
        cross_cache: list = None,
        encoder_mask: mx.array = None,
    ) -> mx.array:
        # Pre-norm self-attention
        h = self.self_attn(self.self_attn_ln(x), mask=mask, cache=self_cache)
        x = x + h
        # Pre-norm cross-attention
        h = self.cross_attn(self.cross_attn_ln(x), enc_hidden, cache=cross_cache, encoder_mask=encoder_mask)
        x = x + h
        # Pre-norm FFN
        h = self.fc2(gelu(self.fc1(self.ffn_ln(x))))
        return x + h


class ParlerDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = [
            nn.Embedding(cfg.embed_vocab_size, cfg.hidden_size)
            for _ in range(cfg.num_codebooks)
        ]
        self.embed_positions = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.layers = [ParlerDecoderLayer(cfg) for _ in range(cfg.num_layers)]
        self.final_ln = nn.LayerNorm(cfg.hidden_size)
        # Fused head: (9 * lm_vocab_size, hidden)
        self.lm_heads = nn.Linear(cfg.hidden_size, cfg.num_codebooks * cfg.lm_vocab_size, bias=False)

    def embed_audio(self, tokens: mx.array, offset: int = 0) -> mx.array:
        """
        tokens: [B, num_codebooks] int32 — one token per codebook per time step
        Returns sum of codebook embeddings + position embedding: [B, 1, hidden]
        """
        emb = self.embed_audio_no_pos(tokens)
        emb = emb + self.decoder_position(offset, 1)
        return emb

    def embed_audio_no_pos(self, tokens: mx.array) -> mx.array:
        """Embed one delayed audio-token step without positional embeddings."""
        emb = None
        for k in range(self.cfg.num_codebooks):
            e = self.embed_tokens[k](tokens[:, k : k + 1])  # [B, 1, hidden]
            emb = e if emb is None else emb + e
        return emb

    def decoder_position(self, offset: int, length: int) -> mx.array:
        """Sinusoidal/learned decoder position slice in broadcastable shape."""
        return self.embed_positions.weight[offset : offset + length][None, :, :]

    def forward_layers(
        self,
        x: mx.array,
        enc_hidden: mx.array,
        mask: mx.array = None,
        self_caches: list = None,
        cross_caches: list = None,
        encoder_mask: mx.array = None,
    ) -> mx.array:
        for i, layer in enumerate(self.layers):
            sc = self_caches[i] if self_caches is not None else None
            cc = cross_caches[i] if cross_caches is not None else None
            x = layer(x, enc_hidden, mask=mask, self_cache=sc, cross_cache=cc, encoder_mask=encoder_mask)
        return self.final_ln(x)

    def logits(self, hidden: mx.array) -> mx.array:
        """hidden: [B, T, H] → [B, T, num_codebooks, lm_vocab_size]"""
        out = self.lm_heads(hidden)  # [B, T, 9*1088]
        B, T, _ = out.shape
        return out.reshape(B, T, self.cfg.num_codebooks, self.cfg.lm_vocab_size)
