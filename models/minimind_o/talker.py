"""
MiniMind Talker — acoustic generation head, ported from PyTorch to MLX.

TalkerHead: shared base + 8 per-codebook adapter heads (MTP).
TalkerEmbedding: shared base + 8 per-codebook adapter embeddings.
TalkerModule: 4 transformer layers that take Thinker bridge states + audio codes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.minimind_o.config import MiniMindConfig, OmniConfig
from models.minimind_o.thinker import MiniMindBlock, precompute_freqs_cis_np


# ---------------------------------------------------------------------------
# TalkerHead: base lm_head + 8 lightweight adapter heads
# ---------------------------------------------------------------------------

class _TalkerHeadAdapter(nn.Module):
    """Low-rank adapter: Linear(in, rank) + GELU + Linear(rank, out)."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, rank, bias=False)
        self.fc2 = nn.Linear(rank, out_features, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class TalkerHead(nn.Module):
    """Output head that produces 8 codebook logit tensors from a single hidden state."""

    def __init__(self, in_features: int, out_features: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = [
            _TalkerHeadAdapter(in_features, out_features, rank)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> List[mx.array]:
        """Returns list of num_layers tensors, each (batch, seq, out_features)."""
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


# ---------------------------------------------------------------------------
# TalkerEmbedding: base embedding + 8 per-codebook adapter embeddings
# ---------------------------------------------------------------------------

class _TalkerEmbeddingAdapter(nn.Module):
    """Per-codebook adapter: Embedding(vocab, rank) + GELU + Linear(rank, dim)."""
    def __init__(self, vocab: int, rank: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, rank)
        self.proj = nn.Linear(rank, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.gelu(self.emb(x)))


class TalkerEmbedding(nn.Module):
    """
    Embeds 8 parallel audio code channels into a single hidden vector.

    Input: (batch, 8, seq) — 8 codebook token IDs per position
    Output: (batch, seq, hidden) — single combined embedding
    """

    def __init__(self, vocab: int, embedding_dim: int, num_layers: int = 8, rank: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(vocab, embedding_dim)
        self.adapters = [
            _TalkerEmbeddingAdapter(vocab, rank, embedding_dim)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, 8, seq)
        base_out = self.base(x)  # (batch, 8, seq, hidden)
        total = sum(
            base_out[:, i, :] + self.adapters[i](x[:, i, :])
            for i in range(self.num_layers)
        )
        return total / self.num_layers  # (batch, seq, hidden)


# ---------------------------------------------------------------------------
# Projection MLP: Linear + GELU + Linear + RMSNorm
# ---------------------------------------------------------------------------

class _ProjMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.RMSNorm(out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.fc2(nn.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# TalkerModule: 4 transformer layers operating on fused thinker+audio states
# ---------------------------------------------------------------------------

class TalkerModule(nn.Module):
    """
    Acoustic generation module.

    Receives:
        - bridge_states: Thinker hidden states from bridge_layer (batch, seq, thinker_hidden)
        - audio_ids: previous audio codes (batch, 8, seq)
        - optional spk_emb: speaker embedding (batch, spk_emb_size)

    Produces: list of 8 logit tensors, each (batch, seq, audio_vocab_size)
    """

    def __init__(self, config: OmniConfig):
        super().__init__()
        talker_cfg = MiniMindConfig(
            hidden_size=config.talker_hidden_size,
            num_hidden_layers=config.num_talker_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            use_moe=config.use_moe,
            dropout=config.dropout,
            vocab_size=config.audio_vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
        )
        self.talker_config = talker_cfg

        self.layers = [MiniMindBlock(i, talker_cfg) for i in range(config.num_talker_hidden_layers)]
        self.norm = nn.RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)

        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
        self.codec_proj = _ProjMLP(config.talker_hidden_size, config.talker_hidden_size)
        self.embed_proj = _ProjMLP(config.hidden_size, config.talker_hidden_size)

        self.text_scale = mx.array(3.0)
        self.audio_scale = mx.array(1.0)

        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)

        cos_np, sin_np = precompute_freqs_cis_np(
            talker_cfg.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self._freqs_cos: np.ndarray = cos_np
        self._freqs_sin: np.ndarray = sin_np

    def __call__(
        self,
        bridge_states: mx.array,
        audio_ids: mx.array,
        start_pos: int = 0,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        attention_mask: Optional[mx.array] = None,
        spk_emb: Optional[mx.array] = None,
        audio_spk_token: int = 2051,
    ) -> Tuple[List[mx.array], List]:
        seq_len = bridge_states.shape[1]
        past_key_values = past_key_values or [None] * len(self.layers)

        talker_emb = self.embed_tokens(audio_ids)  # (batch, seq, hidden)

        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == audio_spk_token)[..., None]  # (b, seq, 1)
            spk_projected = self.spk_proj(spk_emb)[:, None, :]              # (b, 1, hidden)
            talker_emb = mx.where(spk_mask, spk_projected, talker_emb)

        hidden_states = (
            self.embed_proj(bridge_states) * self.text_scale
            + self.codec_proj(talker_emb) * self.audio_scale
        )

        cos = mx.array(self._freqs_cos[start_pos: start_pos + seq_len])
        sin = mx.array(self._freqs_sin[start_pos: start_pos + seq_len])

        presents: List = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, cos, sin, past_kv, use_cache, attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        audio_logits = self.lm_head(hidden_states)  # list of 8 tensors (b, seq, audio_vocab)
        return audio_logits, presents
