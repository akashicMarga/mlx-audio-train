"""
NV-Whisper audio encoder + sound projector (MLX) for Audex audio understanding.

NV-Whisper is a stock HuggingFace Qwen2AudioEncoder (Whisper-large-v3-style):
mel (128, 3000) -> conv frontend (stride-2) -> +learned positions -> 32 pre-norm
transformer layers -> avg-pool(2) -> LayerNorm -> (750, 1280) per 30s clip.

The projector (Megatron sound_projection) maps encoder hidden -> LLM embedding
space: RMSNorm -> fc1(1280->4096) -> relu^2 -> fc2(4096->2048).

Weight keys match the released checkpoint exactly:
  audio_encoder.{conv1,conv2,embed_positions,layer_norm,layers.N.*}
  audio_projector.{norm,fc1,fc2}
Conv1d weights are transposed torch[out,in,k]->mlx[out,k,in] at conversion time.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 128
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    max_source_positions: int = 1500
    # projector
    audio_encoder_hidden_size: int = 1280
    audio_projector_intermediate_size: int = 4096
    audio_projector_norm_eps: float = 1e-5
    hidden_size: int = 2048          # LLM embedding dim

    @classmethod
    def from_hf(cls, cfg: dict) -> "AudioEncoderConfig":
        ac = cfg["audio_config"]
        return cls(
            num_mel_bins=ac.get("num_mel_bins", 128),
            d_model=ac.get("d_model", 1280),
            encoder_layers=ac.get("encoder_layers", 32),
            encoder_attention_heads=ac.get("encoder_attention_heads", 20),
            encoder_ffn_dim=ac.get("encoder_ffn_dim", 5120),
            max_source_positions=ac.get("max_source_positions", 1500),
            audio_encoder_hidden_size=cfg.get("audio_encoder_hidden_size", 1280),
            audio_projector_intermediate_size=cfg.get("audio_projector_intermediate_size", 4096),
            audio_projector_norm_eps=cfg.get("audio_projector_norm_eps", 1e-5),
            hidden_size=cfg.get("hidden_size", 2048),
        )


class EncoderAttention(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.n_heads = cfg.encoder_attention_heads
        self.head_dim = cfg.d_model // cfg.encoder_attention_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)  # no mask (bidirectional)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.self_attn = EncoderAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class AudioEncoder(nn.Module):
    """Qwen2AudioEncoder: mel -> (B, 750, d_model) per 30s clip."""

    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = [EncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, input_features: mx.array) -> mx.array:
        # input_features: (B, n_mels, 3000) -> (B, 3000, n_mels) for MLX Conv1d
        x = input_features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))                  # (B, 1500, d_model)
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        # avg-pool over pairs (kernel=2, stride=2): 1500 -> 750  (BEFORE layer_norm)
        B, T, D = x.shape
        x = x[:, : (T // 2) * 2, :].reshape(B, T // 2, 2, D).mean(axis=2)
        return self.layer_norm(x)


class SoundProjector(nn.Module):
    """RMSNorm -> fc1 -> relu^2 -> fc2, mapping encoder dim to LLM embed dim."""

    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.audio_encoder_hidden_size, eps=cfg.audio_projector_norm_eps)
        self.fc1 = nn.Linear(cfg.audio_encoder_hidden_size, cfg.audio_projector_intermediate_size, bias=False)
        self.fc2 = nn.Linear(cfg.audio_projector_intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.relu(self.fc1(self.norm(x))) ** 2)


class AudioTower(nn.Module):
    """encoder + projector; audio_encoder.* / audio_projector.* keys."""

    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = AudioEncoder(cfg)
        self.audio_projector = SoundProjector(cfg)

    def __call__(self, input_features: mx.array) -> mx.array:
        return self.audio_projector(self.audio_encoder(input_features))
