"""
Config dataclasses for MiniMind-O.

Architecture is fully config-driven: switching audio encoders, training modes,
freeze strategies, and hyperparameters all come from OmniConfig / OmniTrainingConfig.
No code changes needed to swap components — change the YAML, everything adjusts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Audio encoder registry — hidden size per encoder type
# Add new encoders here; the rest of the code picks it up automatically.
# ---------------------------------------------------------------------------

AUDIO_ENCODER_HIDDEN_SIZES: Dict[str, int] = {
    # ── MLX-native (no PyTorch needed) ──────────────────────────────────────
    "sensevoice":        512,   # SenseVoice-Small — Chinese/Japanese/Korean primary
    "whisper-tiny":      384,   # truly multilingual, lightest
    "whisper-base":      512,   # matches SenseVoice dim — easy swap
    "whisper-small":     768,   # matches Thinker hidden (768d) — projector becomes alignment-only
    "whisper-medium":   1024,   # best Indic quality, larger projector
    "whisper-large-v3": 1280,   # highest quality, heaviest projector

    # ── PyTorch-based (requires real torch + per-language adapter for MMS) ──
    "wav2vec2-base":     768,
    "wav2vec2-large":   1024,
    "mms-300m":         1024,   # needs language adapter activated per language
    "mms-1b":           1280,   # needs language adapter activated per language
}

AUDIO_ENCODER_DEFAULT_PATHS: Dict[str, str] = {
    "sensevoice":        "mlx-community/SenseVoiceSmall",
    "whisper-tiny":      "mlx-community/whisper-tiny-mlx",
    "whisper-base":      "mlx-community/whisper-base-mlx",
    "whisper-small":     "mlx-community/whisper-small-mlx",
    "whisper-medium":    "mlx-community/whisper-medium-mlx",
    "whisper-large-v3":  "mlx-community/whisper-large-v3-mlx",
    "wav2vec2-base":     "facebook/wav2vec2-base-960h",
    "wav2vec2-large":    "facebook/wav2vec2-large-960h",
    "mms-300m":          "facebook/mms-300m",
    "mms-1b":            "facebook/mms-1b",
}

WHISPER_ENCODER_TYPES = {
    "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"
}


# ---------------------------------------------------------------------------
# Base transformer config (Thinker + Talker architecture)
# ---------------------------------------------------------------------------

@dataclass
class MiniMindConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    dropout: float = 0.0
    vocab_size: int = 6400
    bos_token_id: int = 1
    eos_token_id: int = 2
    flash_attn: bool = True
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    # MoE
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: Optional[int] = None
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size


# ---------------------------------------------------------------------------
# Full model config — all architecture decisions live here
# ---------------------------------------------------------------------------

@dataclass
class OmniConfig(MiniMindConfig):
    # ── Audio encoder (config-driven — swap by changing audio_encoder_type) ──
    # Supported types: sensevoice | wav2vec2-base | wav2vec2-large | mms-300m | mms-1b
    # To add a new encoder: register it in AUDIO_ENCODER_HIDDEN_SIZES above.
    audio_encoder_type: str = "sensevoice"
    audio_encoder_path: str = ""   # empty = auto-resolved from type via AUDIO_ENCODER_DEFAULT_PATHS
    audio_hidden_size: int = 0     # 0 = auto-resolved from encoder type; set explicitly to override

    # ── Talker ───────────────────────────────────────────────────────────────
    num_talker_hidden_layers: int = 4
    talker_hidden_size: int = 768

    # ── Audio tokens / vocab ─────────────────────────────────────────────────
    audio_ids: List[int] = field(default_factory=lambda: [16])
    audio_special_token: str = "<|audio_pad|>"
    audio_vocab_size: int = 2112   # 2048 Mimi codes + 64 specials
    audio_pad_token: int = 2049
    audio_stop_token: int = 2050
    audio_spk_token: int = 2051
    spk_emb_size: int = 192        # CAM++ speaker embedding dim

    # ── Chain-of-thought end marker: </think>\n\n ────────────────────────────
    think_end_ids: List[int] = field(default_factory=lambda: [26, 234, 234])

    # ── Vision (SigLIP2-p32-256) ──────────────────────────────────────────────
    image_ids: List[int] = field(default_factory=lambda: [12])
    image_special_token: str = "<|image_pad|>"
    image_hidden_size: int = 768
    image_token_len: int = 64

    # ── Bridge layer (which Thinker hidden state feeds the Talker) ───────────
    bridge_layer: int = -1         # -1 = auto (num_hidden_layers // 2 - 1)

    def __post_init__(self):
        super().__post_init__()

        # Auto-resolve bridge layer
        if self.bridge_layer == -1:
            self.bridge_layer = self.num_hidden_layers // 2 - 1

        # Auto-resolve audio encoder path
        if not self.audio_encoder_path:
            self.audio_encoder_path = AUDIO_ENCODER_DEFAULT_PATHS.get(
                self.audio_encoder_type, ""
            )

        # Auto-resolve audio hidden size from encoder type
        if self.audio_hidden_size == 0:
            self.audio_hidden_size = AUDIO_ENCODER_HIDDEN_SIZES.get(
                self.audio_encoder_type, 512
            )

    @property
    def encoder_hidden_size(self) -> int:
        """Resolved hidden size for the currently configured audio encoder."""
        return AUDIO_ENCODER_HIDDEN_SIZES.get(self.audio_encoder_type, self.audio_hidden_size)


# ---------------------------------------------------------------------------
# Training config — all training decisions live here (loaded from YAML)
# ---------------------------------------------------------------------------

@dataclass
class OmniTrainingConfig:
    """
    Controls what gets trained and how. Loaded from the [training] section of YAML.

    mode:
      "audio_proj"   — train only the audio projector (~1.2M params). Phase 1a.
      "vision_proj"  — train only the vision projector.
      "last_N"       — train projector + last N Thinker layers (N = 1,2,4,...).
      "full"         — train all non-frozen components.

    freeze_backbone:
      "all"          — freeze entire Thinker.
      "none"         — train entire Thinker.
      "last1"        — freeze all Thinker layers except the last 1.
      "last2"        — freeze all Thinker layers except the last 2.

    freeze_talker:
      True           — keep Talker frozen (recommended for audio_proj mode).
      False          — train Talker alongside other components.
    """
    # What to train
    mode: str = "audio_proj"           # audio_proj | vision_proj | last_N | full
    freeze_backbone: str = "all"       # all | none | last1 | last2
    freeze_talker: bool = True

    # Hyperparameters
    epochs: int = 5
    batch_size: int = 4
    grad_accumulation: int = 4
    learning_rate: float = 5e-4
    grad_clip: float = 1.0
    max_seq_len: int = 512
    scheduled_sampling: float = 0.05

    # Data
    data_path: str = ""
    tokenizer_path: str = ""
    num_workers: int = 2

    # Warm-start: path to a Phase 1a projector NPZ to load before training
    load_projector: str = ""

    # Checkpointing
    save_dir: str = "./out/minimind_o"
    save_weight: str = "sft_omni"
    log_interval: int = 50
    save_interval: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> "OmniTrainingConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# YAML loader — single entry point for config files
# ---------------------------------------------------------------------------

def load_config_yaml(path: str):
    """
    Load a YAML config and return (OmniConfig, OmniTrainingConfig).

    YAML structure:
        model:
            audio_encoder_type: mms-300m
            audio_encoder_path: facebook/mms-300m   # optional, auto-resolved
            hidden_size: 768
            ...
        training:
            mode: audio_proj
            freeze_backbone: all
            epochs: 5
            data_path: ./data/indic_train.parquet
            ...
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    model_dict    = raw.get("model", {})
    training_dict = raw.get("training", {})

    # Build OmniConfig: filter to known fields
    omni_fields = {f for f in OmniConfig.__dataclass_fields__}
    base_fields  = {f for f in MiniMindConfig.__dataclass_fields__}
    all_fields   = omni_fields | base_fields
    omni_kwargs  = {k: v for k, v in model_dict.items() if k in all_fields}
    omni_cfg     = OmniConfig(**omni_kwargs)

    train_cfg = OmniTrainingConfig.from_dict(training_dict)

    return omni_cfg, train_cfg
