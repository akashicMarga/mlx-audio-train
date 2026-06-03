"""
OmniVox configuration — Whisper-small + Sarvam-1 (2B) + Talker (Mimi) S2S model.

Architecture (same structure as MiniMind-O, modules swapped):
  MiniMind-O:  SenseVoice → proj → MiniMind-Thinker → bridge → Talker → Mimi
  OmniVox:     Whisper    → proj → Sarvam-1 (Llama) → bridge → Talker → Mimi

Backbone history:
  v0 (deprecated): Qwen2.5-0.5B-Instruct — too small, Chinese/English prior,
                   produced Chinese hallucination on Hindi audio input
  v1 (current):    Sarvam-1 2B (Llama arch) — native Indic pretraining by Sarvam AI

Talker and Mimi are unchanged from MiniMind-O — same 8-codebook Mimi codec,
same Talker architecture (just adapted to 2048d hidden to match Sarvam-1).

Training phases:
  T2A  — text → Mimi codes: train Talker (+ audio_proj) to speak Hindi from text
  1a   — audio → text: train MMAudioProjector (Whisper 768 → Sarvam 2048)
  1b   — audio → text: top-N Sarvam layers + projector
  A2A  — full speech-to-speech: combines everything
"""

from __future__ import annotations

from dataclasses import dataclass
import yaml


# ── Sarvam-1 backbone dims (Llama architecture, 2.5B params) ───────────────────
SARVAM_HIDDEN_SIZE = 2048
SARVAM_VOCAB_SIZE  = 68096
SARVAM_NUM_LAYERS  = 28

# Sarvam-1 has 4096 reserved tokens (<<reserved_token_*>>) at low IDs.
# We repurpose id=3 (<<reserved_token_0>>) as <|audio_pad|> — no embedding resize needed.
AUDIO_PAD_ID    = 3
AUDIO_PAD_TOKEN_STR = "<<reserved_token_0>>"

# ── Mimi codec constants (same as MiniMind-O) ──────────────────────────────────
AUDIO_VOCAB_SIZE  = 2112   # 2048 Mimi codes + 64 specials
AUDIO_PAD_TOKEN   = 2049
AUDIO_STOP_TOKEN  = 2050
AUDIO_SPK_TOKEN   = 2051   # position where speaker embedding is injected


@dataclass
class OmniVoxConfig:
    # ── Sarvam-1 backbone (Llama 2B, native Indic) ──────────────────────────────
    backbone_path: str = "mlx-community/sarvam-1-4bit"
    # Phase 1a / T2A (frozen backbone): 4-bit is fine.
    # Phase 1b / A2A (top layers unfrozen): use fp16:
    #   "sarvamai/sarvam-1" (needs MLX conversion) or convert from 4-bit on the fly.
    backbone_arch: str = "llama"   # selects create_attention_mask import

    hidden_size: int = SARVAM_HIDDEN_SIZE   # 2048
    vocab_size:  int = SARVAM_VOCAB_SIZE    # 68096
    num_hidden_layers: int = SARVAM_NUM_LAYERS  # 28

    # ── Bridge layer ────────────────────────────────────────────────────────────
    # Sarvam layer index (0-based, out of 28) whose hidden states feed Talker.
    # MiniMind-O uses ~37%. We use ~83% for richer semantics: 23/28.
    bridge_layer: int = 23

    # ── Audio encoder (Whisper-small) ───────────────────────────────────────────
    audio_encoder_type: str = "whisper-small"
    audio_encoder_path: str = "mlx-community/whisper-small-mlx"
    audio_hidden_size:  int = 768   # Whisper-small output dim (768d @ 50fps)

    # ── Audio special tokens ────────────────────────────────────────────────────
    audio_pad_id:        int = AUDIO_PAD_ID        # 3 — reuses <<reserved_token_0>>
    audio_pad_token_str: str = AUDIO_PAD_TOKEN_STR # "<<reserved_token_0>>"
    audio_vocab_size:    int = AUDIO_VOCAB_SIZE    # 2112 Mimi tokens
    audio_pad_token:     int = AUDIO_PAD_TOKEN
    audio_stop_token:    int = AUDIO_STOP_TOKEN
    audio_spk_token:     int = AUDIO_SPK_TOKEN

    # ── Talker (acoustic head) ──────────────────────────────────────────────────
    # Hidden dim matches backbone to make bridge projection identity-shaped.
    talker_hidden_size:       int   = SARVAM_HIDDEN_SIZE   # 2048
    num_talker_hidden_layers: int   = 4
    # Talker attention — keep GQA ratio similar to backbone (16/8 = 2:1)
    num_attention_heads:      int   = 16
    num_key_value_heads:      int   = 8
    spk_emb_size:             int   = 192

    # ── Misc (required by TalkerModule / MiniMindBlock internals) ───────────────
    use_moe:                  bool  = False
    dropout:                  float = 0.0
    rms_norm_eps:             float = 1e-6   # Sarvam-1 / Llama uses 1e-6
    rope_theta:               float = 10000.0
    max_position_embeddings:  int   = 2048
    num_experts:              int   = 4
    num_experts_per_tok:      int   = 2
    moe_intermediate_size:    int   = 0

    # ── Chat format markers (Sarvam-1 is Llama-style, no native ChatML) ─────────
    # We use the raw BOS/EOS strings as turn delimiters. Format:
    #   <s>system\n...</s>\n<s>user\n...</s>\n<s>assistant\n...</s>\n
    bos_token: str = "<s>"
    eos_token: str = "</s>"

    # Alias for MiniMind-O compatibility (some utils read config.audio_ids)
    @property
    def audio_ids(self):
        return [self.audio_pad_id]


@dataclass
class OmniVoxTrainingConfig:
    # ── Mode ─────────────────────────────────────────────────────────────────────
    # "t2a"        → text → Mimi codes: train Qwen+Talker (like MiniMind-O T2A)
    # "audio_proj" → A2A Phase 1a: projector only
    # "last2"      → A2A Phase 1b: projector + top-N Qwen layers
    mode: str = "audio_proj"

    unfreeze_top_layers: int = 2   # for "last2" mode

    # Warm-start projector from Phase 1a checkpoint
    load_projector: str = ""

    # ── Data ─────────────────────────────────────────────────────────────────────
    data_path:     str = ""
    val_data_path: str = ""

    # ── Optimisation ─────────────────────────────────────────────────────────────
    epochs:            int   = 5
    batch_size:        int   = 1
    grad_accumulation: int   = 4
    learning_rate:     float = 5e-4
    grad_clip:         float = 1.0
    max_seq_len:       int   = 512

    # ── Checkpointing / logging ───────────────────────────────────────────────────
    save_dir:      str = "./out/omnivox"
    log_interval:  int = 25
    save_interval: int = 200


def load_omnivox_config(yaml_path: str) -> tuple[OmniVoxConfig, OmniVoxTrainingConfig]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    model_raw    = raw.get("model", {})
    training_raw = raw.get("training", {})

    # Auto-resolve audio_hidden_size from encoder type if not explicit
    enc_type = model_raw.get("audio_encoder_type", "whisper-small")
    default_dim = {"whisper-small": 768, "whisper-medium": 1024,
                   "whisper-large-v2": 1280, "whisper-large-v3": 1280}.get(enc_type, 768)
    model_raw.setdefault("audio_hidden_size", default_dim)

    cfg_fields  = OmniVoxConfig.__dataclass_fields__
    train_fields = OmniVoxTrainingConfig.__dataclass_fields__

    model_cfg    = OmniVoxConfig(**{k: v for k, v in model_raw.items() if k in cfg_fields})
    training_cfg = OmniVoxTrainingConfig(**{k: v for k, v in training_raw.items() if k in train_fields})
    return model_cfg, training_cfg
