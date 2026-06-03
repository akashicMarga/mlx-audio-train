"""
OmniVox — full speech-to-speech model (MLX).

Same architecture as MiniMind-O, modules swapped:
  SenseVoice (Chinese)  →  Whisper-small (multilingual, 50fps, 768d)
  MiniMind Thinker      →  Sarvam-1 2B (Llama, native Indic, 2048d, 28 layers)

Talker and Mimi codec are unchanged.

Full pipeline:
  Hindi audio
    → Whisper-small (frozen, 768d @ 50fps)
    → MMAudioProjector(768 → 2048)
    → injected at <|audio_pad|>=3 positions (reuses <<reserved_token_0>>)
    → Sarvam-1 backbone (28 layers, 2048d)
        ↓ bridge hidden states at layer 23
    → Talker (4 layers, 2048d)
        ↓ 8 Mimi codebook logits per step
    → Mimi decoder → 24kHz audio

Why no catastrophic forgetting:
  Backbone is frozen in Phase 1a (only audio_proj trains).
  Whisper and Mimi are always frozen.
  Phase 1b only touches the top-N Sarvam layers at low LR.

Reuses from models.minimind_o:
  MMAudioProjector  — generic (in_dim, out_dim), no change needed
  TalkerModule      — driven by config fields
  load_audio_encoder — Whisper loading already supported
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from models.omnivox.config import OmniVoxConfig
from models.minimind_o.projectors import MMAudioProjector
from models.minimind_o.talker import TalkerModule


class OmniVox(nn.Module):
    """
    Whisper-small → AudioProjector(768→2048) → Sarvam-1 → Talker → Mimi

    Usage:
        model = OmniVox(config)
        model.load_backbone()          # loads Sarvam-1 via mlx_lm
        model.freeze()
        model.audio_proj.unfreeze()    # Phase 1a / A2A projector training
        # model.unfreeze_top_layers(2) # Phase 1b
        # model.talker.unfreeze()      # T2A training
    """

    def __init__(self, config: Optional[OmniVoxConfig] = None):
        super().__init__()
        self.config = config or OmniVoxConfig()

        # Audio projector: Whisper 768d → backbone hidden_size (Sarvam: 2048d)
        self.audio_proj = MMAudioProjector(
            self.config.audio_hidden_size,  # 768
            self.config.hidden_size,        # 2048 for Sarvam-1
        )

        # Talker: 4-layer transformer → 8 Mimi codebook logits
        # Config fields duck-type with OmniConfig as expected by TalkerModule.__init__
        self.talker = TalkerModule(self.config)

        # Backbone (Sarvam-1 / Llama) — loaded separately via load_backbone()
        self.backbone: Optional[Any] = None

    # Backward-compat alias: older code reads model.qwen
    @property
    def qwen(self):
        return self.backbone

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    def load_backbone(self, model_path: Optional[str] = None) -> "OmniVox":
        """
        Load the backbone LLM (Sarvam-1 by default) via mlx_lm.

        Phase 1a / T2A: backbone frozen, 4-bit fine (gradients flow through
        frozen quantized layers without updating them).
        Phase 1b / A2A: needs fp16 for top-layer updates.
        """
        from mlx_lm import load as mlx_load
        path = model_path or self.config.backbone_path
        backbone, _ = mlx_load(path)
        self.backbone = backbone
        return self

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def unfreeze_top_layers(self, n: int = 2) -> None:
        """Unfreeze top-n backbone layers + final norm for Phase 1b."""
        if self.backbone is None:
            raise RuntimeError("Call load_backbone() first")
        layers = self.backbone.model.layers
        total  = len(layers)
        for i in range(total - n, total):
            layers[i].unfreeze()
        self.backbone.model.norm.unfreeze()

    # ------------------------------------------------------------------
    # Audio injection
    # ------------------------------------------------------------------

    def inject_audio_features(
        self,
        token_ids:   mx.array,
        hidden:      mx.array,
        audio_feats: List[Optional[mx.array]],
    ) -> mx.array:
        """Replace <|audio_pad|> (151665) positions with projected Whisper features."""
        marker  = self.config.audio_pad_id
        ids_np  = np.array(token_ids.tolist())
        result  = []

        for b in range(hidden.shape[0]):
            hb  = hidden[b]
            seq = ids_np[b].tolist()
            af  = audio_feats[b] if (audio_feats and audio_feats[b] is not None) else None
            i   = 0

            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        n = min(af.shape[0], i - start)
                        parts: List[mx.array] = []
                        if start > 0:
                            parts.append(hb[:start])
                        parts.append(af[:n])
                        parts.append(hb[start + n:])
                        hb = mx.concatenate(parts, axis=0)
                        af = None
                else:
                    i += 1
            result.append(hb)

        return mx.stack(result, axis=0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids:       mx.array,
        audio_ids:       Optional[mx.array] = None,
        use_cache:       bool = False,
        mlx_audio_feats: Optional[List[Optional[mx.array]]] = None,
        spk_emb:         Optional[mx.array] = None,
        past_key_values: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids:       (B, T) int32 — text token ids; <|audio_pad|> marks audio positions
            audio_ids:       (B, 8, T) int32 — Mimi code history for Talker conditioning.
                             If None, filled with audio_pad_token (silent / unconditioned).
            mlx_audio_feats: per-batch list of (T_audio, 896) pre-projected Whisper features.
                             The caller (forward_step) runs audio_proj explicitly so gradients
                             flow; these are already 896d — inject_audio_features does NOT project.
            spk_emb:         (B, 192) speaker embedding for voice conditioning (optional).

        Returns:
            {
              "logits":       (B, T, 151936) — text token logits from Qwen lm_head
              "audio_logits": list of 8 × (B, T, 2112) — Mimi codebook logits from Talker
              "past_key_values": kv-cache list (when use_cache=True)
            }
        """
        if self.backbone is None:
            raise RuntimeError("Call load_backbone() before forward pass")

        B, T = input_ids.shape

        # Default audio_ids: all padding (unconditioned Talker)
        if audio_ids is None:
            audio_ids = mx.full(
                (B, 8, T), self.config.audio_pad_token, dtype=mx.int32
            )

        # ── Step 1: text embeddings ──────────────────────────────────────────────
        embeds = self.backbone.model.embed_tokens(input_ids)   # (B, T, hidden_size)

        # ── Step 2: inject projected audio features at <|audio_pad|> positions ───
        if mlx_audio_feats is not None:
            embeds = self.inject_audio_features(input_ids, embeds, mlx_audio_feats)

        # ── Step 3: run backbone layers manually to capture bridge states ────────
        # We cannot use backbone(None, input_embeddings=embeds) because it runs all
        # layers internally and we can't intercept bridge states mid-way.
        if self.config.backbone_arch == "llama":
            from mlx_lm.models.llama import create_attention_mask
        elif self.config.backbone_arch == "qwen2":
            from mlx_lm.models.qwen2 import create_attention_mask
        else:
            raise ValueError(f"Unsupported backbone_arch: {self.config.backbone_arch}")

        n_layers = len(self.backbone.model.layers)
        n_thinker_kvs = n_layers
        n_talker_kvs  = len(self.talker.layers)

        if past_key_values is None:
            past_key_values = [None] * (n_thinker_kvs + n_talker_kvs)

        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None else 0
        )

        h    = embeds
        mask = create_attention_mask(h, past_key_values[0])

        bridge_states = h   # fallback: use embeddings if bridge_layer not reached
        thinker_presents: List = []

        for i, (layer, past_kv) in enumerate(
            zip(self.backbone.model.layers, past_key_values[:n_thinker_kvs])
        ):
            result = layer(h, mask, past_kv)
            # mlx_lm layers return (hidden, cache) or just hidden depending on use_cache
            if isinstance(result, tuple):
                h, present = result
            else:
                h, present = result, None
            thinker_presents.append(present)

            if i == self.config.bridge_layer:
                bridge_states = h

        h_normed = self.backbone.model.norm(h)   # (B, T, hidden_size)

        # ── Step 4: text logits from backbone lm_head ────────────────────────────
        if self.backbone.args.tie_word_embeddings:
            text_logits = self.backbone.model.embed_tokens.as_linear(h_normed)
        else:
            text_logits = self.backbone.lm_head(h_normed)              # (B, T, vocab_size)

        # ── Step 5: Talker — bridge states → 8 Mimi codebook logits ─────────────
        audio_logits, talker_presents = self.talker(
            bridge_states=bridge_states,
            audio_ids=audio_ids,
            start_pos=start_pos,
            past_key_values=past_key_values[n_thinker_kvs:],
            use_cache=use_cache,
            spk_emb=spk_emb,
            audio_spk_token=self.config.audio_spk_token,
        )

        return {
            "logits":       text_logits,    # (B, T, vocab_size)
            "audio_logits": audio_logits,   # list of 8 × (B, T, 2112)
            "past_key_values": thinker_presents + talker_presents,
        }

    # ------------------------------------------------------------------
    # Convenience: load projector checkpoint
    # ------------------------------------------------------------------

    def load_projector_checkpoint(self, path: str) -> bool:
        """Load audio_proj weights from .npz (legacy) or .safetensors (preferred)."""
        import os
        if not os.path.exists(path):
            return False

        if path.endswith(".safetensors"):
            data = mx.load(path)  # dict[str, mx.array]
        else:
            np_data = np.load(path, allow_pickle=False)
            data = {k: mx.array(v) for k, v in np_data.items()}

        if "ln.weight" not in data:
            return False
        if data["ln.weight"].shape[0] != self.config.audio_hidden_size:
            return False
        self.audio_proj.load_weights(list(data.items()), strict=False)
        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def param_summary(self) -> str:
        from mlx.utils import tree_flatten
        trainable = sum(p.size for _, p in tree_flatten(self.trainable_parameters()))
        total     = sum(p.size for _, p in tree_flatten(self.parameters()))
        proj_n    = sum(p.size for _, p in tree_flatten(self.audio_proj.parameters()))
        talker_n  = sum(p.size for _, p in tree_flatten(self.talker.parameters()))
        return (
            f"OmniVox | backbone={self.config.backbone_path.split('/')[-1]}"
            f" | projector={proj_n/1e6:.2f}M"
            f" | talker={talker_n/1e6:.2f}M"
            f" | trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M"
        )
