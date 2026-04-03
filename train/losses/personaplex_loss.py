"""
personaplex_loss.py — Loss function for PersonaPlex full-duplex speech model.

Interface: fn(model, batch) -> (loss_scalar, metrics_dict)
matches the Trainer's expected loss_fn signature.

PersonaPlex token layout (B, 17, T):
  Row  0:     text tokens
  Rows 1-8:   assistant audio codebook tokens (depformer slices 0-7)
  Rows 9-16:  user audio codebook tokens (depformer slices 8-15)

Loss = text_CE + audio_loss_weight * assistant_audio_CE
"""
from __future__ import annotations

from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn


def personaplex_loss(
    model,
    batch: Dict[str, mx.array],
    audio_loss_weight: float = 1.0,
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute PersonaPlex training loss.

    Args:
        model:             PersonaPlex Lm instance (LoRA applied, freeze_non_trainable called)
        batch:             {"input_tokens": (B,17,T), "target_tokens": (B,17,T)}
        audio_loss_weight: weight for audio stream loss

    Returns:
        (total_loss, {"text_loss": float, "audio_loss": float})

    KV cache strategy:
        model.transformer_cache (list[RotatingKVCache]) is reset before each forward.
        Without reset, KV states accumulate across batches until max_size=4096 triggers
        expensive trim-concatenation operations — the main training stall on this arch.
        model.depformer_cache is reset automatically inside DepFormer.__call__.
    """
    input_tokens  = batch["input_tokens"]   # (B, 17, T)
    target_tokens = batch["target_tokens"]  # (B, 17, T)

    B, num_streams, T = input_tokens.shape
    text_out_vocab = model.cfg.text_out_vocab_size
    audio_pad      = model.cfg.audio_padding_token
    text_pad       = 3  # PAD token (matches HindiSpeechDataset / PersonaPlexDataset)

    # Reset RotatingKVCache before each forward (critical for training stability)
    for c in model.transformer_cache:
        c.reset()

    # Forward through main transformer
    transformer_out, text_logits = model.forward_codes(input_tokens)

    # ── Text loss (row 0) ──────────────────────────────────────────────────
    text_targets = target_tokens[:, 0, :]                         # (B, T)
    text_mask    = (text_targets != text_pad).astype(mx.float32)  # (B, T)
    text_ce      = nn.losses.cross_entropy(
        text_logits.reshape(-1, text_out_vocab),
        text_targets.reshape(-1),
        reduction="none",
    )
    text_loss = (text_ce * text_mask.reshape(-1)).sum() / (text_mask.sum() + 1e-8)

    # ── Audio loss via DepFormer (assistant rows 1-8 only) ─────────────────
    # DepFormer.__call__ resets depformer_cache internally before iterating slices.
    num_slices        = model.cfg.depformer.num_slices
    depformer_input   = input_tokens[:, :num_slices, :]  # (B, num_slices, T)
    audio_logits_list = model.depformer(
        transformer_out, depformer_input, model.depformer_cache
    )

    audio_loss  = mx.array(0.0)
    audio_count = mx.array(0.0)
    assistant_slices = min(model.cfg.audio_tokens_per_stream, len(audio_logits_list))
    for slice_idx, audio_logits in enumerate(audio_logits_list[:assistant_slices]):
        # Target for depformer slice i is row (i+1) of target_tokens
        audio_targets = target_tokens[:, slice_idx + 1, :]   # (B, T)
        audio_mask    = (audio_targets != audio_pad).astype(mx.float32)
        # Clip targets to valid codebook range (guards against padding token overflow)
        targets_flat  = mx.clip(audio_targets.reshape(-1), 0, audio_logits.shape[-1] - 1)
        slice_ce      = nn.losses.cross_entropy(
            audio_logits.reshape(-1, audio_logits.shape[-1]),
            targets_flat,
            reduction="none",
        )
        audio_loss  = audio_loss  + (slice_ce * audio_mask.reshape(-1)).sum()
        audio_count = audio_count + audio_mask.sum()

    audio_loss = audio_loss / (audio_count + 1e-8)
    total_loss = text_loss + audio_loss_weight * audio_loss

    # Return MLX arrays — trainer evaluates and converts to float after mx.eval()
    metrics = {
        "text_loss":  text_loss,
        "audio_loss": audio_loss,
    }
    return total_loss, metrics
