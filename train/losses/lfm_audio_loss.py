"""
losses/lfm_audio_loss.py — LFM 2.5 Audio training loss.

LFM2.5-Audio is an autoregressive multimodal model.  Its backbone produces
hidden states for every position in the interleaved text+audio sequence.  The
Depthformer audio head then predicts the 8 Mimi codebook tokens per audio frame
sequentially (codebook k is conditioned on codebooks 0..k-1 at the same frame).

TTS loss (text → audio):
    input:   text_ids [B, T_text]  +  audio_codes[:, :-1, :] [B, T-1, 8]
    targets: audio_codes[:, 1:, :] [B, T-1, 8]   (next-frame prediction)
    loss:    mean CE over 8 codebooks, masked to valid frames

ASR loss (audio → text) — optional, not yet wired into the trainer but
    the function is provided so it can be called directly or extended.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_masked(
    logits:          mx.array,   # [B, T, vocab]
    targets:         mx.array,   # [B, T]
    mask:            mx.array,   # [B, T] bool — True = compute loss here
    label_smoothing: float = 0.0,
) -> mx.array:
    """Masked cross-entropy, returns scalar mean over valid positions."""
    B, T, V = logits.shape
    logits_flat  = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    mask_flat    = mask.reshape(-1)

    loss_all = nn.losses.cross_entropy(
        logits_flat, targets_flat,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    masked  = loss_all * mask_flat.astype(loss_all.dtype)
    n_valid = mx.maximum(mask_flat.sum(), 1)
    return masked.sum() / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# TTS loss — text → audio
# ─────────────────────────────────────────────────────────────────────────────

def lfm_audio_tts_loss(
    model,
    batch:                 Dict[str, mx.array],
    label_smoothing:       float = 0.0,
    text_loss_weight:      float = 0.0,  # >0 adds auxiliary CE on text predictions
) -> Tuple[mx.array, Dict]:
    """
    TTS training loss for LFM 2.5 Audio.

    Teacher-forced next-frame prediction:
        input  = audio_codes[:, :-1, :]   [B, T-1, 8]
        target = audio_codes[:, 1:, :]    [B, T-1, 8]

    The model is called with:
        text_tokens   = text_ids          (text conditioning)
        audio_features = None             (no input audio in TTS mode)
        audio_codes   = audio_codes_in    (teacher-forced audio input)

    Returns (total_loss, metrics_dict) where metrics include per-codebook losses.
    """
    text_ids    = batch["text_ids"]     # [B, T_text]
    audio_codes = batch["audio_codes"]  # [B, T_audio, C]  C may be 32 (all Mimi codebooks)
    audio_mask  = batch["audio_mask"]   # [B, T_audio]

    # Teacher forcing: input is all-but-last frame, target is all-but-first frame
    audio_in  = audio_codes[:, :-1, :]  # [B, T_audio-1, C]
    audio_tgt = audio_codes[:, 1:,  :]  # [B, T_audio-1, C]
    loss_mask = audio_mask[:, 1:]        # [B, T_audio-1]

    # Forward pass — audio_features=None for TTS (no input speech)
    text_logits, audio_logits = model(
        text_tokens    = text_ids,
        audio_features = None,
        audio_codes    = audio_in,
    )
    # audio_logits: list of num_codebooks tensors, each [B, T_audio_out, vocab]
    # T_audio_out should equal T_audio-1 (matches audio_in length)

    # ── Audio codebook losses ─────────────────────────────────────────────────
    num_codebooks = len(audio_logits)
    cb_losses = []
    for k in range(num_codebooks):
        logits_k = audio_logits[k]                       # [B, T_out, vocab]
        T_out    = logits_k.shape[1]
        target_k = audio_tgt[:, :T_out, k]               # [B, T_out]
        mask_k   = loss_mask[:, :T_out]                  # [B, T_out]
        cb_losses.append(
            cross_entropy_masked(logits_k, target_k, mask_k, label_smoothing)
        )

    # Mean over codebooks (all codebooks contribute equally)
    audio_loss = mx.sum(mx.stack(cb_losses)) / num_codebooks

    total_loss = audio_loss

    # ── Optional auxiliary text CE loss ───────────────────────────────────────
    text_loss = mx.array(0.0)
    if text_loss_weight > 0 and text_logits is not None:
        text_mask = batch.get("text_mask")
        if text_mask is not None:
            T_text  = text_ids.shape[1]
            T_logit = text_logits.shape[1]
            T_min   = min(T_text - 1, T_logit)
            if T_min > 0:
                text_logits_shift = text_logits[:, :T_min, :]
                text_targets      = text_ids[:, 1:T_min + 1]
                text_mask_shift   = text_mask[:, 1:T_min + 1]
                text_loss = cross_entropy_masked(
                    text_logits_shift, text_targets, text_mask_shift, label_smoothing
                )
                total_loss = audio_loss + text_loss_weight * text_loss

    metrics = {
        "loss":       float(total_loss),
        "audio_loss": float(audio_loss),
        "text_loss":  float(text_loss),
    }
    for k, cb_l in enumerate(cb_losses):
        metrics[f"cb{k}_loss"] = float(cb_l)

    return total_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ASR loss — audio → text  (auxiliary; for speech-understanding fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def lfm_audio_asr_loss(
    model,
    batch:           Dict[str, mx.array],
    label_smoothing: float = 0.0,
) -> Tuple[mx.array, Dict]:
    """
    ASR training loss for LFM 2.5 Audio.

    When batch contains "modalities", uses model._prefill with the proper
    modality layout so audio is positioned BEFORE the assistant tokens (matching
    inference-time ChatState layout). This lets the model attend to audio when
    predicting the function call.

    Without "modalities" (legacy), audio is appended after text — broken for ASR.

    Batch keys: text_ids, audio_features, text_mask, modalities (optional)
    """
    text_ids       = batch["text_ids"]       # [B, N_text]
    audio_features = batch["audio_features"] # [B, T_mel, 128]
    text_mask      = batch["text_mask"]      # [B, N_text]

    if "modalities" in batch:
        modalities = batch["modalities"]     # [B, N_total]  N_total = N_text + N_audio_frames

        # Forward with audio in the user-turn position (before assistant tokens)
        hidden_states, _ = model._prefill(
            text_tokens    = text_ids,
            audio_features = audio_features,
            audio_codes    = None,
            modalities     = modalities,
        )
        # Apply text projection to ALL positions → [B, N_total, V]
        text_logits_full = model.lfm.embed_tokens.as_linear(hidden_states)

        # Extract logits at TEXT positions only (modality == 1)
        # Assume batch_size=1 for index extraction; generalises to larger batches below
        mods_np   = np.array(modalities[0].tolist())
        text_pos  = np.where(mods_np == 1)[0]        # [N_text]

        # Gather: [B, N_text, V]  — mx.take preserves gradients
        text_pos_mx  = mx.array(text_pos.astype(np.int32))
        text_logits  = mx.take(text_logits_full[0], text_pos_mx, axis=0)[None]

    else:
        # Legacy: audio concatenated at end — model cannot attend to audio for ASR
        text_logits, _ = model(
            text_tokens    = text_ids,
            audio_features = audio_features,
            audio_codes    = None,
        )

    N_text  = text_ids.shape[1]
    T_logit = text_logits.shape[1]
    T_min   = min(N_text - 1, T_logit)

    if T_min <= 0:
        loss = mx.array(0.0)
        return loss, {"loss": 0.0}

    text_logits_shift = text_logits[:, :T_min, :]
    text_targets      = text_ids[:, 1:T_min + 1]
    text_mask_shift   = text_mask[:, 1:T_min + 1]

    loss = cross_entropy_masked(text_logits_shift, text_targets, text_mask_shift, label_smoothing)
    return loss, {"loss": float(loss)}
