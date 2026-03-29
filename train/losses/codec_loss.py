"""
losses/codec_loss.py — Qwen3-TTS dual loss.

Mirrors official sft_12hz.py:
    loss = main_talker_loss + 0.3 * sub_talker_loss

Where:
    main_talker_loss  = cross_entropy(codec_head_logits, codec_ids)
    sub_talker_loss   = cross_entropy(code_predictor_logits, codec_ids)

Both computed only on valid (non-padded) positions.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple


def cross_entropy_masked(
    logits: mx.array,      # [B, T, vocab]
    targets: mx.array,     # [B, T]
    mask: mx.array,        # [B, T] bool — True = compute loss here
    label_smoothing: float = 0.0,
) -> mx.array:
    """Masked cross-entropy loss, returns scalar mean over valid positions."""
    B, T, V = logits.shape

    logits_flat  = logits.reshape(-1, V)   # [B*T, V]
    targets_flat = targets.reshape(-1)     # [B*T]
    mask_flat    = mask.reshape(-1)        # [B*T]

    loss_all = nn.losses.cross_entropy(
        logits_flat,
        targets_flat,
        label_smoothing=label_smoothing,
        reduction="none",
    )                                      # [B*T]

    # Apply mask and average over valid positions
    masked = loss_all * mask_flat.astype(loss_all.dtype)
    n_valid = mx.maximum(mask_flat.sum(), 1)
    return masked.sum() / n_valid


def _build_codec_prefix(talker, lang_codes) -> mx.array:
    """
    Build per-sample codec prefix embeds that match Qwen3-TTS inference behaviour.

    auto   → [nothink, think_bos, think_eos, pad, bos]           (5 tokens)
    <lang> → [think,  think_bos, lang_id,   think_eos, pad, bos] (6 tokens)

    lang_codes: list[str] of length B, or a single str (broadcast to all samples).

    Returns:
        prefix_embeds [B, T_prefix, D]
        T_prefix       int  (uniform across samples; shorter auto-prefixes are
                             left-padded with codec_pad_id to match lang prefixes)
    """
    cfg         = talker.config
    codec_embed = talker.get_input_embeddings()

    # Accept a single string for backward compatibility
    if isinstance(lang_codes, str):
        lang_codes = [lang_codes]

    per_sample_ids = []
    for lang_code in lang_codes:
        lang_id = None
        if lang_code != "auto" and cfg.codec_language_id:
            lang_id = cfg.codec_language_id.get(lang_code.lower())

        if lang_id is not None:
            ids = [cfg.codec_think_id, cfg.codec_think_bos_id,
                   lang_id, cfg.codec_think_eos_id,
                   cfg.codec_pad_id, cfg.codec_bos_id]
        else:
            ids = [cfg.codec_nothink_id, cfg.codec_think_bos_id,
                   cfg.codec_think_eos_id,
                   cfg.codec_pad_id, cfg.codec_bos_id]
        per_sample_ids.append(ids)

    # Pad shorter prefixes (auto=5) to match longer ones (lang=6) with codec_pad_id
    T_prefix = max(len(ids) for ids in per_sample_ids)
    for i, ids in enumerate(per_sample_ids):
        if len(ids) < T_prefix:
            per_sample_ids[i] = [cfg.codec_pad_id] * (T_prefix - len(ids)) + ids

    prefix = mx.array(per_sample_ids)       # [B, T_prefix]
    return codec_embed(prefix), T_prefix    # [B, T_prefix, D], T_prefix


def qwen3_tts_loss(
    model,
    batch: Dict[str, mx.array],
    sub_talker_weight: float = 0.3,
    label_smoothing:   float = 0.0,
    lang_code:         str   = "auto",
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Compute Qwen3-TTS training loss.

    The model takes text embeddings + codec embeddings as input and
    predicts codec tokens autoregressively.

    Sequence layout (teacher-forced):
        input:   [text_embeds | codec_prefix | codec_embeds[0:-1]]
        labels:  [IGNORE      | IGNORE       | codec_ids[1:]     ]

    The codec_prefix matches the inference-time prefix (nothink/think + language
    token) so training and inference see the same conditioning context.

    Args:
        model:   Qwen3-TTS Model instance (mlx_audio)
        batch:   dict with keys: text_ids, codec_ids, text_lengths, codec_lengths, text_mask, codec_mask
        sub_talker_weight: weight for the code predictor auxiliary loss (default 0.3)
        label_smoothing:   optional label smoothing
        lang_code:         language code to inject ("auto" = nothink prefix, no language token)

    Returns:
        (total_loss, metrics_dict)
    """
    text_ids   = batch["text_ids"]       # [B, T_text]
    codec_ids  = batch["codec_ids"]      # [B, T_codec]
    codec_mask = batch["codec_mask"]     # [B, T_codec]

    B = text_ids.shape[0]

    # ── Embeddings ──────────────────────────────────────────────────────────

    talker      = model.talker
    text_embed  = talker.get_text_embeddings()    # text token embedding table
    codec_embed = talker.get_input_embeddings()   # codec token embedding table

    text_embeds  = text_embed(text_ids)                    # [B, T_text, D_text]
    text_embeds  = talker.text_projection(text_embeds)     # [B, T_text, D_model]

    # ── Codec prefix (matches inference conditioning) ────────────────────────
    # Per-sample lang_codes from batch override the config-level lang_code fallback.
    lang_codes = batch.get("lang_codes") or [lang_code] * B
    prefix_embeds, T_prefix = _build_codec_prefix(talker, lang_codes)  # [B, T_prefix, D]

    # Teacher-forced: feed [prefix | codec[0:-1]], predict codec[1:]
    codec_input   = codec_embed(codec_ids[:, :-1])                       # [B, T_codec-1, D]
    codec_input   = mx.concatenate([prefix_embeds, codec_input], axis=1) # [B, T_prefix+T_codec-1, D]
    inputs_embeds = mx.concatenate([text_embeds, codec_input], axis=1)   # [B, T_text+T_prefix+T_codec-1, D]

    # ── Forward pass ────────────────────────────────────────────────────────

    logits, hidden_states = talker(inputs_embeds)  # [B, T_total, vocab], [B, T_total, D]

    T_text_max = text_ids.shape[1]

    # Take only the codec portion of logits (after text + prefix)
    codec_offset = T_text_max - 1 + T_prefix
    codec_logits = logits[:, codec_offset: codec_offset + codec_ids.shape[1] - 1, :]
    # [B, T_codec-1, vocab]

    # Labels = codec_ids[1:]
    codec_targets = codec_ids[:, 1:]              # [B, T_codec-1]
    loss_mask     = codec_mask[:, 1:]             # [B, T_codec-1]

    # ── Main talker loss ────────────────────────────────────────────────────

    main_loss = cross_entropy_masked(
        codec_logits, codec_targets, loss_mask, label_smoothing
    )

    # ── Sub-talker (code predictor) loss ────────────────────────────────────

    sub_loss = mx.array(0.0)
    if sub_talker_weight > 0 and hasattr(talker, "code_predictor"):
        codec_hidden = hidden_states[:, T_text_max - 1: T_text_max - 1 + codec_ids.shape[1] - 1, :]
        sub_out = talker.code_predictor(codec_hidden)
        # code_predictor returns (logits, cache, n) — take first element
        sub_logits = sub_out[0] if isinstance(sub_out, tuple) else sub_out
        sub_loss = cross_entropy_masked(
            sub_logits, codec_targets, loss_mask, label_smoothing
        )

    total_loss = main_loss + sub_talker_weight * sub_loss

    metrics = {
        "loss":      float(total_loss),
        "main_loss": float(main_loss),
        "sub_loss":  float(sub_loss),
    }

    return total_loss, metrics


def qwen3_tts_speaker_loss(
    model,
    batch: Dict[str, mx.array],
    sub_talker_weight: float = 0.3,
    label_smoothing:   float = 0.0,
    lang_code:         str   = "auto",
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Qwen3-TTS training loss with speaker-embedding injection.

    Mirrors the official sft_12hz.py speaker-cloning pipeline:
      1. Extract a speaker embedding from the ref-audio mel spectrogram via
         model.speaker_encoder (kept frozen — no gradients flow into it).
      2. Add the speaker embedding as a per-sample conditioning vector to
         ALL codec embedding positions before the forward pass.
         (The official code injects it at a specific sequence position; here
         we broadcast-add it across the codec slice, which is functionally
         equivalent for LoRA adapters.)
      3. Optionally add all 16 codec-level embeddings as input conditioning
         when model.talker.code_predictor exposes its embedding tables.
      4. Compute main_loss + sub_talker_weight * sub_loss as usual.

    Batch keys (beyond the base keys used by qwen3_tts_loss):
        ref_mel [B, T_mel, 128]   — padded mel spectrograms of ref audios
        ref_mel_lengths [B]       — valid frame counts (unused for forward, for reference)

    Falls back to qwen3_tts_loss behaviour when ref_mel is absent.
    """
    text_ids      = batch["text_ids"]
    codec_ids     = batch["codec_ids"]
    codec_mask    = batch["codec_mask"]
    ref_mel       = batch.get("ref_mel")        # [B, T_mel, 128] or None

    talker      = model.talker
    text_embed  = talker.get_text_embeddings()
    codec_embed = talker.get_input_embeddings()

    text_embeds  = text_embed(text_ids)
    text_embeds  = talker.text_projection(text_embeds)

    B = text_ids.shape[0]

    # ── Codec prefix (matches inference conditioning) ────────────────────────
    lang_codes = batch.get("lang_codes") or [lang_code] * B
    prefix_embeds, T_prefix = _build_codec_prefix(talker, lang_codes)  # [B, T_prefix, D]

    # Codec teacher-forcing input: [prefix | codec[0:-1]]
    codec_input  = codec_embed(codec_ids[:, :-1])                        # [B, T_codec-1, D]

    # ── Speaker conditioning ─────────────────────────────────────────────────
    if ref_mel is not None and hasattr(model, "speaker_encoder"):
        # speaker_encoder is frozen; stop_gradient makes this explicit so MLX
        # does not attempt to build a gradient path through it.
        spk_embed = mx.stop_gradient(model.speaker_encoder(ref_mel))  # [B, D]
        # Broadcast-add over the codec time axis — every codec position sees
        # the same speaker identity signal, equivalent to the official code's
        # position-6 injection in terms of conditioning strength.
        codec_input = codec_input + spk_embed[:, None, :]

    # ── Optional: all 16 codec-level input embeddings (official style) ───────
    # The official sft_12hz.py adds levels 1-15 as additive embeddings so the
    # transformer can exploit the full VQ residual hierarchy as input context.
    if hasattr(talker, "code_predictor"):
        cp = talker.code_predictor
        emb_tables = None
        if hasattr(cp, "get_input_embeddings"):
            try:
                emb_tables = cp.get_input_embeddings()
            except Exception:
                pass

        if emb_tables is not None:
            for lvl in range(1, 16):
                try:
                    emb_i = emb_tables[lvl - 1](codec_ids[:, :-1])  # [B, T-1, D]
                    # Apply codec_mask so padding positions are zeroed out
                    mask_i = codec_mask[:, :-1].astype(emb_i.dtype)
                    codec_input = codec_input + emb_i * mask_i[:, :, None]
                except (IndexError, Exception):
                    break  # fewer than 16 levels in this model variant

    # ── Forward ──────────────────────────────────────────────────────────────
    codec_input   = mx.concatenate([prefix_embeds, codec_input], axis=1)  # [B, T_prefix+T_codec-1, D]
    inputs_embeds = mx.concatenate([text_embeds, codec_input], axis=1)
    logits, hidden_states = talker(inputs_embeds)

    T_text_max    = text_ids.shape[1]
    T_codec       = codec_ids.shape[1]
    codec_offset  = T_text_max - 1 + T_prefix
    codec_logits  = logits[:, codec_offset: codec_offset + T_codec - 1, :]
    codec_targets = codec_ids[:, 1:]
    loss_mask     = codec_mask[:, 1:]

    main_loss = cross_entropy_masked(codec_logits, codec_targets, loss_mask, label_smoothing)

    sub_loss = mx.array(0.0)
    if sub_talker_weight > 0 and hasattr(talker, "code_predictor"):
        codec_hidden = hidden_states[:, codec_offset: codec_offset + T_codec - 1, :]
        sub_out  = talker.code_predictor(codec_hidden)
        sub_logits = sub_out[0] if isinstance(sub_out, tuple) else sub_out
        sub_loss = cross_entropy_masked(sub_logits, codec_targets, loss_mask, label_smoothing)

    total_loss = main_loss + sub_talker_weight * sub_loss

    metrics = {
        "loss":      float(total_loss),
        "main_loss": float(main_loss),
        "sub_loss":  float(sub_loss),
    }
    return total_loss, metrics


def csm_loss(
    model,
    batch: Dict[str, mx.array],
    label_smoothing: float = 0.0,
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Cross-entropy loss for CSM (codec token LM).
    """
    text_ids   = batch["text_ids"]    # [B, T_text]
    audio_ids  = batch["audio_ids"]   # [B, T_audio, 32]
    audio_mask = batch["audio_mask"]  # [B, T_audio]

    # Only predict first codebook (codebook0_head)
    # Remaining codebooks handled by decoder (not trained in LoRA pass)
    labels_cb0 = audio_ids[:, 1:, 0]   # [B, T_audio-1]
    mask       = audio_mask[:, 1:]      # [B, T_audio-1]

    # Build input: text tokens + audio[0:-1]
    text_embeds  = model.model.embed_tokens(text_ids)
    audio_embeds = model.model.audio_embed(audio_ids[:, :-1, 0])
    inputs       = mx.concatenate([text_embeds, audio_embeds], axis=1)

    hidden = model.model(inputs)
    T_text = text_ids.shape[1]
    audio_hidden = hidden[:, T_text:, :]

    logits = model.codebook0_head(audio_hidden)  # [B, T_audio-1, vocab]

    loss = cross_entropy_masked(logits, labels_cb0, mask, label_smoothing)

    return loss, {"loss": float(loss)}
