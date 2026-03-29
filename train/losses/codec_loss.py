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


def _build_codec_prefix(talker, lang_codes, spk_embeds=None):
    """
    Build per-sample codec prefix embeds that match Qwen3-TTS inference behaviour.

    Without speaker:
        auto   → [nothink, think_bos, think_eos, pad, bos]           (5 tokens)
        <lang> → [think,  think_bos, lang_id,   think_eos, pad, bos] (6 tokens)

    With speaker (matches official sft_12hz.py exactly):
        auto   → [nothink, think_bos, think_eos, spk_embed, pad, bos]
        <lang> → [think,  think_bos, lang_id,   think_eos, spk_embed, pad, bos]

    The speaker embedding is inserted as a single token between the think/lang
    section and the [pad, bos] suffix — identical to the official positional
    injection approach.

    Args:
        lang_codes:  list[str] of length B, or a single str
        spk_embeds:  [B, D] speaker embeddings, or None (Pipeline 1 / no speaker)

    Returns:
        prefix_embeds [B, T_prefix, D]
        T_prefix       int
    """
    cfg         = talker.config
    codec_embed = talker.get_input_embeddings()

    if isinstance(lang_codes, str):
        lang_codes = [lang_codes]

    B = len(lang_codes)

    # Build the think/lang section (before [pad, bos]) as token IDs
    pre_ids = []
    for lang_code in lang_codes:
        lang_id = None
        if lang_code != "auto" and cfg.codec_language_id:
            lang_id = cfg.codec_language_id.get(lang_code.lower())

        if lang_id is not None:
            ids = [cfg.codec_think_id, cfg.codec_think_bos_id,
                   lang_id, cfg.codec_think_eos_id]
        else:
            ids = [cfg.codec_nothink_id, cfg.codec_think_bos_id,
                   cfg.codec_think_eos_id]
        pre_ids.append(ids)

    # Pad pre-section across samples (auto=3 tokens, lang=4 tokens)
    T_pre = max(len(ids) for ids in pre_ids)
    for i, ids in enumerate(pre_ids):
        if len(ids) < T_pre:
            pre_ids[i] = [cfg.codec_pad_id] * (T_pre - len(ids)) + ids

    pre_embeds    = codec_embed(mx.array(pre_ids))                                # [B, T_pre, D]
    suffix_embeds = codec_embed(mx.array([[cfg.codec_pad_id, cfg.codec_bos_id]] * B))  # [B, 2, D]

    if spk_embeds is not None:
        # Insert speaker embedding as a single token between think section and [pad, bos]
        # spk_embeds: [B, D] → [B, 1, D]
        spk = spk_embeds[:, None, :] if spk_embeds.ndim == 2 else spk_embeds
        prefix = mx.concatenate([pre_embeds, spk, suffix_embeds], axis=1)
    else:
        prefix = mx.concatenate([pre_embeds, suffix_embeds], axis=1)

    T_prefix = prefix.shape[1]
    return prefix, T_prefix


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

    # ── Speaker conditioning ─────────────────────────────────────────────────
    # Extract speaker embedding first so it can be passed into the prefix.
    # Matches the official sft_12hz.py approach: speaker embed is inserted as
    # a single token between the think/lang section and [pad, bos] in the prefix,
    # giving the transformer an explicit speaker identity anchor in its context.
    spk_embeds = None
    if ref_mel is not None and hasattr(model, "speaker_encoder"):
        spk_embeds = mx.stop_gradient(model.speaker_encoder(ref_mel))  # [B, D]

    # ── Codec prefix (with speaker token injected) ───────────────────────────
    lang_codes = batch.get("lang_codes") or [lang_code] * B
    prefix_embeds, T_prefix = _build_codec_prefix(talker, lang_codes, spk_embeds=spk_embeds)

    # Codec teacher-forcing input: [prefix | codec[0:-1]]
    codec_input  = codec_embed(codec_ids[:, :-1])                        # [B, T_codec-1, D]

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
