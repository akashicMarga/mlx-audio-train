"""
train/losses/parler_loss.py — Parler TTS training loss.

Parler TTS decoder forward pass (teacher-forced):
    enc_hidden  [B, T_desc, 1024]   ← T5 encoder output (frozen)
    prompt_emb  [B, T_prompt, 1024] ← embed_prompts output (frozen)
    input_seq:  [prompt_emb | codes_emb_delayed[:-1]]
    cross-attn to enc_hidden at each decoder step
    output logits: [B, T_seq, 9, lm_vocab_size]

Loss (delay-pattern cross-entropy):
    For each codebook k at position t, predict codes[k, t+1].
    Only positions where codes_mask[t+1] == True are counted.
    Loss = mean CE across all 9 codebooks × valid positions.

Batch keys (from IndicParlerProcessor.collate):
    desc_ids:    [B, T_desc]
    desc_mask:   [B, T_desc]    bool, True = non-padding
    prompt_ids:  [B, T_prompt]
    codes:       [B, 9, Tp]     delay-pattern applied by processor
    codes_mask:  [B, Tp]        True = real frame (all codebooks active)
"""

from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn


def _masked_ce(
    logits: mx.array,  # [B, T, vocab]
    targets: mx.array, # [B, T]
    mask: mx.array,    # [B, T] bool
    label_smoothing: float = 0.0,
) -> mx.array:
    B, T, V = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        label_smoothing=label_smoothing,
        reduction="none",
    )  # [B*T]
    mask_flat = mask.reshape(B * T).astype(loss.dtype)
    return (loss * mask_flat).sum() / mx.maximum(mask_flat.sum(), 1)


def _build_encoder_mask(desc_mask: mx.array) -> mx.array:
    """
    Convert boolean desc_mask [B, T_desc] → additive cross-attention bias [B, 1, 1, T_desc].
    0 for valid tokens, large negative for padding — added to attention scores.
    """
    # True = valid → 0 bias; False = pad → large negative
    bias = (1.0 - desc_mask.astype(mx.float32)) * -1e9
    return bias[:, None, None, :]  # [B, 1, 1, T_desc]


def parler_tts_loss(
    model,
    batch: Dict[str, mx.array],
    label_smoothing: float = 0.0,
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Compute Parler TTS training loss.

    model: IndicParlerTTS (from models/indic_parler_tts/model.py)
    batch: output of IndicParlerProcessor.collate()

    Returns (loss_scalar, metrics_dict).
    """
    desc_ids    = batch["desc_ids"]     # [B, T_desc]
    desc_mask   = batch["desc_mask"]    # [B, T_desc] bool
    prompt_ids  = batch["prompt_ids"]   # [B, T_prompt]
    codes       = batch["codes"]        # [B, 9, Tp]
    codes_mask  = batch["codes_mask"]   # [B, Tp]

    B, K, Tp = codes.shape
    cfg = model.cfg.decoder
    decoder = model.decoder

    # ── T5 encoder (frozen) ───────────────────────────────────────────────────
    # text_encoder only takes input_ids (no mask) — mask is used in cross-attn
    enc_hidden = mx.stop_gradient(model.text_encoder(desc_ids))  # [B, T_desc, 1024]
    encoder_mask = _build_encoder_mask(desc_mask)                # [B, 1, 1, T_desc]

    # ── Prompt embeddings (frozen) ────────────────────────────────────────────
    prompt_emb = mx.stop_gradient(model.embed_prompts(prompt_ids))  # [B, T_prompt, 1024]
    T_prompt = prompt_ids.shape[1]

    # Add position embeddings to prompt (positions 0..T_prompt-1)
    prompt_pos = decoder.embed_positions.weight[:T_prompt][None, :, :]  # [1, T_prompt, 1024]
    prompt_emb = prompt_emb + prompt_pos

    # ── Codec embeddings (teacher-forced input: codes[:, :, :-1]) ─────────────
    # The processor stores delay-pattern codes in batch["codes"] [B, 9, Tp].
    # Input to decoder at frame t: sum of embed_tokens[k](codes[k, t]) for k=0..8
    codes_in = codes[:, :, :-1]       # [B, 9, Tp-1] — input frames (drop last)
    T_codec = Tp - 1

    # Sum embeddings across K codebooks at each frame
    codes_emb = sum(
        decoder.embed_tokens[k](codes_in[:, k, :]) for k in range(K)
    )  # [B, T_codec, 1024]

    # Add position embeddings (positions T_prompt..T_prompt+T_codec-1)
    codec_pos = decoder.embed_positions.weight[T_prompt : T_prompt + T_codec][None, :, :]
    codes_emb = codes_emb + codec_pos

    # ── Full input sequence ───────────────────────────────────────────────────
    input_seq = mx.concatenate([prompt_emb, codes_emb], axis=1)
    # [B, T_prompt + T_codec, 1024]

    T_seq = input_seq.shape[1]

    # ── Causal self-attention mask ────────────────────────────────────────────
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T_seq)
    causal_mask = causal_mask[None, None, :, :].astype(input_seq.dtype)  # [1, 1, T_seq, T_seq]

    # ── Decoder forward ───────────────────────────────────────────────────────
    hidden = decoder.forward_layers(
        input_seq,
        enc_hidden,
        mask=causal_mask,
        encoder_mask=encoder_mask,
    )  # [B, T_seq, 1024]

    # Take codec portion only (after prompt tokens)
    codec_hidden = hidden[:, T_prompt:, :]  # [B, T_codec, 1024]

    # Logits: [B, T_codec, 9, lm_vocab_size]
    logits = decoder.logits(codec_hidden)

    # ── Loss: output[t] predicts codes[:, t+1] ────────────────────────────────
    # Targets: codes[k, 1:] for each k
    codes_target = codes[:, :, 1:]  # [B, 9, Tp-1]
    mask_target  = codes_mask[:, 1:]  # [B, Tp-1] — shift mask along with targets

    total_loss = mx.array(0.0)
    cb_losses  = []
    for k in range(K):
        logits_k  = logits[:, :, k, :]    # [B, T_codec, lm_vocab_size]
        targets_k = codes_target[:, k, :] # [B, T_codec]
        loss_k    = _masked_ce(logits_k, targets_k, mask_target, label_smoothing)
        total_loss = total_loss + loss_k
        cb_losses.append(float(loss_k))

    total_loss = total_loss / K

    metrics = {
        "loss":         float(total_loss),
        **{f"loss_cb{k}": cb_losses[k] for k in range(K)},
    }
    return total_loss, metrics
