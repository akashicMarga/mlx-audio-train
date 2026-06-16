"""
losses/grpo_loss.py — Phase B of GRPO for Qwen3-TTS.

Objective (all terms on cb0 unless noted):

    L = pg_main                       # advantage-weighted −logπ_θ, NO label smoothing
      + sub_pg_weight · pg_sub        # optional: same, on code_predictor lm_head[0]
      + kl_beta       · kl_k3         # k3 KL(π_θ ‖ π_ref), per-token, ≥0
      + sft_lambda    · sft_ce        # optional: qwen3_tts_loss on a real batch (anchor)

pg_main is the only term that, by itself, optimises the reward; the rest are
anchors. It fits the Trainer contract `fn(model, batch) → (loss, metrics)`, so
`train/trainer.py` is reused as-is — only the *data* changes (rollout groups
instead of dataset batches).

`batch` is ONE prompt's rollout group (see GRPOTrainer):
    text_ids     [G, T_text]   (G identical copies of the prompt)
    codec_ids    [G, T]        sampled cb0 (right-padded after EOS)
    codec_mask   [G, T]        bool, valid frames
    advantages   [G]           group-relative, std-normalised
    ref_logprobs [G, T]        cb0 logp under the reference policy (no grad)
    lang_codes   list[str]     length G
    spk_embeds   [G, D]        optional (Pipeline 2)
    reward_mean  float         optional, raw group reward mean (logging only)
    sft_batch    dict          optional, a real SFT batch for the mixin term

The policy forward here runs with LoRA ENABLED (the trainer never disables it
during the grad step); `ref_logprobs` was precomputed in Phase A against the
chosen reference. See [[grpo-qwen3-design]] for the reference-anchoring choice.
"""

from typing import Dict, Tuple

import mlx.core as mx

from train.grpo.rollout import grpo_codec_logits, gather_token_logprobs
from train.losses.codec_loss import qwen3_tts_loss


def qwen3_tts_grpo_loss(
    model,
    batch: Dict[str, mx.array],
    *,
    kl_beta:        float = 0.05,
    kl_clip:        float = 10.0,   # clamp |logπ_ref − logπ_θ| before exp (k3 stability)
    sub_pg_weight:  float = 0.0,
    sft_lambda:     float = 0.0,
    label_smoothing: float = 0.0,   # only applied to the SFT-mixin term
    lang_code:      str   = "auto",
) -> Tuple[mx.array, Dict[str, float]]:
    codec_ids    = batch["codec_ids"]            # text_ids only needed (concatenated)
    codec_mask   = batch["codec_mask"]
    advantages   = batch["advantages"]            # [G]
    ref_logprobs = batch["ref_logprobs"]          # [G, T], no grad
    spk_embeds   = batch.get("spk_embeds")
    G            = codec_ids.shape[0]

    mask    = codec_mask.astype(mx.float32)       # [G, T]
    n_valid = mx.maximum(mask.sum(), 1.0)
    adv     = advantages[:, None]                 # [G, 1]
    T       = codec_ids.shape[1]
    talker  = model.talker

    # ── Policy forward (LoRA on) — cb0 logits + hidden, validated alignment ──
    if "tf_input_embeds" in batch:
        # Interleaved layout: replay the recorded generate()-path input embeds
        # (frozen, so constant w.r.t. LoRA); grad flows through the transformer.
        logits_full, hidden = talker(batch["tf_input_embeds"])
        codec_offset = int(batch["cb0_offset"])
        logits = logits_full[:, codec_offset: codec_offset + T, :]
    else:
        # Concatenated layout (SFT-matched): rebuild the forward from ids.
        lang_codes = batch.get("lang_codes") or [lang_code] * G
        logits, codec_offset, hidden = grpo_codec_logits(
            model, batch["text_ids"], codec_ids, lang_codes, spk_embeds=spk_embeds
        )                                         # [G, T, V], int, [G, L, D]
    logp = gather_token_logprobs(logits, codec_ids)   # [G, T] = logπ_θ(sampled)

    # ── Policy-gradient term (advantage-weighted NLL, no smoothing) ──────────
    pg = -(adv * logp * mask).sum() / n_valid

    # ── k3 KL(π_θ ‖ π_ref): exp(d) − d − 1,  d = logπ_ref − logπ_θ  (≥0) ─────
    # Clamp d before exp: a sampled token deep in the tail can make d large and
    # blow exp(d) up (and the gradient with it). The clamp bounds that without
    # changing behaviour in the normal small-drift regime.
    d  = mx.clip(mx.stop_gradient(ref_logprobs) - logp, -kl_clip, kl_clip)
    kl = ((mx.exp(d) - d - 1.0) * mask).sum() / n_valid

    loss = pg + kl_beta * kl

    # ── Optional sub-talker PG (code_predictor lm_head[0]) ──────────────────
    sub_pg = mx.array(0.0)
    if sub_pg_weight > 0 and hasattr(talker, "code_predictor"):
        codec_hidden = hidden[:, codec_offset: codec_offset + T, :]
        sub_out    = talker.code_predictor(codec_hidden)
        sub_logits = sub_out[0] if isinstance(sub_out, tuple) else sub_out
        sub_logp   = gather_token_logprobs(sub_logits, codec_ids)
        sub_pg     = -(adv * sub_logp * mask).sum() / n_valid
        loss       = loss + sub_pg_weight * sub_pg

    # ── Optional SFT-mixin (CE anchor on real data) ─────────────────────────
    sft_ce = mx.array(0.0)
    sft_batch = batch.get("sft_batch")
    if sft_lambda > 0 and sft_batch is not None:
        sft_ce, _ = qwen3_tts_loss(
            model, sft_batch, label_smoothing=label_smoothing, lang_code=lang_code
        )
        loss = loss + sft_lambda * sft_ce

    metrics = {
        "loss":        float(loss),
        "pg":          float(pg),
        "kl":          float(kl),
        "sub_pg":      float(sub_pg),
        "sft":         float(sft_ce),
        "reward_mean": float(batch.get("reward_mean", 0.0)),
        "cer_mean":    float(batch.get("cer_mean", 0.0)),
        "adv_abs":     float(mx.abs(advantages).mean()),
        "logp_mean":   float((logp * mask).sum() / n_valid),
    }
    return loss, metrics
