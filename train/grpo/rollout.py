"""
grpo/rollout.py — Phase A of GRPO: on-policy rollouts for Qwen3-TTS.

Two layouts (chosen per run via `layout=`):

  • "concatenated" (sample_rollouts): input is
    `[text_embeds | codec_prefix | codec_embeds[0:-1]]`, identical to
    `qwen3_tts_loss`. Matches SFT training, but NOT the official interleaved
    `model.generate()` — so what GRPO optimizes (concatenated generation) differs
    from how the model is actually used (interleaved), capping the real gain.

  • "interleaved" (sample_rollouts_interleaved): mirrors `model.generate()` /
    `_generate_batch` exactly — text is streamed frame-by-frame via
    `trailing_text_hidden` and summed with all 16 codebook embeddings. This is
    the deployment path, so optimizing it directly removes the train/inference
    skew. Key enabler: the per-frame input embeddings are built from FROZEN
    tables (codec/text embeddings, text_projection — none are LoRA targets), so
    they are constant w.r.t. the trainable params. We therefore *record* the
    exact input-embedding sequence during sampling and *replay* it as one
    teacher-forced forward in Phase B (gradients flow through the transformer
    only) — guaranteeing Phase A↔Phase B agreement.

Either way only **codebook 0** is the policy (talker `codec_head`); cb1–15 come
from the frozen `code_predictor` (sampled to render audio + form the interleaved
feedback, but not part of the PG/KL log-probs).

Public API:
    grpo_codec_logits(model, text_ids, codec_ids, lang_codes, spk_embeds)
        -> (cb0_logits [G, T, V], codec_offset, hidden)   # concatenated forward
    gather_token_logprobs(logits, targets)                # [G, T]
    sample_rollouts(model, text_ids, ...)                 -> dict   (concatenated)
    sample_rollouts_interleaved(model, text, ...)         -> dict   (interleaved)
    decode_codes_to_audio(model, full_codes, mask)        -> list[mx.array]
"""

from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from train.losses.codec_loss import _build_codec_prefix
from train.lora import lora_snapshot, lora_swapped


# ──────────────────────────────────────────────────────────────────────────────
# Shared teacher-forced forward (SFT concatenated layout, cb0 only)
# ──────────────────────────────────────────────────────────────────────────────

def grpo_codec_logits(
    model,
    text_ids:   mx.array,            # [G, T_text]
    codec_ids:  mx.array,            # [G, T] sampled cb0 sequence
    lang_codes: Sequence[str],
    spk_embeds: Optional[mx.array] = None,
):
    """Cb0 logits aligned so that logits[:, k] is the distribution that produced
    codec_ids[:, k] at sample time. Returns (logits [G, T, V], codec_offset,
    hidden [G, L, D]). `hidden` (full sequence) lets the loss slice codec hidden
    states at `codec_offset` for the optional sub-talker (code_predictor) term.

    Layout mirrors qwen3_tts_loss EXCEPT the slice: GRPO scores *every* sampled
    token (targets = codec_ids), so the codec window is length T starting at the
    last prefix token (the BOS), instead of dropping codec_ids[0]. See the
    module docstring / the rollout↔loss alignment note.
    """
    talker      = model.talker
    text_embed  = talker.get_text_embeddings()
    codec_embed = talker.get_input_embeddings()

    text_embeds = talker.text_projection(text_embed(text_ids))      # [G, T_text, D]

    prefix_embeds, T_prefix = _build_codec_prefix(
        talker, list(lang_codes), spk_embeds=spk_embeds
    )                                                               # [G, T_prefix, D]

    codec_input   = codec_embed(codec_ids[:, :-1])                  # [G, T-1, D]
    codec_input   = mx.concatenate([prefix_embeds, codec_input], axis=1)
    inputs_embeds = mx.concatenate([text_embeds, codec_input], axis=1)

    logits, hidden = talker(inputs_embeds)                         # [G, L, V], [G, L, D]

    T_text = text_ids.shape[1]
    T      = codec_ids.shape[1]
    # Last prefix token (BOS) sits at index T_text + T_prefix - 1 and its logit
    # predicts codec_ids[0]; the next T-1 positions predict codec_ids[1:].
    codec_offset = T_text - 1 + T_prefix
    cb0_logits   = logits[:, codec_offset: codec_offset + T, :]    # [G, T, V]
    return cb0_logits, codec_offset, hidden


def gather_token_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """Log p(target) per position. logits [G, T, V], targets [G, T] -> [G, T].

    log_softmax is computed in float32 regardless of the logits dtype: bfloat16
    log_softmax is wildly inaccurate in the low-probability tail, and GRPO
    compares policy vs reference log-probs (and exponentiates their difference in
    the k3 KL), so both sides must use the same, accurate precision.
    """
    logprobs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    return mx.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]


# ──────────────────────────────────────────────────────────────────────────────
# Phase A: autoregressive cb0 sampling + frozen code_predictor for cb1..15
# ──────────────────────────────────────────────────────────────────────────────

def sample_rollouts(
    model,
    text_ids:        Union[mx.array, Sequence[int]],   # [T_text] one prompt
    *,
    lang_code:       str = "auto",
    group_size:      int = 4,
    max_new_tokens:  int = 240,
    temperature:     float = 0.9,
    top_p:           float = 0.95,
    top_k:           int = 50,
    spk_embeds:      Optional[mx.array] = None,         # [1, D] for Pipeline 2
    compute_ref:     bool = True,
    ref_params:      Optional[Dict[str, mx.array]] = None,
) -> Dict[str, mx.array]:
    """Sample `group_size` cb0 sequences for ONE prompt, decode-ready.

    Batching one prompt at a time (G identical copies) keeps every sequence the
    same length, so the talker needs no padding/attention mask. Drive B prompts
    by calling this B times.

    `ref_params` is the frozen reference adapter snapshot (from `lora_snapshot`
    at GRPO start) used for the KL reference; if None, a snapshot of the *current*
    adapter is used (ref == policy → KL 0, only useful for standalone testing).

    Returns dict:
        codec_ids    [G, T]      sampled cb0 (right-padded with codec_pad_id after EOS)
        codec_mask   [G, T]      bool, True for frames 0..EOS inclusive
        full_codes   [G, T, 16]  all codebooks (cb0 + frozen code_predictor) for decode
        ref_logprobs [G, T]      cb0 logp under the reference policy (float32, no grad)
                                 (zeros where compute_ref=False)
        gen_lengths  [G]         valid length per sequence (incl. EOS)
    """
    cfg     = model.talker.config
    eos_id  = cfg.codec_eos_token_id
    pad_id  = cfg.codec_pad_id
    n_groups = cfg.num_code_groups
    talker  = model.talker

    # Special-token range to suppress during cb0 sampling (mirror generate()).
    suppress = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != eos_id]

    text_ids = mx.array(text_ids) if not isinstance(text_ids, mx.array) else text_ids
    if text_ids.ndim == 1:
        text_ids = text_ids[None, :]
    text_ids = mx.broadcast_to(text_ids, (group_size, text_ids.shape[1]))  # [G, T_text]

    text_embeds = talker.text_projection(talker.get_text_embeddings()(text_ids))
    lang_codes  = [lang_code] * group_size
    prefix_embeds, _ = _build_codec_prefix(talker, lang_codes, spk_embeds=spk_embeds)

    # Prime the talker with [text | prefix]; the prefix ends in BOS, so the last
    # position's logit is exactly ctx_0 → s_0 from the module's alignment note.
    inputs_embeds = mx.concatenate([text_embeds, prefix_embeds], axis=1)
    cache      = talker.make_cache()
    code_cache = talker.code_predictor.make_cache()

    finished     = mx.zeros((group_size,), dtype=mx.bool_)
    eos_fill     = mx.full((group_size, 1), eos_id, dtype=mx.int32)
    cb0_steps:  List[mx.array] = []   # each [G, 1]
    frame_steps: List[mx.array] = []  # each [G, n_groups]
    gen_lengths = [max_new_tokens] * group_size

    step_input = inputs_embeds
    for step in range(max_new_tokens):
        logits, hidden = talker(step_input, cache=cache)

        cb0 = model._sample_token_batch(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            suppress_tokens=suppress, eos_token_id=eos_id,
        )                                                       # [G, 1]
        cb0 = mx.where(finished[:, None], eos_fill, cb0)

        newly = cb0[:, 0] == eos_id
        # record length at first EOS
        for b in range(group_size):
            if bool(newly[b]) and not bool(finished[b]):
                gen_lengths[b] = step + 1
        finished = finished | newly

        # --- frozen code_predictor: cb1..15 for this frame (decode only) -------
        code_tokens = [cb0]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None; c.values = None; c.offset = 0
        for code_idx in range(n_groups - 1):
            if code_idx == 0:
                code_0_embed = talker.get_input_embeddings()(cb0)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_input = talker.code_predictor.codec_embedding[code_idx - 1](
                    code_tokens[-1]
                )
            code_logits, code_cache, _ = talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx,
            )
            nxt = model._sample_token_batch(
                code_logits, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            code_tokens.append(nxt)
        frame = mx.concatenate(code_tokens, axis=1)             # [G, n_groups]

        cb0_steps.append(cb0)
        frame_steps.append(frame)

        # SFT-layout feedback: next codec input is the cb0 embedding ONLY.
        step_input = talker.get_input_embeddings()(cb0)         # [G, 1, D]

        mx.eval(cb0, frame, finished)
        if bool(mx.all(finished)):
            break
        if step > 0 and step % 50 == 0:
            mx.clear_cache()

    T = len(cb0_steps)
    codec_ids  = mx.concatenate(cb0_steps, axis=1)              # [G, T]
    full_codes = mx.stack(frame_steps, axis=1)                 # [G, T, n_groups]

    pos = mx.arange(T)[None, :]
    glen = mx.array(gen_lengths)[:, None]
    codec_mask = pos < glen                                    # True for 0..EOS inclusive
    # Right-pad cb0 after EOS with pad_id (keeps embeddings well-defined).
    codec_ids = mx.where(codec_mask, codec_ids, mx.array(pad_id, dtype=codec_ids.dtype))

    ref_logprobs = mx.zeros((group_size, T))
    if compute_ref:
        snapshot = ref_params if ref_params is not None else lora_snapshot(model)
        with lora_swapped(model, snapshot):
            ref_logits, _, _ = grpo_codec_logits(
                model, text_ids, codec_ids, lang_codes, spk_embeds=spk_embeds
            )
            ref_logprobs = mx.stop_gradient(gather_token_logprobs(ref_logits, codec_ids))
        mx.eval(ref_logprobs)

    return {
        "codec_ids":    codec_ids,
        "codec_mask":   codec_mask,
        "full_codes":   full_codes,
        "ref_logprobs": ref_logprobs,
        "gen_lengths":  mx.array(gen_lengths),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Interleaved layout — mirrors model.generate() / _generate_batch
# ──────────────────────────────────────────────────────────────────────────────

def _cb0_logits_from_tf_input(model, tf_input_embeds: mx.array, cb0_offset: int, T: int):
    """Forward a precomputed interleaved input-embedding sequence and slice the
    cb0 logits aligned to the sampled frames. Shared by the ref pass and the loss
    (so policy/reference run the identical forward, differing only in adapters)."""
    logits, _ = model.talker(tf_input_embeds)                 # [G, L, V]
    return logits[:, cb0_offset: cb0_offset + T, :]           # [G, T, V]


def sample_rollouts_interleaved(
    model,
    text:            str,                                # ONE prompt (string, not ids)
    *,
    lang_code:       str = "auto",
    group_size:      int = 4,
    max_new_tokens:  int = 240,
    temperature:     float = 0.9,
    top_p:           float = 0.95,
    top_k:           int = 50,
    ref_audio=None,                                      # Pipeline 2 voice cloning
    compute_ref:     bool = True,
    ref_params:      Optional[Dict[str, mx.array]] = None,
) -> Dict[str, mx.array]:
    """Sample `group_size` rollouts for ONE prompt using the REAL generate()
    interleaved layout, recording the exact input-embedding sequence for a
    consistent Phase-B forward.

    Returns dict:
        codec_ids       [G, T]            sampled cb0 (pad after EOS)
        codec_mask      [G, T]            valid frames
        full_codes      [G, T, 16]        all codebooks → decode to audio
        tf_input_embeds [G, P+T-1, D]     prefill ++ recorded frame inputs[:-1]
        cb0_offset      int               slice start so logits[:,off:off+T] ↔ codec_ids
        ref_logprobs    [G, T]            cb0 logp under reference (float32, no grad)
        gen_lengths     [G]
    """
    talker  = model.talker
    cfg     = talker.config
    eos_id  = cfg.codec_eos_token_id
    pad_id  = cfg.codec_pad_id
    n_groups = cfg.num_code_groups
    G = group_size

    suppress = [i for i in range(cfg.vocab_size - 1024, cfg.vocab_size) if i != eos_id]

    # Generation-path prefill (chat template, codec prefix, speaker token, first
    # text token) + the streamed trailing text. Single prompt → tile to G.
    prefill1, trailing1, tts_pad_embed = model._prepare_generation_inputs(
        text, language=lang_code, ref_audio=ref_audio,
    )
    D = prefill1.shape[-1]
    P = prefill1.shape[1]
    prefill   = mx.broadcast_to(prefill1, (G, P, D))
    trailing  = mx.broadcast_to(trailing1, (G, trailing1.shape[1], D))
    max_tr    = trailing.shape[1]
    pad_b     = mx.broadcast_to(tts_pad_embed, (G, 1, D))
    arange_g  = mx.arange(G)

    cache      = talker.make_cache()
    code_cache = talker.code_predictor.make_cache()
    finished   = mx.zeros((G,), dtype=mx.bool_)
    eos_fill   = mx.full((G, 1), eos_id, dtype=mx.int32)
    trail_idx  = mx.zeros((G,), dtype=mx.int32)

    cb0_steps:   List[mx.array] = []
    frame_steps: List[mx.array] = []
    rec_inputs:  List[mx.array] = []        # I_t = next input embed after frame t
    gen_lengths = [max_new_tokens] * G

    step_input = prefill
    for step in range(max_new_tokens):
        logits, hidden = talker(step_input, cache=cache)
        cb0 = model._sample_token_batch(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            suppress_tokens=suppress, eos_token_id=eos_id,
        )
        cb0 = mx.where(finished[:, None], eos_fill, cb0)
        newly = cb0[:, 0] == eos_id
        for b in range(G):
            if bool(newly[b]) and not bool(finished[b]):
                gen_lengths[b] = step + 1
        finished = finished | newly

        # frozen code_predictor: cb1..15 for this frame
        code_tokens = [cb0]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None; c.values = None; c.offset = 0
        for code_idx in range(n_groups - 1):
            if code_idx == 0:
                code_input = mx.concatenate(
                    [code_hidden, talker.get_input_embeddings()(cb0)], axis=1)
            else:
                code_input = talker.code_predictor.codec_embedding[code_idx - 1](
                    code_tokens[-1])
            code_logits, code_cache, _ = talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx)
            code_tokens.append(model._sample_token_batch(
                code_logits, temperature=temperature, top_k=top_k, top_p=top_p))
        frame = mx.concatenate(code_tokens, axis=1)           # [G, n_groups]

        cb0_steps.append(cb0); frame_steps.append(frame)

        # interleaved next input = streamed text + summed 16 codebook embeds
        clamped = mx.minimum(trail_idx, max_tr - 1)
        text_e  = trailing[arange_g, clamped, :][:, None, :]
        exhausted = clamped >= max_tr - 1
        text_e  = mx.where(exhausted[:, None, None], pad_b, text_e)
        trail_idx = trail_idx + (~finished).astype(mx.int32)
        codec_e = talker.get_input_embeddings()(cb0)
        for j, code in enumerate(code_tokens[1:]):
            codec_e = codec_e + talker.code_predictor.codec_embedding[j](code)
        next_input = text_e + codec_e                          # [G, 1, D]
        rec_inputs.append(next_input)
        step_input = next_input

        mx.eval(cb0, frame, finished, next_input)
        if bool(mx.all(finished)):
            break
        if step > 0 and step % 50 == 0:
            mx.clear_cache()

    T = len(cb0_steps)
    codec_ids  = mx.concatenate(cb0_steps, axis=1)             # [G, T]
    full_codes = mx.stack(frame_steps, axis=1)                # [G, T, n_groups]

    # Teacher-forcing input = prefill ++ recorded frame inputs I_0..I_{T-2}.
    # Forward over this reproduces the per-step logits (logit at prefill end → cb0[0],
    # logit at I_{t-1} → cb0[t]); cb0_offset = P-1 slices T logits ↔ codec_ids.
    if T > 1:
        rec = mx.concatenate(rec_inputs[: T - 1], axis=1)     # [G, T-1, D]
        tf_input = mx.concatenate([prefill, rec], axis=1)     # [G, P+T-1, D]
    else:
        tf_input = prefill
    cb0_offset = P - 1

    pos  = mx.arange(T)[None, :]
    glen = mx.array(gen_lengths)[:, None]
    codec_mask = pos < glen
    codec_ids  = mx.where(codec_mask, codec_ids, mx.array(pad_id, dtype=codec_ids.dtype))

    ref_logprobs = mx.zeros((G, T))
    if compute_ref:
        snapshot = ref_params if ref_params is not None else lora_snapshot(model)
        with lora_swapped(model, snapshot):
            ref_logits = _cb0_logits_from_tf_input(model, tf_input, cb0_offset, T)
            ref_logprobs = mx.stop_gradient(gather_token_logprobs(ref_logits, codec_ids))
        mx.eval(ref_logprobs)

    return {
        "codec_ids":       codec_ids,
        "codec_mask":      codec_mask,
        "full_codes":      full_codes,
        "tf_input_embeds": tf_input,
        "cb0_offset":      cb0_offset,
        "ref_logprobs":    ref_logprobs,
        "gen_lengths":     mx.array(gen_lengths),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Decode rollouts → audio (frozen speech_tokenizer), for the reward model
# ──────────────────────────────────────────────────────────────────────────────

def decode_codes_to_audio(
    model,
    full_codes: mx.array,        # [G, T, num_code_groups]
    codec_mask: mx.array,        # [G, T] bool
) -> List[mx.array]:
    """Decode each rollout's valid frames to a waveform. Returns list of [samples].

    Requires model.speech_tokenizer to be loaded (GRPO needs the codec decoder,
    unlike pre-tokenized SFT which can run without it).
    """
    if getattr(model, "speech_tokenizer", None) is None:
        raise RuntimeError(
            "GRPO rollout decode needs model.speech_tokenizer; load it before training "
            "(it is not required for pre-tokenized SFT)."
        )
    G = full_codes.shape[0]
    lengths = codec_mask.astype(mx.int32).sum(axis=1).tolist()
    audios: List[mx.array] = []
    for b in range(G):
        n = int(lengths[b])
        codes = full_codes[b: b + 1, :n, :]            # [1, n, num_code_groups]
        audio = model._decode_chunk(codes)
        mx.eval(audio)
        audios.append(audio)
    return audios
