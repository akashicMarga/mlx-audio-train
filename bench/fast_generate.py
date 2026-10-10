"""
A copy of mlx-audio 0.4.4's Qwen3-TTS base-model decode loop
(`Model.generate` → base path, non-ICL), with optional switches for the loop
levers. With every switch off it must be bit-identical to stock — that is the
first thing tests/benchmarks check (lever `loop-copy`).

Switches:
  device_rep_penalty  keep a fixed-shape [vocab] "seen" mask on the device and
                      apply the repetition penalty with mx.where, instead of
                      building a Python set of token ids every frame. Removes
                      the per-frame host round trip the penalty needs, and the
                      shape change that would force mx.compile to retrace.
  compile_code_pred   run the whole 15-pass code-predictor frame (embeddings,
                      5 layers × 15 passes, heads, sampling, embedding sum) as
                      one mx.compile'd function. The stock modules are reused;
                      their KV cache is replaced inside the trace by a pure
                      concatenating cache (same keys/values as KVCache returns).
                      Random state is a compile input/output, so sampling keys
                      advance exactly as in stock. Fusion may change rounding,
                      so this is not expected to be bit-exact.
  compile_talker      decode-step Talker forward as one mx.compile'd call on fixed
                      shapes: after a stock prefill, each layer's K/V move into a
                      fixed-capacity buffer (prompt + max_tokens, rounded to 64);
                      every step writes its slot with slice_update and attends
                      under a boolean mask over unfilled slots. Position ids and
                      mask are passed explicitly, so one trace serves the whole
                      utterance. Masked-slot attention may change rounding.
  async_eval          pipeline frames: build frame t+1's graph while frame t
                      evaluates (mx.async_eval) and check frame t's EOS one
                      frame late. Requires device_rep_penalty — with the host
                      penalty, sampling t+1 needs token t on the host, so there
                      is nothing to overlap.

Only the base path is reimplemented. ICL voice cloning (ref_audio + ref_text),
voice_design and custom_voice models fall through to the stock method.
Upstream source: mlx_audio/tts/models/qwen3_tts/qwen3_tts.py (MIT).
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx
from mlx_lm.sample_utils import apply_top_k, categorical_sampling

from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.qwen3_tts import qwen3_tts as q3

UPSTREAM_VERSION = "0.4.4"


@dataclass
class LoopOptions:
    device_rep_penalty: bool = False
    async_eval: bool = False
    compile_code_pred: bool = False
    compile_talker: bool = False
    cp_sampler: str = "single"        # "single" (stock) | "fast" (top-p over the top-k survivors)

    def __post_init__(self):
        if self.async_eval and not self.device_rep_penalty:
            raise ValueError("async_eval needs device_rep_penalty (host penalty forces a sync per frame)")
        if self.compile_talker and not self.device_rep_penalty:
            raise ValueError("compile_talker needs device_rep_penalty (host penalty forces a sync per frame)")


def _sample_cb0_device(logits, *, temperature, top_k, top_p, repetition_penalty,
                       seen, suppress_idx, eos_token_id, min_p=0.0):
    """Same arithmetic as Model._sample_token, penalty applied from a device mask."""
    logits = logits[:, -1, :]
    logits = mx.put_along_axis(logits, suppress_idx[None, :],
                               mx.array(float("-inf"), logits.dtype), axis=-1)
    if seen is not None and repetition_penalty != 1.0:
        penalized = mx.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        logits = mx.where(seen, penalized, logits)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    if temperature != 1.0:
        logits = logits / temperature
    eos_logit = logits[:, eos_token_id: eos_token_id + 1]
    if top_k > 0 and top_k < logits.shape[-1]:
        logits = apply_top_k(logits, top_k)
    logits = q3._apply_probability_filters(logits, top_p, min_p)
    logits = mx.put_along_axis(logits, mx.array([[eos_token_id]], dtype=mx.int32), eos_logit, axis=-1)
    return categorical_sampling(logits, 1.0)[:, None]


def _sample_topk_topp(logits, *, temperature, top_k, top_p, **_ignored):
    """Code-predictor sampler (no penalty / suppression / EOS handling, as stock uses
    for codebooks 1–15) that applies top-p over the top_k survivors only.

    Stock: apply_top_k masks to k survivors, then log_softmax and argsort over the
    whole vocabulary for top-p. Here: gather the k survivors (same argpartition as
    apply_top_k), log_softmax + sort over k values, then scatter the kept logits back
    into a full-vocab -inf array and sample with the same categorical call — so the
    random draw has the same shape and the result matches stock up to rounding in
    the softmax normaliser at the top-p threshold."""
    logits = logits[:, -1, :]
    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    if temperature != 1.0:
        logits = logits / temperature
    v = logits.shape[-1]
    if not (0 < top_k < v):
        top_k = v
    idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]      # [B, k]
    sub = mx.take_along_axis(logits, idx, axis=-1)                            # [B, k]
    keep_sub = mx.ones(sub.shape, dtype=mx.bool_)
    if 0.0 < top_p < 1.0:
        logprobs = sub - mx.logsumexp(sub, axis=-1, keepdims=True)
        order = mx.argsort(logprobs, axis=-1)                                 # ascending, as stock
        sorted_p = mx.exp(mx.take_along_axis(logprobs, order, axis=-1))
        cum = mx.cumsum(sorted_p, axis=-1)
        keep_sorted = cum > 1 - top_p
        inv = mx.argsort(order, axis=-1)
        keep_sub = mx.take_along_axis(keep_sorted, inv, axis=-1)
    kept = mx.where(keep_sub, sub, mx.array(-float("inf"), sub.dtype))
    full = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    full = mx.put_along_axis(full, idx, kept, axis=-1)
    return categorical_sampling(full, 1.0)[:, None]


class _ConcatCache:
    """Trace-time stand-in for mlx_lm KVCache: concatenates, never mutates arrays."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k, v):
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=2)
        self.values = v if self.values is None else mx.concatenate([self.values, v], axis=2)
        self.offset += k.shape[2]
        return self.keys, self.values


class _SlotCache:
    """Trace-time cache over a fixed-capacity buffer: writes slot `pos`, returns the whole buffer."""

    def __init__(self, keys, values, pos):
        self.keys, self.values, self.pos = keys, values, pos
        self.offset = 0          # read by the Talker model; unused because position_ids are passed

    def update_and_fetch(self, k, v):
        self.keys = mx.slice_update(self.keys, k, self.pos, axes=(2,))
        self.values = mx.slice_update(self.values, v, self.pos, axes=(2,))
        return self.keys, self.values


def _talker_step_fn(model):
    """Compiled: (x [1,1,H], pos int32 [1], keys[L], values[L]) -> (logits, hidden, keys, values)."""
    fn = model.__dict__.get("_bench_talker_step")
    if fn is not None:
        return fn
    talker = model.talker

    @partial(mx.compile, inputs=[talker.state])
    def step(x, pos, keys, values):
        cap = keys[0].shape[2]
        caches = [_SlotCache(k, v, pos) for k, v in zip(keys, values)]
        position_ids = mx.broadcast_to(pos.reshape(1, 1, 1), (3, 1, 1)).astype(mx.int32)
        mask = (mx.arange(cap) <= pos)[None, None, None, :]
        logits, hidden = talker(x, position_ids=position_ids, mask=mask, cache=caches)
        return logits, hidden, [c.keys for c in caches], [c.values for c in caches]

    model.__dict__["_bench_talker_step"] = step
    return step


def _to_slot_buffers(cache, capacity):
    """Copy a prefilled mlx_lm KVCache list into fixed-capacity buffers."""
    keys, values = [], []
    for c in cache:
        k = c.keys[..., : c.offset, :]
        v = c.values[..., : c.offset, :]
        b, h, _, d = k.shape
        K = mx.zeros((b, h, capacity, d), dtype=k.dtype)
        V = mx.zeros((b, h, capacity, d), dtype=v.dtype)
        keys.append(mx.slice_update(K, k, mx.array([0]), axes=(2,)))
        values.append(mx.slice_update(V, v, mx.array([0]), axes=(2,)))
    return keys, values


def _code_pred_frame_fn(model, temperature, top_k, top_p, sampler="single"):
    """Compiled: (code_hidden [1,1,H], cb0 [1,1]) -> (codes 1..15 [1,15], embed sum [1,1,H]).

    The embed sum is talker_emb(cb0) + Σ cp_emb[i](code_i): the codec part of the
    next Talker input, exactly as the stock loop builds it."""
    cache_key = (temperature, top_k, top_p, sampler)
    sample = {"single": model._sample_token, "batch": model._sample_token_batch,
              "fast": _sample_topk_topp}[sampler]
    fns = model.__dict__.setdefault("_bench_cp_fns", {})
    if cache_key in fns:
        return fns[cache_key]
    talker = model.talker
    cp = talker.code_predictor
    n_groups = model.config.talker_config.num_code_groups
    state = [mx.random.state, talker.state]

    @partial(mx.compile, inputs=state, outputs=[mx.random.state])
    def frame(code_hidden, cb0):
        caches = [_ConcatCache() for _ in cp.model.layers]
        tokens = [cb0]
        emb_sum = talker.get_input_embeddings()(cb0)
        for code_idx in range(n_groups - 1):
            if code_idx == 0:
                code_input = mx.concatenate([code_hidden, talker.get_input_embeddings()(cb0)], axis=1)
            else:
                code_input = cp.codec_embedding[code_idx - 1](tokens[-1])
            logits, _, _ = cp(code_input, cache=caches, generation_step=code_idx)
            tok = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            tokens.append(tok)
            emb_sum = emb_sum + cp.codec_embedding[code_idx](tok)
        return mx.concatenate(tokens[1:], axis=1), emb_sum

    fns[cache_key] = frame
    return frame


def _result(model, audio, segment_idx, token_count, elapsed, **kw):
    dur = audio.shape[0] / model.sample_rate
    return GenerationResult(
        audio=audio, samples=audio.shape[0], sample_rate=model.sample_rate,
        segment_idx=segment_idx, token_count=token_count,
        audio_duration=q3.format_duration(dur),
        real_time_factor=dur / elapsed if elapsed > 0 else 0,
        prompt={"tokens": token_count, "tokens-per-sec": token_count / elapsed if elapsed > 0 else 0},
        audio_samples={"samples": audio.shape[0],
                       "samples-per-sec": audio.shape[0] / elapsed if elapsed > 0 else 0},
        processing_time_seconds=elapsed, peak_memory_usage=mx.get_peak_memory() / 1e9, **kw)


def generate(model, text, opts: LoopOptions, *, voice=None, instruct=None, temperature=0.9,
             lang_code="auto", ref_audio=None, ref_text=None, split_pattern="\n",
             max_tokens=4096, stream=False, streaming_interval=2.0, top_k=50, top_p=1.0,
             repetition_penalty=1.05, verbose=False, **kwargs):
    tts_type = getattr(model.config, "tts_model_type", "base")
    use_icl = ref_audio is not None and ref_text is not None and model.speech_tokenizer.has_encoder
    if tts_type != "base" or use_icl:
        yield from type(model).generate(
            model, text, voice=voice, instruct=instruct, temperature=temperature,
            lang_code=lang_code, ref_audio=ref_audio, ref_text=ref_text,
            split_pattern=split_pattern, max_tokens=max_tokens, stream=stream,
            streaming_interval=streaming_interval, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, verbose=verbose, **kwargs)
        return
    if ref_audio is not None:
        ref_audio = q3.load_audio(ref_audio, sample_rate=model.sample_rate)

    segments = ([s.strip() for s in text.split(split_pattern) if s.strip()]
                if split_pattern else [text])
    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [i for i in range(config.vocab_size - 1024, config.vocab_size)
                       if i != eos_token_id]
    suppress_idx = mx.array(suppress_tokens, dtype=mx.int32)
    vocab_ids = mx.arange(config.vocab_size)[None, :]

    for segment_idx, segment_text in enumerate(segments):
        start_time = time.time()
        input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
            segment_text, language=lang_code, speaker=voice, ref_audio=ref_audio, ref_text=ref_text)
        cache = model.talker.make_cache()
        code_cache = model.talker.code_predictor.make_cache()
        state = {"trailing_idx": 0, "seen": None, "ids": [], "kv": None, "pos": None}
        capacity = -(-(input_embeds.shape[1] + max_tokens + 2) // 64) * 64

        def talker_forward(input_embeds):
            timers = getattr(model, "_bench_talker_timer", None)   # (prefill, step) lists when instrumented
            if not opts.compile_talker or state["kv"] is None:   # stock path, or the prefill
                if timers is None:
                    out = model.talker(input_embeds, cache=cache)
                else:
                    mx.eval(input_embeds)
                    t0 = time.perf_counter()
                    out = model.talker(input_embeds, cache=cache)
                    mx.eval(*out)
                    (timers[0] if input_embeds.shape[1] > 1 else timers[1]).append(time.perf_counter() - t0)
                if opts.compile_talker:          # prefill done: move K/V into slot buffers
                    state["kv"] = _to_slot_buffers(cache, capacity)
                    state["pos"] = cache[0].offset
                return out
            step = _talker_step_fn(model)
            pos = mx.array([state["pos"]], dtype=mx.int32)
            if timers is not None:
                mx.eval(input_embeds)
                t0 = time.perf_counter()
            logits, hidden, keys, values = step(input_embeds, pos, *state["kv"])
            if timers is not None:
                mx.eval(logits, hidden, keys, values)
                timers[1].append(time.perf_counter() - t0)
            state["kv"] = (keys, values)
            state["pos"] += 1
            return logits, hidden

        def build_frame(input_embeds):
            """Graph for one frame; returns (next_input_embeds, is_eos, all_codes, next_token)."""
            logits, hidden = talker_forward(input_embeds)
            if opts.device_rep_penalty:
                next_token = _sample_cb0_device(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, seen=state["seen"],
                    suppress_idx=suppress_idx, eos_token_id=eos_token_id)
                hit = vocab_ids == next_token
                state["seen"] = hit if state["seen"] is None else (state["seen"] | hit)
            else:
                next_token = model._sample_token(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=state["ids"] if state["ids"] else None,
                    suppress_tokens=suppress_tokens, eos_token_id=eos_token_id)
            is_eos = next_token[0, 0] == eos_token_id

            if opts.compile_code_pred:
                cp_frame = _code_pred_frame_fn(model, temperature, top_k, top_p, sampler=opts.cp_sampler)
                timer = getattr(model, "_bench_cp_timer", None)
                if timer is None:
                    rest, codec_embed = cp_frame(hidden[:, -1:, :], next_token)
                else:                      # instrumented pass: time the compiled frame as one unit
                    ch = hidden[:, -1:, :]
                    mx.eval(ch, next_token)
                    t0 = time.perf_counter()
                    rest, codec_embed = cp_frame(ch, next_token)
                    mx.eval(rest, codec_embed)
                    timer.append(time.perf_counter() - t0)
                all_codes = mx.concatenate([next_token, rest], axis=1)
                ti = state["trailing_idx"]
                if ti < trailing_text_hidden.shape[1]:
                    text_embed = trailing_text_hidden[:, ti: ti + 1, :]
                    state["trailing_idx"] = ti + 1
                else:
                    text_embed = tts_pad_embed
                return text_embed + codec_embed, is_eos, all_codes, next_token

            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]
            for c in code_cache:
                c.keys = None
                c.values = None
                c.offset = 0
            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = model.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_input = model.talker.code_predictor.codec_embedding[code_idx - 1](code_tokens[-1])
                code_logits, _, _ = model.talker.code_predictor(
                    code_input, cache=code_cache, generation_step=code_idx)
                code_tokens.append(model._sample_token(code_logits, temperature=temperature,
                                                       top_k=top_k, top_p=top_p))
            all_codes = mx.concatenate(code_tokens, axis=1)

            ti = state["trailing_idx"]
            if ti < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, ti: ti + 1, :]
                state["trailing_idx"] = ti + 1
            else:
                text_embed = tts_pad_embed
            codec_embed = model.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[i](code)
            return text_embed + codec_embed, is_eos, all_codes, next_token

        generated_codes = []
        if stream:
            chunk_size = max(1, int(streaming_interval * 12.5))
            decoded = 0
            chunk_start = time.time()
            model.speech_tokenizer.decoder.reset_streaming_state()

        def emit_stream_chunk(final):
            nonlocal decoded, chunk_start
            n_new = len(generated_codes) - decoded
            codes = mx.transpose(mx.stack(generated_codes[decoded:], axis=1), (0, 2, 1))
            mx.eval(codes)
            audio_chunk = model.speech_tokenizer.decoder.streaming_step(codes).squeeze(1)[0]
            mx.eval(audio_chunk)
            res = _result(model, audio_chunk, segment_idx, n_new, time.time() - chunk_start,
                          is_streaming_chunk=True, is_final_chunk=final)
            decoded = len(generated_codes)
            chunk_start = time.time()
            mx.clear_cache()
            return res

        if opts.async_eval:
            nxt = build_frame(input_embeds)
            mx.async_eval(*nxt)
        for step in range(max_tokens):
            if opts.async_eval:
                cur = nxt
                if step + 1 < max_tokens:
                    nxt = build_frame(cur[0])      # graph for t+1 while t evaluates
                    mx.async_eval(*nxt)
                input_embeds, is_eos, all_codes, next_token = cur
                if is_eos.item():
                    break
            else:
                input_embeds, is_eos, all_codes, next_token = build_frame(input_embeds)
                mx.eval(input_embeds, is_eos)          # stock: single sync point per frame
                if is_eos.item():
                    break
                if not opts.device_rep_penalty:
                    state["ids"].append(int(next_token[0, 0]))
            generated_codes.append(all_codes)
            if step > 0 and step % 50 == 0:
                mx.clear_cache()
            if stream and len(generated_codes) - decoded >= chunk_size:
                yield emit_stream_chunk(final=False)

        if stream:
            if len(generated_codes) > decoded:
                yield emit_stream_chunk(final=True)
            model.speech_tokenizer.decoder.reset_streaming_state()
            continue
        if not generated_codes:
            continue
        codes = mx.stack(generated_codes, axis=1)
        audio, audio_lengths = model.speech_tokenizer.decode(codes)
        audio = audio[0]
        valid_len = int(audio_lengths[0])
        if 0 < valid_len < audio.shape[0]:
            audio = audio[:valid_len]
        mx.eval(audio)
        yield _result(model, audio, segment_idx, len(generated_codes), time.time() - start_time)
        mx.clear_cache()


def install(model, opts: LoopOptions):
    """Route model.generate through this loop (instance attribute; class untouched)."""
    def _gen(text, **kw):
        return generate(model, text, opts, **kw)
    model.generate = _gen
    model._bench_loop = opts
    return model


# ── GRPO-shaped group generation (Part 4, §7) ────────────────────────────────
def _sample_cb0_rows(logits, *, temperature, top_k, top_p, repetition_penalty, seen,
                     suppress_idx, eos_token_id, min_p=0.0):
    """_sample_cb0_device for B rows: per-row presence masks, same arithmetic."""
    logits = logits[:, -1, :]
    b = logits.shape[0]
    logits = mx.put_along_axis(logits, mx.broadcast_to(suppress_idx[None, :], (b, suppress_idx.shape[0])),
                               mx.array(float("-inf"), logits.dtype), axis=-1)
    if seen is not None and repetition_penalty != 1.0:
        penalized = mx.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        logits = mx.where(seen, penalized, logits)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    if temperature != 1.0:
        logits = logits / temperature
    eos_logit = logits[:, eos_token_id: eos_token_id + 1]
    if top_k > 0 and top_k < logits.shape[-1]:
        logits = apply_top_k(logits, top_k)
    logits = q3._apply_probability_filters(logits, top_p, min_p)
    logits = mx.put_along_axis(logits, mx.full((b, 1), eos_token_id, dtype=mx.int32), eos_logit, axis=-1)
    return categorical_sampling(logits, 1.0)[:, None]


def generate_group(model, text, group_size, *, lang_code="auto", temperature=0.9, top_k=50,
                   top_p=1.0, repetition_penalty=1.05, max_tokens=240, compile_code_pred=True, **_ignored):
    """G rollouts of one prompt in one batch: plain KVCache with batch dim G (identical
    prompts need no padding, mask or cache merging), per-row device repetition penalty,
    batched (optionally compiled) code-predictor frame, every row run until all rows hit
    EOS, each row trimmed at its own EOS and decoded with the stock decoder.
    Returns a list of G waveforms (mx.array) and the per-row frame counts."""
    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_idx = mx.array([i for i in range(config.vocab_size - 1024, config.vocab_size)
                             if i != eos_token_id], dtype=mx.int32)
    vocab_ids = mx.arange(config.vocab_size)[None, :]
    g = group_size

    input_embeds, trailing_text_hidden, tts_pad_embed = model._prepare_generation_inputs(
        text, language=lang_code, speaker=None, ref_audio=None, ref_text=None)
    input_embeds = mx.broadcast_to(input_embeds, (g,) + input_embeds.shape[1:])
    cache = model.talker.make_cache()
    cp_frame = _code_pred_frame_fn(model, temperature, top_k, top_p) if compile_code_pred else None
    code_cache = None if compile_code_pred else model.talker.code_predictor.make_cache()

    seen = None
    codes, ends = [], [None] * g
    trailing_idx = 0
    for step in range(max_tokens):
        logits, hidden = model.talker(input_embeds, cache=cache)
        tok = _sample_cb0_rows(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                               repetition_penalty=repetition_penalty, seen=seen,
                               suppress_idx=suppress_idx, eos_token_id=eos_token_id)
        hit = vocab_ids == tok
        seen = hit if seen is None else (seen | hit)
        if cp_frame is not None:
            rest, codec_embed = cp_frame(hidden[:, -1:, :], tok)
            all_codes = mx.concatenate([tok, rest], axis=1)
        else:
            code_tokens, all_codes = model._predict_code_tokens(
                tok, hidden, temperature=temperature, top_k=top_k, top_p=top_p, code_cache=code_cache)
            codec_embed = model._codec_embeds_for_tokens(code_tokens)
        if trailing_idx < trailing_text_hidden.shape[1]:
            text_embed = trailing_text_hidden[:, trailing_idx: trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed
        input_embeds = text_embed + codec_embed
        is_eos = tok[:, 0] == eos_token_id
        mx.eval(input_embeds, is_eos, all_codes)
        eos_rows = is_eos.tolist()
        for r in range(g):
            if ends[r] is None and eos_rows[r]:
                ends[r] = step
        codes.append(all_codes)
        if all(e is not None for e in ends):
            break
        if step > 0 and step % 50 == 0:
            mx.clear_cache()
    ends = [e if e is not None else len(codes) for e in ends]
    stacked = mx.stack(codes, axis=1)              # [G, T, 16]
    wavs = []
    for r in range(g):
        n = ends[r]
        if n == 0:
            wavs.append(mx.zeros((0,), dtype=mx.float32))
            continue
        audio, lengths = model.speech_tokenizer.decode(stacked[r: r + 1, :n, :])
        audio = audio[0]
        valid = int(lengths[0])
        if 0 < valid < audio.shape[0]:
            audio = audio[:valid]
        mx.eval(audio)
        wavs.append(audio)
    return wavs, ends
