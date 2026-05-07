# Indic Parler-TTS — Generation Optimisation Tracker

## Baseline numbers (before any work)
- Short sentence "आज कैसे हो?": **0.33s audio in 7.7s** (broken — loop ran to max_steps)
- Long sentence "मुझे कल सुबह...": **0.33s audio in 8.6s** (broken — early EOS from sampling bug)

## Current numbers (after all fixes below)
- Short sentence "आज कैसे हो?": **2.49s audio in 4.1s**
- Long sentence "मुझे कल सुबह...": **5.10s audio in 3.9s**
- Generation time now scales with sentence length instead of always running to max_steps.

---

## Done

### Sampling pipeline (model.py)

- **`apply_top_k` / `apply_top_p` from mlx_lm** — replaced custom implementations with the
  `@mx.compile`-decorated kernels from `mlx_lm.sample_utils`. Compiled kernels skip Python
  re-tracing on every step after the first call.

- **Bug fix: renormalise after top-k before top-p** — after `apply_top_k` masks non-top-k tokens
  to `-inf`, the surviving tokens' log-probs no longer sum to 1. `apply_top_p` uses `exp(logprobs)`
  and a cumulative sum; if total mass < `1 - top_p` (easily happens with diffuse distributions over
  1088 tokens), it masks *everything* to `-inf` and `mx.random.categorical` produces undefined
  output (was defaulting to token 0 or EOS → spurious early termination → 0.33s audio). Fixed by
  adding a second `logsumexp` renormalisation between `apply_top_k` and `apply_top_p`.

- **Sampler closure** (`_make_mlx_sampler`) — hyperparams and the `temps` array are captured once
  per `generate()` call as closure constants. `do_top_k` / `do_top_p` branches resolved at creation
  time so the hot path has no per-step Python branching.

- **Precomputed EOS suppression deltas** — `eos_deltas[num_cb+1, num_cb]` built once before the
  loop. `_suppress_eos` now does an O(1) MLX index instead of rebuilding a Python list each step.

- **Removed redundant `mx.eval` before `tolist()`** — both `mx.eval(tokens0)` and
  `mx.eval(tokens_mx)` were no-ops; `tolist()` already forces evaluation.

### Execution model (model.py)

- **Dedicated generation stream** — forward + sample inside `with mx.stream(generation_stream):`.
  Keeps generation in its own Metal command queue.

- **`wired_limit` context** — pins model weights in wired (fast) Apple Silicon memory for the
  encode + AR decode phase so the Metal driver can't page them mid-loop.

### Loop termination (model.py)

- **Repetition detection** — if all 9 codebooks produce the *exact* same token vector for 20
  consecutive steps (checked after the 9-step delay warm-up), the model has looped and generation
  stops. Root cause: for short sentences the model finishes speech content in ~30 steps then gets
  stuck on a fixed vector (e.g. `[568, 804, 10, ...]`) that decodes to near-silence. Without this
  fix the loop ran all 869 steps (7.7s) regardless of sentence length; now it exits in ~30–50
  steps for short inputs. Long sentences still exit via EOS as before.

---

## TODO — remaining optimisations

### Tier 1 — highest impact, moderate complexity

#### 1. `mx.async_eval` pipeline (double-buffering)

**Expected gain: 1.5–2.5× step throughput**

mlx_lm's `generate_step` fires `mx.async_eval(next_y)` *before* syncing the current step's result,
so the GPU is computing step N+1 while Python is processing step N's tokens. Currently every step
is sequential: forward → sample → `tolist()` (GPU→CPU sync) → Python update → build input → repeat.

**How to implement:**
1. Precompute per-step delay masks as a stacked MLX bool array `[max_steps, num_cb]` (see item 4).
2. Build `tokens_this_step` from the previous `tokens_mx` lazily using `mx.where(delay_mask, ...)`.
3. Queue the next forward + sample without waiting for current step's `tolist()`.
4. Fire `mx.async_eval(tokens_mx, next_tokens_mx)` — GPU runs both in background.
5. Call `tokens_mx.tolist()` — sync happens here but GPU may already be done.

**Blocker:** `_suppress_eos` uses `first_unfinished` which is updated by the Python EOS check each
step. Two clean escape routes:
- Drop per-step EOS suppression entirely — the staggered drain (`step >= target_T + k`) already
  guarantees clean termination at max duration. EOS suppression is a quality aid, not correctness.
- Apply EOS as a post-hoc clamp on `tokens_list` after `tolist()` instead of pre-suppressing.

---

#### 2. Replace Python list `generated` with preallocated MLX tensor

**Expected gain: reduces per-step Python overhead; prerequisite for item 1**

`generated = [[bos] * (max_steps + 2) for _ in range(num_cb)]` requires a Python list comprehension
to build `tokens_this_step` (reads from the list) and a Python loop to write `tokens_list` back.
Keeping the buffer as an MLX tensor allows lazy indexing so the GPU never stalls on Python.

**How:** `generated_mx = mx.full((num_cb, max_steps + 2), bos, dtype=mx.int32)`; index with
`generated_mx[:, step]` and write with `.at[:, step+1].set(tokens_mx)`.

**Remaining blocker:** EOS check (`generated[first_unfinished][step+1] == eos`) still needs a
Python value, so one `tolist()` per step is unavoidable until the EOS check is restructured.

---

#### 3. Generator / streaming refactor

**Expected gain: first audio chunk in < 1s, enables real-time use**

Current API blocks until the full utterance is ready. A generator variant decodes partial frames
through DAC every N steps and yields chunks immediately.

**Interface sketch:**
```python
def stream_generate(self, ...) -> Generator[np.ndarray, None, None]:
    for step in range(1, max_steps):
        ...
        if step % chunk_steps == 0 and step > num_cb:
            partial = extract_valid_frames(generated, up_to=step)
            if partial:
                yield np.array(self.dac.decode(mx.array(partial)))
```
Chunk size of 50 steps ≈ 600 ms of audio. Use `mlx_audio.tts.models.base.GenerationResult`
dataclass for the yielded type (matches the mlx-audio ecosystem).

---

### Tier 2 — quality + secondary speed

#### 4. Tensorise delay-pattern input construction

**Expected gain: removes per-step Python list comprehension; prerequisite for item 1**

The BOS fill `step > k else bos` is currently a Python conditional rebuilt every step. Precompute
the full delay mask matrix once before the loop:

```python
delay_masks = mx.array(
    [[step > k for k in range(num_cb)] for step in range(max_steps)],
    dtype=mx.bool_,
)
bos_fill = mx.full((1, num_cb), bos, dtype=mx.int32)
# inside loop (tokens_prev_mx is the previous step's sampled tensor):
tokens_this_step = mx.where(delay_masks[step], tokens_prev_mx[None], bos_fill)
```

This makes the step input construction a single MLX op on resident tensors with no Python loop.

---

#### 5. EOS biasing after minimum duration

**Expected gain: reduces over-generation tail on long sentences**

After a minimum duration, add a progressively increasing positive bias to the EOS logit for CB0.
Steers the model toward termination without hard-forcing it.

```python
min_eos_step = int(0.5 * 44100 / 512)   # 0.5s warm-up
eos_bias_rate = 0.05                      # per step
eos_bias_max  = 2.0
if step > min_eos_step:
    bias = min(eos_bias_max, eos_bias_rate * (step - min_eos_step))
    logits = logits.at[0, eos].add(bias)
```
Apply before `_suppress_eos`. Tune `eos_bias_rate` to taste — too high causes clipping.

---

#### 6. Repetition penalty on codec tokens

**Expected gain: better audio quality, fewer stuck loops**

Codec LMs are prone to texture repetition. `mlx_lm.sample_utils.make_repetition_penalty` is
already compiled and can be applied per-codebook on a sliding window of recent CB0 tokens (the
coarse codebook drives rhythm; penalising it is enough).

```python
from mlx_lm.sample_utils import make_repetition_penalty
rep_penalty = make_repetition_penalty(penalty=1.3, context_size=20)
# inside loop, before sampler:
logits = rep_penalty(recent_cb0_tokens, logits)
```

Note: the current repetition *detection* (stops the loop) handles the degenerate case, but
a repetition *penalty* (biases against repeating) would prevent it reaching the threshold and
produce smoother output near sentence boundaries.

---

### Tier 3 — architecture / systems

#### 7. KV cache as a typed object (mlx_lm KVCache)

Current cache is `[[] for _ in range(num_layers)]` — Python lists grown by appending tensors.
mlx_lm's `KVCache` class stores K/V in contiguous preallocated arrays with `offset`-tracked
appends. Benefits: lower allocator pressure, better Metal memory locality, supports serialisation.

**Blocker:** Requires changes in `decoder.py` to accept the mlx_lm cache interface. Do this after
items 1–3 are landed (biggest-bang items first).

---

### Tier 4 — NumPy cleanup (low effort, minor gains)

#### 8. Remove residual NumPy from post-loop code

Three one-liners in `model.py` that are trivially replaceable:

| Line | Current | Fix |
|------|---------|-----|
| 461 | `return np.zeros(0, dtype=np.float32)` | `return np.array(mx.zeros(0))` or change return type |
| 463 | `codes = np.array(frames, dtype=np.int32).T` | `codes = mx.array(frames, dtype=mx.int32).T` — stay MLX into DAC decode, skip the CPU round-trip |
| 469 | `return np.array(audio, dtype=np.float32)` | Return `mx.array` directly; move the numpy conversion to the caller (`generate.py`) at the soundfile write boundary |

No speed impact inside the AR loop (all post-loop), but cleans up the data flow and makes the
return type consistent if streaming is added later.

---

### Tier 5 — HuggingFace parity

These match features present in the original `ai4bharat/indic-parler-tts` HuggingFace
implementation that are not yet in the MLX port.

#### 9. Batch generation (batch_size > 1)

Current code hardcodes `[1, ...]` shapes everywhere (encoder, prompt embed, KV caches, logits).
The HF implementation supports batched generation with left-padded inputs and an attention mask.

**What's needed:**
- Attention mask on the decoder self-attention during prefill
- KV cache that tracks per-sample offsets (mlx_lm `BatchKVCache` handles this)
- Sampler operating on `[B, num_cb, vocab]` logits
- EOS tracking per sample in the batch

**Benefit:** generate N utterances in roughly the same wall-clock time as 1.

---

#### 10. Attention mask for variable-length prompt inputs

Current decoder steps pass `mask=None` after prefill. For batch_size=1 this is fine (no padding).
For batch > 1 or very long prompts, missing the cross-attention mask causes the decoder to attend
to padding tokens in the T5 encoder output, degrading quality.

**What's needed:** Pass `encoder_attention_mask` (from the T5 tokenizer's output) through to
the cross-attention layers in `decoder.py`. This is a 2–3 line change in `forward_layers` and
the cross-attention call.

---

#### 11. `ParlerTTSLogitsProcessor` full parity

The HF implementation uses `ParlerTTSLogitsProcessor` which:
1. Suppresses EOS for codebooks not yet unlocked (we have this)
2. **Suppresses all non-EOS tokens once a codebook has EOS'd** — once CB k generates EOS, future
   steps for CB k should be forced to EOS too. We don't do this; we just let the staggered drain
   handle it via `step >= target_T + k`.

**What's needed:** After `first_unfinished` advances past codebook k, force `generated[k][step+1]`
to EOS regardless of what the sampler returns. This prevents "re-entry" into non-EOS tokens for
finished codebooks and keeps the delay pattern clean.

---

#### 12. Classifier-free guidance (CFG)

Some Parler TTS variants (and Dia, which uses the same architecture) support CFG by running two
forward passes per step — one conditioned, one unconditioned — and interpolating:
`logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)`.

The unconditioned pass uses a zeroed description embedding. Not implemented in the MLX port.
Optional feature; only relevant if the model was trained with CFG.

---

## Recommended implementation order for next session

```
1. Tensorise delay-pattern (item 4) — small, self-contained, unblocks item 1
2. mx.async_eval pipeline (item 1) — biggest throughput win
3. Preallocate generated tensor (item 2) — pairs with item 1
4. Streaming generator (item 3) — UX win, requires items 1–2 for best latency
5. Repetition penalty (item 6) — quality improvement over the current hard-stop heuristic
6. EOS biasing (item 5) — refinement, tune after 1–4 are stable
7. HF parity: force-EOS finished codebooks (item 11) — correctness fix, small
8. HF parity: cross-attention mask (item 10) — correctness fix, 3 lines
9. NumPy cleanup (item 8) — cosmetic, do anytime
10. KV cache refactor (item 7) — last, highest complexity
11. Batch generation (item 9) — requires item 7 + 10
```

---

## Key files

| File | Role |
|------|------|
| `models/indic_parler_tts/model.py` | Main model: weight loading, `generate()`, `_make_mlx_sampler` |
| `models/indic_parler_tts/decoder.py` | ParlerDecoder: `forward_layers`, `embed_audio`, KV cache |
| `models/indic_parler_tts/generate.py` | Public API: `load_model()`, `generate()`, `_trim_silence` |
| `models/indic_parler_tts/config.py` | Config: `eos_token_id=1024`, `bos_token_id=1025`, `codebook_size=1024` |

## Reference implementations

| Source | What to borrow |
|--------|---------------|
| `mlx_lm.sample_utils` | `apply_top_k`, `apply_top_p`, `make_repetition_penalty` (all `@mx.compile`) |
| `mlx_lm.generate` | `generate_step` async pipeline, `wired_limit`, `generation_stream` |
| `mlx_audio.tts.models.dia.dia` | Multi-codebook loop structure, `GenerationResult` yield pattern |
