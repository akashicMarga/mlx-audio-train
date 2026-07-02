# Pipeline 3 (proposal): GRPO Post-Training for TTS

Design sketch for an RL refinement stage that runs *after* LoRA SFT
(Pipeline 1 or 2). Inspired by Kyutai's interactivity-alignment work
([arXiv 2606.11167](https://hf.co/papers/2606.11167)), which post-trains
full-duplex speech models with GRPO using axis-specific rewards. Their
problem (turn-taking, interruptions) doesn't apply to TTS, but the
structural insight does: **token-level cross-entropy never directly
optimizes the things we actually evaluate** — intelligibility (WER/CER)
for language adaptation, speaker similarity for voice cloning. GRPO lets
us optimize those metrics directly, with no reward model and no value
network.

**Status: implemented** (`train/grpo/`, `train/losses/grpo_loss.py`,
`configs/qwen3_tts_hindi_grpo.yaml`; run via `scripts/train.py` with
`pipeline: grpo`). Use this when SFT plateaus (e.g. the Hindi adapter is fluent
but WER is stuck, or speaker similarity stalls). Not worth the rollout cost
before that.

> **Note on this doc vs. the code.** A few details below were revised during
> implementation; the corrections are called out inline with **[impl]**. The
> two that matter most: (1) the policy is **codebook 0 only** — SFT supervises
> only cb0 (`processors/qwen3_tts.py` stores `codes[0,0]`); cb1–15 come from the
> frozen `code_predictor` at decode time. (2) The KL **reference is a frozen
> snapshot of the SFT adapter** (`lora_snapshot`/`lora_swapped`), *not* a
> LoRA-disable toggle — see the KL section for why the toggle is incorrect.

---

## Why GRPO fits this stack

- The Qwen3-TTS talker is an autoregressive codec-token LM — sampling K
  rollouts per prompt is exactly the GRPO setup (same as text LLMs).
- GRPO is critic-free: advantages are computed *within a group* of
  rollouts for the same prompt, so no value network and no extra memory.
- We only update LoRA params (reuse `get_trainable_params` from
  `train/lora.py`), so the optimizer/grad path is identical to the
  existing `Trainer` — including the `_strip_empty` workaround for the
  frozen `speech_tokenizer`.
- A frozen reference policy comes from a one-time **snapshot of the SFT
  adapter** at stage start. **[impl]** The original idea here — "run with LoRA
  deltas disabled" — is *wrong* on two counts: disabling anchors to the *base*
  model (fighting the SFT adaptation), and the disabled path runs bfloat16 while
  the enabled path upcasts to float32, so the two diverge ~3 nats over 28 layers
  and the k3 KL explodes. The snapshot runs the *identical* float32 forward,
  differing only in adapter weights → KL is exactly 0 at stage start.

## Training step anatomy

Each GRPO step has two phases:

```
Phase A — rollout (no grad):
  for each of B prompts:
    sample G codec sequences from the policy (temperature ~0.9)
  decode codec → waveform (frozen speech_tokenizer code2wav)
  score each rollout → scalar reward
  advantage A_i = (r_i − mean_group(r)) / (std_group(r) + 1e-4)

Phase B — update (grad, reuses SFT forward path):
  one teacher-forced forward over the sampled sequences
  loss = −mean_tokens( A_i · log p_θ(token) )
         + β · KL(π_θ ‖ π_ref)            # anti-degradation anchor
         + λ · CE(ground-truth batch)      # optional SFT mixin
  backprop through LoRA params only, AdamW step
```

We stay strictly on-policy (one gradient step per rollout batch), so no
PPO-style importance ratios or clipping are needed — the loss reduces to
advantage-weighted log-likelihood. This is the simplest correct GRPO
variant and avoids storing old log-probs.

### Phase A: rollouts

`model.generate()` (used by `train/audio_logging.py`) yields audio but
doesn't expose codec token IDs, which we need for the teacher-forced
update. So the rollout loop drives the talker directly, mirroring the
prompt construction in `train/losses/codec_loss.py`:

```python
# train/grpo/rollout.py (sketch)

def sample_rollouts(model, text_ids, lang_codes, spk_embeds, *,
                    group_size, max_new_tokens, temperature, top_p):
    """Sample G codec sequences per prompt. Returns codec_ids [B*G, T] (right-
    padded after EOS) plus an attention/loss mask. No gradients."""
    talker = model.talker
    # Identical conditioning to SFT — this is what makes the log-probs in
    # Phase B consistent with what was sampled here:
    text_embeds = talker.text_projection(talker.get_text_embeddings()(text_ids))
    prefix, _   = _build_codec_prefix(talker, lang_codes, spk_embeds=spk_embeds)

    # Repeat each prompt G times → batch of B*G, then autoregressive loop
    # with KV cache: logits → temperature/top-p sample → append, stop at
    # codec_eos_id or max_new_tokens.
    ...
```

Notes:
- The conditioning prefix (think/lang tokens, speaker embed for
  Pipeline 2) must match `qwen3_tts_loss` / `qwen3_tts_speaker_loss`
  exactly — both functions already share `_build_codec_prefix`, so the
  rollout module imports it rather than re-implementing.
- **[impl] Layout decision.** Sampling and Phase-B teacher-forcing both use the
  **SFT concatenated layout** (`[text | prefix | codec…]`), *not* the official
  `generate()` interleaving (which sums text+codec frame-by-frame via
  `trailing_text_hidden`). Those are different distributions on the same weights;
  GRPO's on-policy correctness only needs Phase A and Phase B to agree, and the
  concatenated layout is what the SFT adapter was trained on. So the rollout
  feeds back **only the cb0 embedding** each step (matching the loss), and cb1–15
  are sampled solely to decode audio — they don't condition the cb0 stream.
- **[impl] Prompt tokenisation** uses `Qwen3TTSProcessor.encode_text` (same as
  SFT), not the chat-template wrapper `generate()` builds, for the same
  same-regime reason.
- Decode to audio with the frozen `model.speech_tokenizer` (code2wav),
  same path inference uses. Decoding is pure scoring input — wrap the
  whole phase in `mx.stop_gradient` semantics (just don't trace it).
- Prompts come from the existing train JSONL — only `text` (and
  `ref_audio` for Pipeline 2) are needed; codec labels are only used by
  the optional SFT-mixin term.

### Rewards

Mirroring the paper's design: cheap programmatic rewards per axis we care
about, plus one quality anchor to prevent degradation.

| Reward | Pipeline | Signal | Weight (start) |
|---|---|---|---|
| `r_intel` | 1 (language) | `1 − min(1, CER(ASR(audio), prompt_text))` via mlx-whisper; CER over WER for Hindi/Devanagari robustness | 1.0 |
| `r_spk` | 2 (cloning) | cosine(`speaker_encoder(mel(audio))`, `speaker_encoder(ref_mel)`) — encoder is already loaded and frozen | 1.0 |
| `r_len` | both | penalty −1 if generation hit `max_new_tokens` without EOS; small penalty for >60% trailing-silence frames (degeneracy guard) | 0.5 |
| `r_kl` | both | not a reward — per-token KL penalty in the loss (see below) | β = 0.02–0.1 |

The KL term plays the role of the paper's LLM-based quality reward: it
stops the policy from gaming ASR (e.g. slow robotic over-articulation)
by anchoring it to the SFT distribution. **[impl]** Reference log-probs are
computed once per rollout via `lora_swapped(model, snapshot)` — one extra
forward with the frozen SFT adapter installed, on the same float32 path the
policy uses. (`set_lora_enabled`/`lora_disabled` exist in `train/lora.py` but
must **not** be used as the KL reference; see "Why GRPO fits this stack".)
Two further fixes the implementation needed: log-probs are computed in
**float32** regardless of logits dtype (bfloat16 `log_softmax` is garbage in the
low-probability tail), and the k3 log-ratio is **clamped** to ±`kl_clip`
(default 10) before `exp` to stop a deep-tail sampled token from blowing up the
estimator.

ASR is the throughput bottleneck. Use `mlx-whisper` (small or
large-v3-turbo) in the same process; batch the G rollouts of one prompt
together. For Hindi specifically, normalize text (strip punctuation,
NFC) before CER.

### Phase B: update

The loss fn fits the existing `Trainer.value_and_grad_fn` contract
(`fn(model, batch) → (loss, metrics)`), so `train/trainer.py` is reused
mostly as-is — only the data fed to it changes (rollout batches instead
of dataset batches):

```python
# train/losses/grpo_loss.py (sketch)

def qwen3_tts_grpo_loss(model, batch, *, kl_beta=0.05, sft_lambda=0.0):
    """batch: text_ids, codec_ids (sampled), codec_mask, advantages [B*G],
    ref_logprobs [B*G, T] (precomputed in Phase A, no grad)."""
    # Same forward as qwen3_tts_loss up to codec_logits
    logits = ...                                  # [B*G, T, V]
    logprobs = nn.log_softmax(logits, axis=-1)
    tok_lp   = mx.take_along_axis(logprobs, targets[..., None], -1)[..., 0]

    adv  = batch["advantages"][:, None]           # [B*G, 1]
    mask = batch["codec_mask"][:, 1:]
    pg   = -(adv * tok_lp * mask).sum() / mask.sum()

    kl   = ((tok_lp - batch["ref_logprobs"]) * mask).sum() / mask.sum()
    loss = pg + kl_beta * kl
    # + sft_lambda * qwen3_tts_loss(...) on a ground-truth batch if enabled
    return loss, {"loss": float(loss), "pg": float(pg), "kl": float(kl),
                  "reward_mean": float(batch["advantages_raw_mean"])}
```

**[impl]** Two alignment details the sketch above glosses: (1) GRPO scores
**every** sampled token, so the codec window is `logits[:, off : off+T]` against
`targets = codec_ids` (full) — unlike `qwen3_tts_loss`, which drops `codec_ids[0]`.
(2) The PG term has **no label smoothing** (smoothing distorts the gradient);
smoothing is reserved for the optional SFT-mixin. Both `pg` and the reference
forward share one helper, `grpo_codec_logits`, so the slice can't drift apart.

**[impl]** Rather than overriding the inner loop, `GRPOTrainer` subclasses
`Trainer` and feeds it a `GRPORolloutLoader` that runs Phase A lazily per pull
and yields a Phase-B batch; the base loop then does loss+backward+step and
auto-logs every metric the loss returns. Set `grad_accumulation =
prompts_per_step` so one optimizer step == one on-policy rollout batch.
Checkpointing, LR schedule, TensorBoard, and JSONL logging are untouched. Watch
`reward_mean`, `cer_mean`, and `kl` — reward up while KL stays bounded is the
health signal.

## Config sketch

```yaml
# configs/qwen3_tts_hindi_grpo.yaml — extends the SFT config
model:   { ...same as qwen3_tts_hindi.yaml... }
lora:    { ...same; resume adapters from SFT checkpoint... }

grpo:
  init_adapters: "./checkpoints/qwen3-hindi/checkpoint-best/adapters.safetensors"
  group_size: 4            # G rollouts per prompt
  prompts_per_step: 4      # B → 16 rollouts/step
  max_new_tokens: 240      # ~20s at 12Hz; match dataset durations
  temperature: 0.9
  top_p: 0.95
  kl_beta: 0.05
  sft_lambda: 0.1          # optional CE anchor on real data
  rewards:
    intelligibility: { weight: 1.0, asr_model: "mlx-community/whisper-large-v3-turbo", metric: cer, language: hi }
    length_penalty:  { weight: 0.5 }
    # speaker_similarity: { weight: 1.0 }   # Pipeline 2 only

trainer:
  learning_rate: 5.0e-6    # ~4x lower than SFT — RL is noisy
  max_steps: 300
  grad_accumulation: 1     # one rollout batch = one step (on-policy)
  grad_clip: 1.0
```

## Compute reality check (M-series)

Per step at the sketch settings: 16 rollouts × ~15s audio ≈ 180 codec
tokens each (0.6B 8-bit talker, batched with KV cache), plus codec
decode, plus 16 ASR passes, plus two teacher-forced forwards (policy +
reference) and one backward. Expect **1–3 min/step** dominated by
rollouts+ASR → a 300-step run is an overnight job on an M5 Pro. Knobs if
too slow: whisper-small for ASR, `group_size: 3`, shorter prompts,
reward only on a 10s cap.

## What we are *not* doing

- No duplex/conversational rewards (turn-taking etc.) — that part of the
  Kyutai recipe needs a full-duplex model; revisit if/when CSM training
  matures into conversational finetuning.
- No learned reward model and no PPO critic — programmatic rewards +
  group-relative advantages only.
- No replication of their released weights (Moshi/PersonaPlex 7B RL
  checkpoints are gated PyTorch models; their training code is not
  public).

## Implementation order  *(all done — file map for the built pipeline)*

1. ✅ Reference policy in `train/lora.py` — `lora_snapshot()` + `lora_swapped()`
   (the float32-safe, SFT-anchored reference). `set_lora_enabled()`/`lora_disabled()`
   also exist but are *not* the KL reference (see KL section).
2. ✅ `train/grpo/rollout.py` — `sample_rollouts` (cb0 sampler + frozen
   `code_predictor` for cb1–15 → decode), `grpo_codec_logits` (shared aligned
   forward), `gather_token_logprobs`, `decode_codes_to_audio`.
3. ✅ `train/grpo/rewards.py` — CER via mlx-whisper + length/silence guard,
   group advantages; speaker-sim reward behind `w_speaker` (Pipeline 2).
4. ✅ `train/losses/grpo_loss.py` (`qwen3_tts_grpo_loss`) + `train/grpo/trainer.py`
   (`GRPORolloutLoader` + `GRPOTrainer`).
5. ✅ `configs/qwen3_tts_hindi_grpo.yaml`; wired into `scripts/train.py` via
   `pipeline: grpo`. Smoke-tested end-to-end (5 steps, `group_size: 2`,
   whisper-tiny): loop runs, reference KL anchors at 0, optimizer steps, ckpt
   saves. Meaningful reward-rise needs a real SFT'd adapter as the start (the
   base model gives zero-variance rewards → zero advantage) and full-length
   rollouts — GRPO is a refinement stage, not a from-scratch trainer.

## Validated result (first real run)

Interleaved-layout run on the Hindi SFT adapter
([akashicmarga/qwen3-tts-hindi-lora](https://hf.co/akashicmarga/qwen3-tts-hindi-lora)),
400 steps, `kl_beta=0.08`, 863 IndicVoices-R "Read" prompts → published as
[akashicmarga/qwen3-tts-hindi-lora-grpo](https://hf.co/akashicmarga/qwen3-tts-hindi-lora-grpo).
Held-out eval (120 sentences, 2 seeds; CER capped at 1.0):

| | CER ↓ | DNSMOS OVRL ↑ | SIG ↑ |
|---|---|---|---|
| SFT baseline | 0.205 | 3.237 | 3.573 |
| GRPO (interleaved) | **0.183** | **3.271** | **3.608** |

CER −11% rel, better on 75/120 (Wilcoxon p≈7e-4; paired t p≈0.09 — the mean is
outlier-sensitive, the rank test is the robust claim). MOS small but significant
(p<0.05). KL stayed ~0.015 — and **DNSMOS went up, not down**, which is the
direct evidence the policy did *not* hack the CER reward into robotic
over-articulation. The concatenated-layout run (v1) lost to SFT on the same
eval — confirming the interleaved layout was essential.

## Next steps / known limitations

Ordered roughly by value-per-effort. Items 1–3 are small and unblock the
ablation (4).

1. ✅ **In-loop legibility (cheap, do first).** *(done)*
   - **Zero-variance skip ratio per window** — `GRPORolloutLoader.pop_window_metrics()`
     reports `skip_ratio` / `skipped` / `built` over each logging window; the base
     `Trainer` drains it on the `log_every_n_steps` cadence (`train/skip_ratio`).
     A high rate = sparse PG signal even though the run "looks alive."
   - **Periodic fixed-prompt held-out eval** — `train/grpo/eval.py:make_grpo_audio_eval_fn`,
     wired into the base `Trainer.audio_eval_fn` hook (now fires on its own
     `eval_every_n_steps`, independent of a val_loader). Every window it rolls out
     a FIXED prompt set with a FIXED seed and logs `grpo_eval/{cer,duration_s,
     trailing_silence,kl}` + a sample waveform. Set a dedicated `grpo.eval.jsonl`
     for a truly held-out set; default is the first `num_prompts` of train. Needs
     `tensorboard_dir`.

2. ✅ **Sequence-normalized PG option** (`pg_norm: token | sequence`). *(done)*
   `token` (default) is the original `Σ/total_tokens` — length-weighted, the known
   GRPO length bias (Dr. GRPO / DAPO). `sequence` reduces via
   `train/losses/grpo_loss.py:_masked_reduce` to the mean over rollouts of each
   rollout's per-token mean, so every rollout contributes equally regardless of
   length. The same reduction is applied to the PG (main + sub) **and** KL terms
   so `kl_beta` keeps its meaning across modes. Empirically v3 already went the
   *right* way on length under `token`; the axis is now switchable for the
   ablation (4).

3. ✅ **Anti-reward-hacking guard.** *(done — off by default)* CER rewards
   ASR-friendly speech, which can degenerate into slow/over-enunciated audio. v3
   did *not* (MOS rose, KL tiny), but the defence is run-dependent, so it ships
   opt-in: `length_penalty.speaking_rate_min_cps` (0 = off; ~8–10 for Hindi) adds
   a graded penalty in `train/grpo/rewards.py:length_reward` when **voiced**
   chars/sec falls below the floor (trailing silence excluded so it doesn't
   double-count the silence term). The rate is logged as `grpo_eval/speaking_rate`.
   Keep `kl_beta` up and the length guard on. Note DNSMOS is English-trained → a
   rough relative proxy for Hindi naturalness, not a native MOS; a Hindi-tuned MOS
   or human listening is the real check.

4. ✅ **Controlled ablation** — *first cheap sweep run; directional, not yet
   significant.* Tooling: `scripts/grpo_ablation.py` (one config per cell → clean
   `scripts/train.py` subprocess → collects each run's `grpo_eval.jsonl`) +
   `scripts/grpo_heldout_eval.py` (post-hoc held-out CER, the real verdict the
   noisy 6-prompt in-loop eval can't give). Runs land OUTSIDE the repo.

   ```bash
   python scripts/grpo_ablation.py --base-config <grpo>.yaml --out-root ~/grpo_ablation --steps 100
   python scripts/grpo_heldout_eval.py --config <grpo>.yaml --ablation-root ~/grpo_ablation \
       --heldout-jsonl <val>.jsonl --num-sentences 100 --seeds 2
   ```

   **First sweep** (focused: layout fixed = interleaved, the proven winner;
   `pg_norm` × `sft_lambda`; group_size 2, 100 steps — deliberately cheap to
   compare axes, *not* a full validated run). Held-out CER (100 sentences × 2
   seeds, capped at 1.0; from the SFT adapter):

   | cell | CER ↓ | Δ vs SFT | median | cps | wilcoxon p |
   |---|---|---|---|---|---|
   | SFT baseline | 0.146 | — | 0.128 | 8.3 | — |
   | **sequence + sft 0.1** | **0.136** | **−0.010** | 0.120 | 7.6 | 0.10 |
   | sequence + sft 0 | 0.163 | +0.017 | 0.127 | 7.5 | 0.17 |
   | token + sft 0 | 0.184 | +0.039 | 0.126 | 8.4 | 0.25 |
   | token + sft 0.1 | 0.214 | +0.068 | 0.133 | 8.1 | 0.18 |

   Reading it honestly: **nothing is significant** (all p > 0.10 at this scale).
   But two directional findings: (a) **sequence-norm PG beats token-norm in both
   `sft` settings** — the clearest axis signal, consistent with the #2 length-bias
   argument; (b) **only `sequence + sft 0.1` beat SFT** (−6.7% rel, median also
   down), i.e. the two new switches help *in combination*. token cells degraded —
   `token+sft0.1`'s mean is outlier-inflated (its median, 0.133, is near baseline),
   so the damage is a few degenerate rollouts, not a uniform shift. No reward
   hacking anywhere (cps 7.5–8.4 vs 8.3, no rate collapse — the #3 guard wasn't
   needed). This does **not** overturn the validated result (gs4, 863 prompts, 400
   steps, −11%); its job is config guidance for the next full run: **sequence-norm
   PG + `sft_lambda` 0.1**. Open: re-run at gs4 / longer / more seeds to move from
   directional to significant, and add the concatenated-layout arm.

   **Follow-up sweep at gs4** (group_size 4, prompts_per_step 4, kl_beta 0.08,
   200 steps; interleaved layout, same `pg_norm` × `sft_lambda` grid). Held-out CER
   (100 sentences × 3 seeds, capped at 1.0; from the SFT adapter):

   | cell | CER ↓ | Δ vs SFT | rel. | median | DNSMOS | cps | wilcoxon p |
   |---|---|---|---|---|---|---|---|
   | SFT baseline | 0.163 | — | — | 0.126 | 3.28 | 7.9 | — |
   | sequence + sft 0.1 | 0.122 | −0.041 | −25.4% | 0.110 | 3.33 | 7.7 | 2.5e-05 |
   | sequence + sft 0 | 0.120 | −0.043 | −26.2% | 0.107 | 3.31 | 7.8 | 2.1e-09 |
   | token + sft 0.1 | 0.119 | −0.045 | −27.2% | 0.102 | 3.29 | 8.2 | 6.7e-10 |
   | **token + sft 0** | **0.114** | **−0.049** | **−29.8%** | 0.103 | 3.30 | 7.9 | 5.7e-09 |

   Now it's significant and the story flips clean: **every GRPO cell beats SFT by
   25–30% relative CER, all p ≪ 1e-4** (down to 5.7e-9). DNSMOS is flat-to-up
   everywhere (3.28 → 3.29–3.33) and cps stays 7.7–8.2 vs 7.9 → **no reward
   hacking**. The config axis, significant at last, is a **near-tie with a mild
   gradient toward token-norm** (best cell `token + sft 0`) — it does *not*
   reproduce gs2's `sequence + sft 0.1` edge. Takeaway: at gs4 the knob choice is
   second-order; the first-order, robust, reproducible effect is **GRPO ≫ SFT**.
   gs2's directional ranking was small-sample noise, not a real config preference.
   `scripts/grpo_heldout_eval.py` now checkpoints per-adapter (`heldout_partial.json`)
   and resumes, so an interrupted eval only redoes the in-flight cell. Open:
   concatenated-layout arm still unrun; a Hindi-native MOS to confirm the
   DNSMOS-flat quality read.

5. ✅ **Richer reward for a bigger gain** — *wired, off by default.* CER rewards
   legibility, not quality (a robotic but transcribable clip still scores well), so
   CER-only plateaus (~step 200). `train/grpo/rewards.py:naturalness_reward` adds a
   **reference-free DNSMOS (P.835)** term behind `w_mos`: `r_nat = (OVRL−1)/4 ∈
   [0,1]`, combined as `… + w_mos·r_nat`. Enable with a `rewards.naturalness`
   block (`weight`, `metric: ovrl|sig|bak`); needs `pip install speechmos
   onnxruntime`. Logged as `mos_mean` in training and `grpo_eval/dnsmos_ovrl` in the
   in-loop eval; `scripts/grpo_heldout_eval.py` now reports a DNSMOS column matching
   the validated-result table. DNSMOS is English-trained → a **relative** proxy for
   Hindi, not a native MOS; pair the CER term with it (don't drop CER), and the
   real check is still a Hindi-tuned MOS or human listening. Open: tune `w_mos` vs
   `w_intel` on a real run — start ~0.3–0.5 so CER stays the primary signal.

6. ✅ **Pipeline 2 (speaker cloning) interleaved** — *validated end-to-end.*
   Config: `configs/qwen3_tts_speaker_grpo.yaml` (`layout: interleaved`,
   `speaker_similarity.weight > 0`, a `train_jsonl` with `ref_audio` per record).
   The real fix: the interleaved rollout clones the voice via the generate() path,
   which calls `extract_speaker_embedding(ref_audio)` expecting a **24 kHz waveform**
   — but the loader was handing it the ref_audio **path string** (the bug that kept
   Pipeline 2 concatenated-only). Now `build_grpo_prompts` loads the waveform once
   (`ref_audio_wav`) and the loader/eval thread it through. Observability:
   `spk_sim_mean` in training metrics, `grpo_eval/spk_sim` in the in-loop eval.
   Validated by a 40-step interleaved run (CER + speaker-sim reward, sequence-norm
   PG): trains cleanly, KL anchors, checkpoint saves. **Finding:** with a strong
   base model the speaker-sim **saturates (~0.995 cosine)** — it already clones
   near-perfectly from ref_audio — so the term acts as a *guard*, not the learning
   driver; CER does the driving. `scripts/grpo_heldout_eval.py` auto-detects
   Pipeline 2 (held-out prompts carrying `ref_mel`): it clones from `ref_audio` and
   adds a `spk_sim` column next to CER/MOS, so cells can be ranked on voice match
   too. The saturation means the held-out speaker-sim gap between SFT and GRPO is
   small — to make speaker-sim the *driver* you'd need a harder cloning setup
   (unseen target voices, or a weaker init) where it isn't already ~1.0.

Also fixed in passing: the base `Trainer` only broke its inner batch loop on
`max_steps`, so leftover epochs each pulled one extra batch — cheap for SFT, but an
expensive wasted rollout for GRPO. It now breaks the epoch loop too.
