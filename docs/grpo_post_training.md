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

**Status: design only — not implemented.** Use this when SFT plateaus
(e.g. the Hindi adapter is fluent but WER is stuck, or speaker similarity
stalls). Not worth the rollout cost before that.

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
- A frozen reference policy comes for free with LoRA: running the model
  with LoRA deltas disabled *is* the SFT-anchored reference for the KL
  penalty (or snapshot the adapters at stage start).

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
by anchoring it to the SFT distribution. With QLoRA the reference
log-probs cost one extra forward with the LoRA contribution zeroed —
add a `set_lora_enabled(model, flag)` toggle to `train/lora.py` that
scales `lora_b` output by 0/1.

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

A thin `GRPOTrainer` orchestrates the two phases; it can subclass
`Trainer` and override the inner loop, keeping checkpointing, LR
schedule, TensorBoard, and JSONL logging untouched. Log `reward_mean`,
`cer_mean`, `spk_sim_mean`, and `kl` — reward going up while KL stays
bounded is the health signal.

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

## Implementation order

1. `set_lora_enabled()` toggle in `train/lora.py` (reference policy).
2. `train/grpo/rollout.py` — batched sampler sharing `_build_codec_prefix`;
   validate by decoding rollouts and listening.
3. `train/grpo/rewards.py` — CER reward via mlx-whisper + length guard;
   speaker-sim reward behind `include_ref_mel`.
4. `train/losses/grpo_loss.py` + `GRPOTrainer` two-phase loop.
5. `configs/qwen3_tts_hindi_grpo.yaml`; smoke test = 5 steps,
   `group_size: 2`, dummy reward, assert reward_mean rises on a toy task.
