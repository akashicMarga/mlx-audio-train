# Pocket TTS distillation on Apple Silicon (MLX)

Port of Kyutai's Pocket TTS teacher→student distillation
([blog, 2026-10-20](https://kyutai.org/blog/2026-10-20-pocket-tts-small/),
[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) `training/`,
PRs [#364](https://github.com/kyutai-labs/pocket-tts/pull/364) /
[#367](https://github.com/kyutai-labs/pocket-tts/pull/367)) to MLX.

## The recipe

Pocket TTS is a CALM model ([arXiv:2509.06926](https://arxiv.org/abs/2509.06926)). It is a
causal transformer over continuous Mimi latents (32-d, 12.5 Hz). Each frame's backbone
output `z` feeds a small MLP sampler head (LSD, drifting or pointwise; 1-step decode) and
an EOS head. Kyutai trains a 24-layer teacher, then distils it:

| | teacher | student |
|---|---|---|
| backbone | 24 layers, d 1024 (302M) | 6 layers (75M); optionally narrower |
| heads | trained | **copied from the teacher, frozen** |
| target | `z_null + cfg·(z_cond − z_null)`, cfg 2 | regress with masked MSE on `z` |
| init | — | teacher's bottom+top layers ("ends"), strongest heads / channels / FFN units, SVD text table |
| optimiser | — | AdamW lr 4e-4 cosine, warmup 1k, batch 64, EMA 0.9999, 200k steps |

Because the heads are frozen, the student must emit `z` in the teacher's space, so
**the z width must match** (#367). A teacher with no `z_dim` exposes d_model (1024), and
then the student must keep d_model 1024. Students can still be shallower, use fewer or
grouped attention heads, a narrower FFN and a low-rank text table. CFG is baked in, so the
student samples at cfg 1 (one backbone pass per frame).

Kyutai's result (8×H100): the distilled 6-layer student matches or beats its teacher
(WER 0.76% vs 0.82%, speaker sim 0.921 vs 0.929, UTMOS 4.35 vs 4.33). WER reaches parity by
~50k steps; similarity and UTMOS keep climbing until ~150k.

## What is in this repo

| file | |
|---|---|
| `models/pocket_tts/model.py` | MLX FlowLM, with weight names identical to the PyTorch state dict. Covers z_dim, attn_dim, GQA, dim_feedforward, low-rank LUT, and lsd / flow_matching / drifting / pointwise heads. KV-cache sampling with optional guidance. |
| `models/pocket_tts/shrink.py` | numpy port of `shrink_checkpoint.py` (depth + width seeding; PCA cut not ported) |
| `models/pocket_tts/distill.py` | student builder (seeding, frozen heads, z-width / head-width checks), CFG teacher target, masked MSE; experimental `fold_z_bottleneck` |
| `models/pocket_tts/objectives.py` | **fine-tuning objective**: LSD flow loss (jvp self-distillation term, learned uncertainty weighting), EOS loss, CFG dropout, `flow_batch_multiplier`, latent-stats EMA |
| `models/pocket_tts/finetune.py` | fine-tune builder: same-language, new-language (`reset_text_embedding`), from scratch; `freeze_head` |
| `models/pocket_tts/teacher_cache.py` | precomputed teacher targets: run the teacher once, then train the student only (~3× faster steps) |
| `models/pocket_tts/train_utils.py` | shared loop: AdamW + warmup/cosine, grad accumulation, clipping, EMA, checkpoints, resume, validation, benchmark |
| `models/pocket_tts/data.py` | latent dataset: word-boundary prompt cuts (official recipe), `ref_audio` prompts or a same-clip fallback; length-bucketed batches |
| `models/pocket_tts/text.py` | text normalisation applied identically to tokenizer fitting, training and sampling; `hindi`: digits / ₹ / % → Hindi words (lakh / crore) |
| `models/pocket_tts/codec.py` | official PyTorch Mimi: encode audio → latents, decode latents → audio |
| `models/pocket_tts/weights.py` | load released `model.safetensors` (local or `hf://`, also community configs); export checkpoints in the official layout plus `config.yaml` and tokenizer |
| `scripts/pocket_tts_precompute_latents.py` | audio → Mimi latents (once) |
| `scripts/pocket_tts_distill.py` | distillation (`--smoke-test`, `--benchmark`, `--cache-teacher`, resume) |
| `scripts/pocket_tts_finetune.py` | fine-tuning / new-language teacher / from scratch (`--smoke-test`, `--benchmark`, resume) |
| `scripts/pocket_tts_train_tokenizer.py` | SentencePiece tokenizer for a new language, with the released tokenizers' spec |
| `scripts/pocket_tts_sample.py` | generate with any checkpoint at any guidance (teachers want cfg ≈ 2; `pocket-tts generate` has no guidance option) |
| `scripts/pocket_tts_fold_bottleneck.py` | experimental: add `z_dim` to a teacher so narrow students become possible |
| `configs/pocket_tts/*.yaml` | English: 24→6 (official), 24→6 slim, 24(z64)→8L-512 (experimental) |
| `configs/pocket_tts/hindi/*.yaml` | Hindi experiments H1–H4, see below |
| `tests/test_pocket_tts.py` | 22 tests, including parity against the official PyTorch code |

## Verification done (Linux CPU container, MLX 0.32.4, torch 2.14)

* **Numerical parity with the official PyTorch code**, using random weights for three
  architecture variants: the released 100M layout, the small-model layout (z_dim + attn_dim +
  GQA + low-rank LUT + drifting head) and a pointwise head. Teacher-forced `z` (conditioned
  and null branch), sampler-head output and EOS logits all agree to 1e-3. A deliberate
  RoPE bug makes the test fail, so the check is sensitive.
* **Autoregressive generation** (KV cache, cfg 1 and 2, temp 0) matches the official
  `TrainableTTS.generate` to 1.2e-5 over 12 frames.
* **Seeding** matches `shrink_checkpoint.shrink` + `shrink_width` tensor for tensor. The
  SVD text table is compared through its product, since SVD factors are only unique up
  to sign.
* **Exported checkpoints load with `strict=True`** into the official PyTorch FlowLM.
* **Real-size configs build and seed** from a full-size (randomly initialised) 24-layer teacher:

  | config | flow LM total | backbone | trainable |
  |---|---|---|---|
  | teacher `english_2026-09_24l` | 316.0M | 302.1M | 0 (frozen) |
  | `distill_24l_to_6l.yaml` | 89.4M | 75.5M | 79.7M |
  | `distill_24l_to_6l_slim.yaml` | 51.4M | 40.9M | 41.6M |
  | `distill_24l_z64_to_8l_512.yaml` (folded teacher) | 36.6M | 25.2M | — |

* **End-to-end smoke run** (`--smoke-test`: tiny random 8-layer teacher → 2-layer narrow
  GQA student, synthetic data): distillation loss 0.036 → 0.0098 in 120 steps. Checkpoint
  export and resume (weights, optimiser and EMA restored; next step bit-identical) work.
* **Latent precompute** checked with a randomly initialised Mimi: [T, 32] latents at
  12.5 Hz, and `start`/`duration` segment windows are handled.

**Not verified here:** anything that needs the real weights or a Mac. This container
cannot reach huggingface.co, and MLX's Linux CPU backend is ~45× slower than numpy at
matmul, so no meaningful timing was possible. That includes a real 24→6 run, audio
quality / WER, and Apple-GPU speed. Those are steps 1–3 below.

Added with the fine-tuning port (same container):

* **LSD loss and its gradients match the official PyTorch `LSD`** for both stop-gradient
  modes, given fixed times / noise and a non-trivial uncertainty net. Loss agrees to 2e-4.
  Gradients agree for the head parameters, `w_s_t` and the conditioning `z`. This caught a
  real bug: the `minimal` target must also cut the gradient to `z`.
* EOS loss matches the reference formula. `reset_text_embedding` re-initialises only the
  text table, and `freeze_head` leaves the head untrainable.
* Fine-tune smoke run from scratch: loss 2.40 → 1.02 in 150 steps.
* Cached teacher targets reproduce the live teacher's targets (fp16) and the same loss.
* Sampling script: tiny model plus a random-weight Mimi → 2 s of 24 kHz audio, with cfg 2
  and a voice prompt.
* All Hindi configs build at full size from a stand-in with the Saryps layout (random
  weights). Flow LM sizes: teacher 89.4M → 4-layer 64.3M, 3-layer 51.7M, slim 6-layer
  51.4M. The fine-tune config leaves 79.7M trainable with the head frozen.

## Feasibility

**Weights: yes.** The 24-layer teachers are public: `english_2026-09_24l`, plus 24l
variants for French, German, Spanish, Italian, Portuguese and Dutch. Voice-cloning
weights are gated (accept the terms on `kyutai/pocket-tts`); the
`pocket-tts-without-voice-cloning` copies distil fine but carry no voice prompt.

**Memory: easy.** Frozen teacher in bf16 ≈ 0.6 GB. Student weights, grads, AdamW
moments and EMA ≈ 1.8 GB (6L). Activations at micro-batch 16 × ~190 tokens are a few GB.
16 GB Macs work at micro-batch 16, 8 GB at micro-batch 4–8 with more accumulation.

**Compute: the real cost.** Per optimiser step (64 rows, ~10 s target + ~3 s prompt + text
≈ 190 tokens, null branch ≈ 125 tokens):

| | FLOPs / step |
|---|---|
| teacher, 2 forward passes (cond + null), 302M | ≈ 12 TFLOP |
| student forward + backward, 75M (slim: 41M) | ≈ 5.4 TFLOP (2.9) |
| **total** | **≈ 17.5 TFLOP** (slim ≈ 15) |

As a sanity check, Kyutai's 10.5 steps/s on one H100 implies ≈ 180 TFLOP/s achieved,
which is plausible. On Apple GPUs, at an *assumed* sustained MLX throughput:

| sustained throughput (assumption) | s / step | 50k steps (WER parity) | 200k steps (full) |
|---|---|---|---|
| ~8 TFLOP/s (Max-class) | ~2.2 | ~30 h | ~5 days |
| ~4 TFLOP/s (Pro-class) | ~4.4 | ~2.5 days | ~10 days |

These are estimates. Run `--benchmark` (step 1) to get the real number for your machine.
Levers, roughly in order of payoff:
1. **Precompute teacher targets once** (not implemented yet). This removes ~70% of the
   step, about 3× faster. Storage is ~2 KB per frame (1024-d fp16): ~18 GB for 200 h, but
   the prompt cuts are then fixed rather than resampled each epoch.
2. Teacher in bf16 (already the default in the configs).
3. Stop at ~50k steps: WER is at parity there, and the rest polishes similarity and UTMOS.
4. `cfg_coef: 1` skips the null pass (~35% less compute), but CFG is no longer baked in.
5. Shorter crops (`max_duration_sec: 15`), or the slim student.

**Data: modest.** Distillation only has to match the teacher on its input distribution.
200–2000 h of HiFiTTS-2 with Kyutai's precomputed word alignments
(`uv run training/scripts/prepare_data.py --hours 200` in the pocket-tts repo) gives the
exact official prompt-cut recipe. Latents cost ~6 GB per 1000 h.

**Scope limits: honest version.**
* *100M-class 6-layer student from the public teacher*: fully supported. This is the
  official recipe.
* *Slimmer student at d_model 1024* (GQA, half FFN, low-rank text table; ~51M flow LM):
  supported. It is not a Kyutai config, so quality is unknown until you train it.
* *Kyutai's 18m / 7m / 2m models*: **not reproducible from public weights as is.** Their
  teachers have `z_dim: 64` and are not released. They also use head distillation (a new
  192-d drifting/pointwise head trained against the frozen one), a smaller separately
  trained Mimi decoder, and REFL reward fine-tuning, none of which is ported. The
  experimental `pocket_tts_fold_bottleneck.py` folds a rank-64 bottleneck into the public
  teacher (reduced-rank regression on the head inputs) so narrow students can be tried.
  Whether the folded teacher still sounds right has to be measured before distilling
  from it.
* *A new language (e.g. Hindi)*: distillation needs a teacher that speaks it. There is
  no public Hindi Pocket TTS, so you would first fine-tune the 24-layer teacher
  (upstream `finetune_language.yaml`, PyTorch/GPU). Alternatively, port the LSD flow
  objective to MLX (`mx.jvp` exists). Then distil with this code.

## Step-by-step plan

**Step 0 — done in this change.** MLX port, seeding, distillation loop, export, and parity
tests against the official code.

**Step 1 — sanity and speed on your Mac (≈ 15 min).**
```bash
pip install mlx pyyaml safetensors huggingface_hub tokenizers sentencepiece
python scripts/pocket_tts_distill.py --smoke-test
huggingface-cli login   # accept the terms on kyutai/pocket-tts first (or use the -without-voice-cloning weights)
python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml --benchmark 10
```
Gate: if s/step × 50k is acceptable, go on. If not, implement precomputed teacher
targets first (lever 1).

**Step 2 — optional parity check on real weights.**
`git clone https://github.com/kyutai-labs/pocket-tts && POCKET_TTS_REPO=... pytest tests/test_pocket_tts.py`.
Optionally compare `FlowLM.generate_latents` with `pocket-tts generate` on the 24l
teacher at temp 0.

**Step 3 — data (≈ hours, mostly download).**
```bash
# in a pocket-tts checkout: HiFiTTS-2 subset + word alignments
uv run training/scripts/prepare_data.py --hours 200
# here: encode to latents with the teacher's Mimi (CPU or MPS); needs `pip install pocket-tts`
python scripts/pocket_tts_precompute_latents.py --input <pocket-tts>/data/train_aligned.jsonl \
    --out-dir data/pocket_latents/train --config configs/pocket_tts/models/english_2026-09_24l.yaml
python scripts/pocket_tts_precompute_latents.py --input <pocket-tts>/data/valid_aligned.jsonl \
    --out-dir data/pocket_latents/valid --config configs/pocket_tts/models/english_2026-09_24l.yaml
```
Your own `{"audio", "text", "ref_audio"}` JSONL also works. Without word alignments,
prompts come from `ref_audio`, or else from a window of the same clip.

**Step 4 — pilot (10k steps).**
`python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml --max-steps 10000`
then `pocket-tts generate --config checkpoints/pocket-distill-6l/step_0010000/config.yaml --voice v.wav --text "..."`.
Gate: `loss` / `val_loss` falling steadily (watch with `scripts/watch_training.py --log
checkpoints/pocket-distill-6l/train_log.jsonl`) and intelligible audio at cfg 1. Measure WER
with `training/eval/librispeech.py` from the pocket-tts repo on the exported checkpoint.

**Step 5 — full run to 50k → 200k steps.** Resume with `--resume <ckpt dir>`. Compare
against the released `english_2026-09_2` 6-layer model, which comes from the same
teacher family.

**Step 6 — smaller students (research).** Train `distill_24l_to_6l_slim.yaml`. For
d_model < 1024: fold the bottleneck, check the folded teacher's WER, then train
`distill_24l_z64_to_8l_512.yaml`. Porting head distillation and the small Mimi decoder
comes next if you want the true 7m / 2m sizes.

**Step 7 — new language (research).** Fine-tune the 24-layer teacher for the language,
then repeat steps 3–5.

## Hindi experiments (`configs/pocket_tts/hindi/`)

Pocket TTS reads raw text through a 4,000-piece SentencePiece table, not phonemes.
That suits Devanagari, whose spelling is close to pronunciation. The weak spots are
schwa deletion, numbers and abbreviations, and English words. `normalize_text: hindi`
spells out digits, ₹ and %. Use the same setting when fitting the tokenizer, training
and sampling.

Public Hindi starting points (CC-BY-4.0, credit the authors):
[saryps-labs/pocket-tts-hindi](https://huggingface.co/saryps-labs/pocket-tts-hindi), a
6-layer model distilled from a 24-layer Hindi teacher trained on 932 h of IndicVoices +
Kathbath (WER 23.3% / CER 11.5% on held-out speakers, real audio 26.7% / 13.9%).
Also [Thanushttk/pocket-tts-multilingual](https://huggingface.co/Thanushttk/pocket-tts-multilingual),
6 layers, Hindi plus 5 other Indian languages and English. No 24-layer Hindi teacher is
public.

| id | experiment | script / config | time on an M5 Pro (estimate*) |
|---|---|---|---|
| H1a/b/c | Saryps 6-layer → 4-layer / 3-layer / slim 6-layer students (**cfg 1**: the teacher is already distilled) | `distill` · `distill_saryps6l_to_{4l,3l,6l_slim}.yaml` | ~7–15 h to 50k steps |
| H2 | adapt Saryps 6-layer to your voices / domain (head frozen, no dropout) | `finetune` · `finetune_saryps6l.yaml` | ~3–7 h for 10–15k steps |
| H3 | **Hindi 24-layer teacher** from the English 24-layer (new tokenizer, fresh text table) | `train_tokenizer` + `finetune` · `teacher_24l_from_english.yaml` | ~10–21 h to 15k steps (WER plateau); days for acoustic polish |
| H3′ | Hindi 24-layer teacher from scratch (the Saryps recipe, 400k steps) | `finetune` · `teacher_24l_scratch.yaml` | weeks (multi-machine territory) |
| H4a/b | your H3 teacher → 6-layer / slim 4-layer students (**cfg 2**) | `distill` · `distill_teacher24l_to_{6l,4l_slim}.yaml` | ~1–2 days to 50k (≈3× less with `--cache-teacher`) |
| H4c | H3 teacher folded to z_dim 64 → 8-layer d512 student (experimental) | `fold_bottleneck` + `distill` · `distill_teacher24l_z64_to_8l_512.yaml` | ~0.5–1 day to 50k |

\* From FLOP counts per batch-64 step at an *assumed* 5–10 TFLOP/s sustained MLX
throughput. Run `--benchmark 10` on any config for the real number.

```bash
# data (once): your Hindi JSONL {"audio", "text", "ref_audio"} -> latents
python scripts/pocket_tts_precompute_latents.py --input data/hindi/train.jsonl \
    --out-dir data/pocket_latents/hindi_train --config configs/pocket_tts/models/english_2026-09_24l.yaml
python scripts/pocket_tts_precompute_latents.py --input data/hindi/valid.jsonl \
    --out-dir data/pocket_latents/hindi_valid --config configs/pocket_tts/models/english_2026-09_24l.yaml

# H1: smaller students from the public Hindi model
python scripts/pocket_tts_distill.py --config configs/pocket_tts/hindi/distill_saryps6l_to_4l.yaml

# H3 -> H4: your own Hindi teacher, then students
python scripts/pocket_tts_train_tokenizer.py --out data/hindi_tokenizer --normalize-text hindi data/hindi/train.jsonl
python scripts/pocket_tts_finetune.py --config configs/pocket_tts/hindi/teacher_24l_from_english.yaml
python scripts/pocket_tts_sample.py --config checkpoints/pocket-hindi-teacher-24l/step_0015000/config.yaml \
    --cfg 2 --voice voice.wav --text "आज मौसम बहुत अच्छा है" --normalize-text hindi --out teacher.wav
python scripts/pocket_tts_distill.py --config configs/pocket_tts/hindi/distill_teacher24l_to_6l.yaml \
    --cache-teacher data/pocket_teacher_cache/hi24l          # optional, then set data.teacher_cache
python scripts/pocket_tts_distill.py --config configs/pocket_tts/hindi/distill_teacher24l_to_6l.yaml
```

The latents are the same for every experiment: all these models share Kyutai's Mimi
codec. Distillation configs point their teacher at a step directory. Edit the path to
the checkpoint you chose, and score it before distilling (sample at cfg 2).

Not ported: head distillation (smaller 192-d drifting / pointwise heads), the
small-model Mimi decoder, REFL reward fine-tuning, the drifting objective, and Muon.
Together these are what take Kyutai's English models down to 18M / 7M / 2M.
