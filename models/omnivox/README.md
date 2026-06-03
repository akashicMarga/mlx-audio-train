# OmniVox — research notes

A small, experimental **speech-to-speech model for Hindi**.

This is a research experiment, not a product. The goal is to learn — by
building — what it takes to make a compact Hindi conversational S2S model
run end-to-end on Apple Silicon. Audio in → audio out, no off-the-shelf
ASR/TTS in the loop at inference time.

---

## Research objective

> Can a ≤2B-parameter speech-to-speech model hold a basic conversation
> in Hindi, trained on a few thousand hours (or less) of Hindi audio,
> on a single M-series Mac?

The model has to:

1. Understand Hindi speech (no language ID, no romanisation).
2. Reason in Hindi at the text-token level inside the LLM backbone.
3. Generate Hindi speech directly as Mimi codes (no external TTS).

We are explicitly **not** building a modular ASR → LLM → TTS pipeline.
The pipeline approach works today and is not interesting as research —
the interesting question is whether the single-model formulation
(MiniMind-O / Moshi-style) survives at small scale when the dominant
LLM is multilingual but not Hindi-native.

---

## Architecture

```
Hindi audio (24kHz)
  └─► Whisper-small encoder (frozen, 768d @ 50fps)
        └─► MMAudioProjector (768 → backbone_hidden)         [trainable]
              └─► injected at <|audio_pad|> positions
                    └─► LLM backbone (Qwen2.5-0.5B / Sarvam-1)
                          │
                          ├─► text head (Hindi text tokens)
                          │
                          └─► bridge layer 20/24 → TalkerModule (4-layer)
                                └─► 8 Mimi codebook logits per step
                                      └─► Mimi decoder → 24kHz audio
```

Modules reused unchanged from `models/minimind_o/`:
- `MMAudioProjector` (generic `in_dim → out_dim`)
- `TalkerModule` (4-layer transformer + 8 MTP heads + speaker proj)

Modules new in OmniVox:
- `OmniVox` (wires backbone + projector + talker)
- `OmniVoxConfig` (Qwen / Sarvam dim wiring, audio token IDs, bridge layer)
- Phase 1a / 1b / T2A training scripts

### Why this shape

Three reasons for choosing the MiniMind-O shape over alternatives:

1. **Mimi is language-agnostic**: Mimi codes are acoustic units, not
   linguistic ones. A Talker trained on one language can in principle
   be fine-tuned on another with much less data than training from scratch.
2. **The bridge layer is cheap**: routing hidden states from a single
   LLM layer into a small Talker keeps the trainable footprint tiny
   (~60M params) compared to a full speech LM.
3. **Apple Silicon constraint**: 0.5B–2B backbone + small Talker fits
   in 16GB unified memory and trains in MLX without distributed setup.

---

## Backbone candidates

| Model | Params | Hindi-native | MLX | Notes |
|-------|--------|--------------|-----|-------|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 0.5B | No (token coverage exists, but Hindi is a minor mix) | ✓ | Currently in use. Generates Chinese/code when uncertain — the prior dominates. |
| `mlx-community/sarvam-1-4bit` | 2B | **Yes** — pretrained on Hindi + 10 Indic langs by Sarvam AI (Indian) | ✓ | Llama architecture, drop-in replacement. 4× larger but native Hindi. **Top candidate for next iteration.** |
| `sarvamai/sarvam-1-v0.5` | 2B | Yes | needs conversion | Older Sarvam-1 release. |
| `sarvamai/OpenHathi-7B-Hi-v0.1-Base` | 7B | Yes (Hindi-only) | needs conversion | Too large for this experiment. |

**Recommendation**: switch to `mlx-community/sarvam-1-4bit` for the next
training run. The Chinese/code hallucination in Phase 1a output is
strong evidence the Qwen prior is fighting the Hindi signal. Sarvam-1
is the right size and the right prior — built by Sarvam AI on Indic
data from scratch, Llama architecture so it slots into our existing
manual-forward loop with only hidden_size + tokenizer changes.

Trade-off: 4× the backbone params means slower training and more memory.
Acceptable given the research budget.

---

## Training phases

The intended order is **T2A → Phase 1a → Phase 1b**.

### T2A (text → Mimi codes)
Teach the LLM backbone + Talker to *speak* Hindi from Hindi text.
Backbone frozen, Talker + audio_proj trainable (~60M params).
Without this step the Talker is random and the model cannot produce
any coherent audio — A2A on top of a random Talker is meaningless.

### Phase 1a (projector alignment)
Freeze backbone + Talker. Train only `MMAudioProjector` (~1.5M params)
to map Whisper features into the backbone's embedding space so the
backbone can transcribe / understand Hindi speech.

### Phase 1b (top-N backbone layers + projector)
Unfreeze top 2 backbone layers + final norm. Same Whisper → projector
→ backbone path, but now the backbone itself can adapt to Hindi audio
features. Needs fp16 backbone (the 4-bit version can't be updated).

### A2A (full S2S, future)
Combine: Whisper input + trained projector + Phase 1b backbone +
trained T2A Talker. End-to-end speech-in, speech-out.

---

## Current status (as of this README)

| Phase | Ran? | Outcome |
|-------|------|---------|
| T2A | Yes, 25 epochs total | **Failed.** Train loss 6.6 → 4.9, val loss 6.4 → 6.7 (regressed). Audio output is essentially noise. |
| Phase 1a | Yes, 5 epochs | Train loss 1.84 → 0.19, val loss stuck at ~0.49. Generation: 1 correct Hindi char then Chinese/Python hallucination. |
| Phase 1b | No | Script ready, never run. |
| A2A inference | No | No script written. |

### What we learned

1. **The T2A code-alignment is wrong.** `align_codes_to_len` in
   `scripts/omnivox_t2a_train.py` subsamples 12.5fps Mimi codes to
   match the 512-token text sequence length and places all 8 codebooks
   in parallel at the same positions. MiniMind-O's `dataset.py` instead
   places codes at `pos = asst_start + li + i` — a staggered/diagonal
   layout where codebook `li` is offset by `li` positions from frame 0.
   This is the MTP (Multi-Token Prediction) layout and is the format
   the Talker is architecturally designed to learn. **Until this is
   fixed, T2A cannot learn anything meaningful.**

2. **The backbone prior dominates Phase 1a.** A 1.5M-param projector
   cannot redirect a 500M frozen LLM away from its dominant pretraining
   distribution. Phase 1b (unfreezing top layers) is the architectural
   fix; Sarvam-1 (replacing the prior itself) is the data fix.

3. **`talker.npz` was never saved.** All 25 T2A epoch checkpoints
   contain only `audio_proj.npz`. The Talker state — the actual thing
   we were training — was lost. Either the save path is buggy, or the
   ~60M-param Talker exceeded an implicit npz size limit and failed
   silently. Need to investigate and switch to safetensors / split
   shards.

4. **1900 samples is the floor, not the goal.** The current Hindi
   dataset (`/Users/akashsingh/Documents/exps/hindi/`) has 1900 train
   clips. FLEURS Hindi train has more (~1.7k–3k depending on filter).
   For T2A from scratch we need 10k+ clips. For T2A fine-tuned from
   a pretrained MiniMind-O Talker, 1900 might be enough as a starting
   experiment.

---

## Open research questions

These are the actual things we want to learn from this experiment:

1. **Does a Chinese-pretrained Talker transfer to Hindi via fine-tune?**
   Mimi codes are acoustic — in theory the Talker shouldn't care about
   language. MiniMind-O weights are available at
   `~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/` —
   loading those into our Talker before T2A could collapse the
   "1900 samples is too few" problem.

2. **Does swapping Qwen → Sarvam-1 fix the Phase 1a hallucination?**
   If the projector + frozen backbone can produce Hindi text with a
   Hindi-native backbone (without unfreezing any layers), that tells
   us the Phase 1a architecture is sound and Qwen was just the wrong
   prior. If it still hallucinates, the projector itself is the issue.

3. **What's the minimum bridge-layer choice?** We use layer 20/24 of
   Qwen. MiniMind-O used 3/8 (~37%). For Sarvam-1 (Llama-2B) at 22
   layers, what's the right bridge? Earlier = less semantic, easier
   audio prediction. Later = richer semantic, harder for Talker to
   decode.

4. **Can we do without speaker conditioning?** OmniVox keeps the
   192d CAM++ speaker embedding path from MiniMind-O. For a "basic
   conversation" target we may not need voice cloning at all — a
   single learned default voice is enough and removes a whole data
   requirement (speaker embeddings per utterance).

5. **Token boundary alignment**: Hindi text tokens and Mimi audio frames
   are at different rates. MiniMind-O's diagonal placement implicitly
   handles this. Is there a better explicit alignment (e.g. CTC-style)
   that gives the Talker stronger supervision?

---

## Next steps (in order)

1. **Fix T2A code alignment** — port MiniMind-O's `dataset.py`
   staggered placement into `scripts/omnivox_t2a_train.py`. This is
   the single biggest unblock.

2. **Fix Talker checkpoint saving** — switch from `np.savez` to
   safetensors or split shards; verify the file actually loads back.

3. **Swap backbone to Sarvam-1** — `mlx-community/sarvam-1-4bit`.
   Update `OmniVoxConfig` (hidden_size, num_layers, vocab, audio_pad
   token ID for the Sarvam tokenizer). Rerun Phase 1a.

4. **Try MiniMind-O Talker warm-start** — load `pytorch_model.bin`
   from the cached MiniMind-3O snapshot, copy the Talker weights into
   our `TalkerModule`. T2A then becomes "fine-tune Chinese Talker to
   Hindi" instead of "train from scratch with 1900 samples".

5. **Write an inference script** — `scripts/omnivox_infer.py` that
   takes a WAV, runs the full A2A path, and decodes Mimi codes back
   to audio. Needed to actually evaluate the model qualitatively.

6. **Scale data only if 1+4 don't unlock learning** — if T2A still
   doesn't converge with correct alignment and warm-start, the issue
   is data, and we go pull more (FLEURS Hi full train, IndicTTS,
   IndicSUPERB, custom scrape).

---

## File layout

```
models/omnivox/
├── README.md          ← this file
├── __init__.py
├── config.py          ← OmniVoxConfig + load_omnivox_config(yaml)
└── model.py           ← OmniVox nn.Module (loads backbone, wires Talker)

configs/omnivox/
├── whisper_small_phase1a.yaml
├── whisper_small_phase1b.yaml
└── whisper_small_t2a.yaml

scripts/
├── omnivox_prepare_t2a.py     ← JSONL + Mimi → T2A parquet
├── omnivox_t2a_train.py       ← T2A trainer (NEEDS ALIGNMENT FIX)
├── omnivox_phase1a_train.py   ← projector-only
└── omnivox_phase1b_train.py   ← projector + top-N backbone
```

## References

- MiniMind-O (upstream): https://github.com/jingyaogong/minimind-o
- Sarvam-1: https://hf.co/sarvamai/sarvam-1
- Sarvam-1 MLX 4-bit: https://hf.co/mlx-community/sarvam-1-4bit
- Mimi codec: kyutai/moshiko-mlx-bf16
- Whisper-small MLX: mlx-community/whisper-small-mlx
