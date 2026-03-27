# mlx-audio-train

A **LoRA / QLoRA finetuning pipeline** for MLX TTS models on Apple Silicon.
Supports **Qwen3-TTS** and **CSM**, with two distinct training strategies.

---

## Two Training Pipelines

### Pipeline 1 — Language Adaptation (LoRA)

**Use this when:** you want the model to speak a new language or accent (e.g. Hindi).
Voice identity still comes from a reference audio clip at inference — you are teaching the *language*, not baking in a specific voice.

```bash
python scripts/train.py --config configs/qwen3_tts_hindi.yaml
```

**How it works:**
- Applies LoRA adapters to all attention + MLP layers inside the `talker` transformer
- Loss: `main_codec_loss + 0.3 × sub_talker_loss` (teacher-forced codec prediction)
- Does **not** use `ref_audio` during training — speaker identity is irrelevant
- After training: load the adapter at inference with any ref audio to clone any voice in the new language

**Good for:** Hindi, regional languages, new accents, domain-specific speech styles

---

### Pipeline 2 — Speaker Voice Cloning (LoRA + Speaker Embedding)

**Use this when:** you want to permanently bake a specific person's voice into the model so it speaks as that person without needing a reference clip at inference.

```bash
# Step 1 — train
python scripts/train.py --config configs/qwen3_tts_speaker.yaml

# Step 2 — bake the speaker identity into codec_embedding[3000]
python scripts/bake_speaker_embedding.py \
    --config     configs/qwen3_tts_speaker.yaml \
    --checkpoint checkpoints/qwen3-speaker/checkpoint-final \
    --output     checkpoints/qwen3-speaker/custom_voice_model
```

**How it works (mirrors official `sft_12hz.py`):**
1. Every training sample must have a `ref_audio` field pointing to a clip of the **target speaker**
2. During each forward pass, `model.speaker_encoder` extracts a speaker embedding from the ref audio mel spectrogram
3. The speaker embedding is injected into the codec embedding sequence as additive conditioning — all 16 VQ codec levels feed back as input context
4. LoRA adapters learn to generate codec tokens conditioned on that speaker identity
5. After training, `bake_speaker_embedding.py` averages the speaker embeddings across the training set and writes the mean into `talker.model.codec_embedding.weight[3000]` — the reserved custom-voice slot — then patches `config.json` with `tts_model_type: custom_voice`

**Good for:** cloning a single specific voice, podcast/audiobook voice replication, personal TTS

---

## Choosing Between the Two

| | Language Adaptation | Speaker Voice Cloning |
|---|---|---|
| **Config** | `qwen3_tts_hindi.yaml` | `qwen3_tts_speaker.yaml` |
| **model_type** | `qwen3_tts` | `qwen3_tts_speaker` |
| **ref_audio in data** | Not required | Required for every sample |
| **Epochs** | 10–15 | 3 (more risks overfitting) |
| **LoRA rank** | 8 | 16 |
| **Effective batch** | 32 | 16 |
| **Post-training step** | None | `bake_speaker_embedding.py` |
| **At inference** | Needs ref audio to pick a voice | Works without ref audio |

You can **combine both**: train with the language config first, then run the speaker config on top using the language-adapted checkpoint as the starting point.

---

## Quick Start

```bash
# Install dependencies
pip install mlx-audio soundfile scipy datasets transformers gradio pyyaml safetensors

# Verify pipeline works (no data needed)
python scripts/train.py --config configs/qwen3_tts_hindi.yaml --smoke-test
```

### Pipeline 1 — Hindi Language Adaptation

```bash
# 1. Download Hindi dataset
python scripts/prepare_hindi_dataset.py --source hf --output data/hindi

# 2. Pre-tokenize audio to codec IDs (run once — saves .codec.npy files)
python scripts/preprocess_dataset.py --input data/hindi/train.jsonl data/hindi/val.jsonl

# 3. Train
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# 4. Demo — compare before/after with a reference voice
python scripts/demo.py --adapter checkpoints/qwen3-hindi/checkpoint-best
```

### Pipeline 2 — Speaker Voice Cloning

Your JSONL must have `ref_audio` on every line, all pointing to the same target speaker:

```json
{"audio": "data/speaker/clip1.wav", "text": "Hello world", "ref_audio": "data/speaker/ref.wav"}
{"audio": "data/speaker/clip2.wav", "text": "How are you", "ref_audio": "data/speaker/ref.wav"}
```

```bash
# 1. Pre-tokenize audio
python scripts/preprocess_dataset.py \
    --input data/speaker/train.jsonl data/speaker/val.jsonl

# 2. Train (3 epochs recommended)
python scripts/train.py --config configs/qwen3_tts_speaker.yaml

# 3. Bake speaker embedding into the model
python scripts/bake_speaker_embedding.py \
    --config     configs/qwen3_tts_speaker.yaml \
    --checkpoint checkpoints/qwen3-speaker/checkpoint-final \
    --output     checkpoints/qwen3-speaker/custom_voice_model

# 4. Use the baked model at inference (no ref_audio needed)
#    The output dir contains adapters.safetensors + speaker_embedding.npy
```

---

## Project Structure

```
mlx-audio-train/
│
├── scripts/
│   ├── train.py                    # Main finetuning entry point
│   ├── bake_speaker_embedding.py   # Post-training: write speaker → codec_embedding[3000]
│   ├── preprocess_dataset.py       # Pre-tokenize audio → .codec.npy files
│   ├── prepare_hindi_dataset.py    # Download & format Hindi TTS data
│   └── demo.py                     # Gradio voice-cloning demo
│
├── configs/
│   ├── qwen3_tts_hindi.yaml        # Pipeline 1: language adaptation
│   └── qwen3_tts_speaker.yaml      # Pipeline 2: speaker voice cloning
│
├── train/
│   ├── lora.py                     # LoRA / QLoRA layer implementations
│   ├── trainer.py                  # Training loop (AdamW, grad accum, checkpointing)
│   └── losses/
│       └── codec_loss.py           # qwen3_tts_loss, qwen3_tts_speaker_loss, csm_loss
│
├── data/
│   ├── audio_utils.py              # Audio I/O, resampling, mel spectrogram
│   ├── base_dataset.py             # JSONL dataset + BatchIterator (prefetch, length-sort)
│   └── processors/
│       ├── qwen3_tts.py            # audio→codec tokens, text→IDs, ref_mel extraction
│       └── csm.py                  # CSM: Mimi RVQ codes + LLaMA tokenizer
│
└── checkpoints/                    # Saved LoRA adapters
```

---

## Training Scripts Reference

### `scripts/train.py`

```bash
python scripts/train.py --config CONFIG [OPTIONS]

Options:
  --smoke-test        Run 5 steps with dummy data (no dataset needed)
  --resume PATH       Resume from a checkpoint directory
  --lora-rank N       Override LoRA rank
  --lr FLOAT          Override learning rate
  --epochs N          Override num_epochs
  --max-steps N       Stop after N optimizer steps
```

### `scripts/bake_speaker_embedding.py`  *(Pipeline 2 only)*

```bash
python scripts/bake_speaker_embedding.py \
    --config     CONFIG_YAML        \
    --checkpoint CKPT_DIR           \
    --output     OUTPUT_DIR         \
    [--slot N]                      \ # codec_embedding row to use (default 3000)
    [--fuse-lora]                     # merge LoRA into base weights
```

### `scripts/preprocess_dataset.py`

Pre-tokenizes audio to `.codec.npy` files once, so training skips the speech tokenizer entirely (major speedup + avoids OOM).

```bash
python scripts/preprocess_dataset.py --input data/hindi/train.jsonl [data/hindi/val.jsonl ...]
```

---

## Core Library

### `train/lora.py`

- **`LoRALinear`** — wraps `nn.Linear` (full-precision base)
- **`QLoRALinear`** — wraps `nn.QuantizedLinear` (quantized base + bf16 LoRA delta)
- **`apply_lora(model, config)`** — recursive in-place patching; scoped to `talker` for Qwen3-TTS
- **`get_trainable_params(model)`** — returns only `lora_a`/`lora_b` tensors (avoids 40+ GB gradient memory)
- **`save_adapters / load_adapters`** — tiny safetensors checkpoint (~46 MB for rank-8)

### `train/trainer.py`

- AdamW + gradient accumulation + gradient norm clipping
- Cosine / linear / constant LR schedule with warmup
- Per-step and per-epoch checkpointing (LoRA adapters only)
- Batched gradient norm computation (single GPU sync per optimizer step)

### `train/losses/codec_loss.py`

| Function | Pipeline | Description |
|---|---|---|
| `qwen3_tts_loss` | Language Adaptation | `main_loss + 0.3 × sub_talker_loss`; no speaker conditioning |
| `qwen3_tts_speaker_loss` | Speaker Cloning | Same loss + speaker embedding injected from `ref_mel`; all 16 codec levels as input context |
| `csm_loss` | CSM | Cross-entropy on first codebook head |

### `data/audio_utils.py`

No PyTorch dependency — uses `soundfile` + `scipy.signal`.

- `load_audio(path, target_sr)` — load + resample any audio file
- `normalize_loudness(audio)` — RMS normalization to −23 dBFS
- `trim_silence(audio)` — energy-based silence trimming
- `mel_spectrogram(audio, sr)` — log-mel spectrogram matching Qwen3-TTS speaker encoder params (n_fft=1024, n_mels=128, hop=256)

### `data/base_dataset.py — BatchIterator`

- `sort_by_length=True` — groups similar-length sequences to minimise padding waste
- `prefetch=2` — loads the next N batches in a background thread, overlapping CPU I/O with GPU compute

---

## Config Reference

### Language Adaptation (`configs/qwen3_tts_hindi.yaml`)

| Key | Value | Notes |
|-----|-------|-------|
| `model.model_type` | `qwen3_tts` | |
| `lora.rank` | `8` | |
| `trainer.num_epochs` | `15` | |
| `trainer.batch_size` | `2` | M-series Metal pressure |
| `trainer.grad_accumulation` | `16` | effective batch = 32 |
| `trainer.learning_rate` | `2e-5` | |
| `trainer.label_smoothing` | `0.1` | helps with character diversity |
| `processor.include_ref_mel` | `false` | not needed for language training |

### Speaker Voice Cloning (`configs/qwen3_tts_speaker.yaml`)

| Key | Value | Notes |
|-----|-------|-------|
| `model.model_type` | `qwen3_tts_speaker` | enables speaker loss |
| `lora.rank` | `16` | more capacity for speaker detail |
| `trainer.num_epochs` | `3` | more epochs → overfitting risk |
| `trainer.grad_accumulation` | `8` | effective batch = 16 |
| `trainer.learning_rate` | `2e-5` | matches official |
| `trainer.label_smoothing` | `0.0` | sharp speaker predictions |
| `processor.include_ref_mel` | `true` | enables mel → speaker_encoder path |
| `processor.speaker_name` | `custom_speaker` | written into config.json by bake script |

---

## Supported Models

| Model | `model_type` | Status |
|-------|-------------|--------|
| Qwen3-TTS 0.6B Base 8bit | `qwen3_tts` / `qwen3_tts_speaker` | Working |
| Qwen3-TTS 1.7B Base 8bit | `qwen3_tts` / `qwen3_tts_speaker` | Same pipeline, change `model_id` |
| CSM (Sesame) | `csm` | Processor + loss implemented |
| Kokoro | — | Planned |
| Chatterbox | — | Planned |

---

## Comparison with Official Qwen3-TTS Finetuning

The [official sft_12hz.py](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) is full fine-tuning on CUDA with PyTorch + HuggingFace Accelerate.
This repo is LoRA / QLoRA on Apple Silicon with MLX.

| | Official | This repo |
|---|---|---|
| Framework | PyTorch + Accelerate | MLX |
| Training mode | Full fine-tune (all weights) | LoRA (1–2% of weights) |
| Precision | bfloat16 | 8-bit quantised base + bf16 LoRA delta |
| Effective batch | 8 (bs=2 × accum=4) | 16–32 |
| Epochs | 3 | 3 (speaker) / 10–15 (language) |
| Speaker injection | Position 6 in dual-channel format | Broadcast over codec sequence |
| All 16 codec levels | Yes (additive input embeddings) | Yes (via `code_predictor` when available) |
| Post-training step | Bake speaker to `codec_embedding[3000]` | `bake_speaker_embedding.py` (same) |
| Checkpoint size | Full model (~1.2 GB) | Adapters only (~46 MB) |
