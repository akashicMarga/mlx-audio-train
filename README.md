# mlx-audio-train

A general-purpose **LoRA / QLoRA finetuning pipeline** for MLX TTS models on Apple Silicon. Currently supports **Qwen3-TTS** and **CSM**, with Hindi finetuning as the primary use case.

---

## Project Structure

```
mlx-audio-train/
│
├── scripts/                  # Entry-point scripts you actually run
│   ├── train.py              # Main finetuning script
│   ├── demo.py               # Gradio voice cloning demo
│   └── prepare_hindi_dataset.py  # Dataset download & preparation
│
├── configs/                  # YAML training configs
│   └── qwen3_tts_hindi.yaml  # Qwen3-TTS 0.6B — Hindi LoRA config
│
├── train/                    # Core training library
│   ├── lora.py               # LoRA / QLoRA layer implementations
│   ├── trainer.py            # Training loop (AdamW, grad accum, checkpointing)
│   └── losses/
│       └── codec_loss.py     # Model-specific loss functions
│
├── data/                     # Data loading & processing
│   ├── audio_utils.py        # Audio I/O, resampling, loudness, silence trimming
│   ├── base_dataset.py       # Universal JSONL dataset + batch iterator
│   └── processors/
│       ├── qwen3_tts.py      # Qwen3-TTS: audio→codec tokens, text→token IDs
│       └── csm.py            # CSM: audio→Mimi RVQ codes, text→LLaMA token IDs
│
└── checkpoints/              # Saved LoRA adapters (created during training)
    └── qwen3-hindi/
        ├── checkpoint-best/
        ├── checkpoint-step_XXXXXXX/
        └── train_log.jsonl
```

---

## Scripts

### `scripts/train.py` — Finetuning entry point

The main script. Reads a YAML config, loads the model, applies LoRA, then trains.

```bash
# Full Hindi finetuning
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# Quick smoke test — 5 steps with dummy data, no dataset needed
python scripts/train.py --config configs/qwen3_tts_hindi.yaml --smoke-test

# Resume from a checkpoint
python scripts/train.py --config configs/qwen3_tts_hindi.yaml \
    --resume checkpoints/qwen3-hindi/checkpoint-step_0000200

# Override config values from CLI
python scripts/train.py --config configs/qwen3_tts_hindi.yaml \
    --lora-rank 16 --lr 1e-4 --epochs 5 --max-steps 500
```

**What it does:**
1. Loads model via `mlx-audio` (`mlx_audio.tts.utils.load_model`)
2. Patches matching layers with LoRA/QLoRA adapters (231 layers for Qwen3-TTS 8bit)
3. Builds dataset from JSONL files using the model-specific processor
4. Runs the training loop via `Trainer` — saves only adapter weights, not the full model

---

### `scripts/demo.py` — Gradio Voice Cloning Demo

Side-by-side comparison of Base (random voice) vs Cloned (your reference voice).

```bash
# Basic launch
python scripts/demo.py

# With a finetuned Hindi adapter
python scripts/demo.py --adapter checkpoints/qwen3-hindi/checkpoint-best

# Custom port
python scripts/demo.py --port 7860 --share
```

**Modes:**
| Mode | When | Model used |
|------|------|-----------|
| Base | Always (left output) | `Qwen3-TTS-12Hz-0.6B-Base-8bit` |
| Cloned | When ref audio uploaded (right output) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` |

**Inputs:**
- **Text** — what to synthesise (Hindi, English, Hinglish)
- **Reference audio** — 3–10 sec clip to clone the voice from (upload or record)
- **Reference text** — transcript of ref audio (optional, auto-transcribed via Whisper if blank)
- **Adapter path** — point to a finetuned checkpoint to compare quality before/after training
- **Speed / Temperature** — generation controls

---

### `scripts/prepare_hindi_dataset.py` — Dataset preparation

Downloads and formats Hindi TTS data into JSONL files ready for training.

```bash
# From HuggingFace (easiest — cdactvm/indic-tts-hindi, IIT Madras corpus)
python scripts/prepare_hindi_dataset.py --source hf --output data/hindi

# Limit samples for a quick test run
python scripts/prepare_hindi_dataset.py --source hf --output data/hindi --max-samples 500

# From OpenSLR IndicTTS (manual download required)
python scripts/prepare_hindi_dataset.py --source indictts --output data/hindi

# From your own audio folder (WAV + TXT pairs)
python scripts/prepare_hindi_dataset.py --source custom \
    --audio-dir /path/to/wavs --output data/hindi
```

**Output:** `data/hindi/train.jsonl` and `data/hindi/val.jsonl` (95/5 split)

Each line in the JSONL:
```json
{"audio": "data/hindi/audio/hi_00001.wav", "text": "नमस्ते दुनिया"}
```

---

## Core Library

### `train/lora.py` — LoRA / QLoRA

Patches any MLX model's Linear layers with low-rank adapters in-place.

- **`LoRALinear`** — wraps `nn.Linear` (full-precision base weights)
- **`QLoRALinear`** — wraps `nn.QuantizedLinear` (quantized base, trainable A/B in bf16)
- **`apply_lora(model, config)`** — recursively patches matching layer names; scopes to `talker` submodule for Qwen3-TTS to avoid touching the speech tokenizer
- **`get_trainable_params(model)`** — returns only `lora_a`/`lora_b` tensors for the optimizer
- **`save_adapters / load_adapters`** — saves/loads only the tiny adapter weights (not the full model)

Key design choice: base weights are **not** frozen with `stop_gradient` (would block gradient flow to `lora_a`). Freezing is enforced by passing only LoRA params to the optimizer.

```python
from train.lora import apply_lora, LoRAConfig
apply_lora(model, LoRAConfig(rank=8, alpha=16, model_type="qwen3_tts"))
```

### `train/trainer.py` — Training loop

Model-agnostic trainer. Plug in any model + loss function.

- AdamW optimizer with linear warmup + cosine/linear/constant LR decay
- Gradient accumulation (default: 4 steps → effective batch = `batch_size × grad_accum`)
- Global gradient norm clipping
- Per-step and per-epoch checkpointing (saves only LoRA adapters)
- Best-model tracking based on validation loss
- JSON log of all metrics

```python
from train.trainer import Trainer, TrainerConfig
trainer = Trainer(TrainerConfig(output_dir="./checkpoints", num_epochs=10, ...))
trainer.train(model, train_loader, loss_fn, val_loader)
```

### `train/losses/codec_loss.py` — Loss functions

- **`qwen3_tts_loss`** — Teacher-forced dual loss matching the official Qwen3-TTS training:
  ```
  loss = main_talker_loss + 0.3 × sub_talker_loss
  ```
  Input sequence: `[text_embeds | codec_embeds[:-1]]` → predict `codec_ids[1:]`

- **`csm_loss`** — Cross-entropy loss on the first codebook head for CSM

### `data/base_dataset.py` — Universal dataset

Reads JSONL, loads audio, applies the model-specific processor, and batches.

```json
{"audio": "path/to/clip.wav", "text": "transcript", "ref_audio": "optional/ref.wav"}
```

### `data/processors/qwen3_tts.py` — Qwen3-TTS processor

Converts raw audio + text into the tensors the model expects:
- Audio → codec token IDs via `Qwen3TTSSpeechTokenizer` (VQ-VAE at 12Hz)
- Text → token IDs via `AutoTokenizer`
- `collate_qwen3()` — pads to max length, returns `mx.array` batch dict

### `data/audio_utils.py` — Audio utilities

No PyTorch dependency. Uses `soundfile` + `scipy.signal`.
- `load_audio(path, target_sr)` — load + resample
- `normalize_loudness(audio)` — RMS normalization
- `trim_silence(audio)` — energy-based silence trimming
- `validate_audio(audio, sr, min_dur, max_dur)` — duration + sanity checks

---

## Quick Start

```bash
# 1. Install dependencies
pip install mlx-audio soundfile scipy datasets transformers gradio pyyaml

# 2. Smoke test — verify the pipeline works (no data needed)
python scripts/train.py --config configs/qwen3_tts_hindi.yaml --smoke-test

# 3. Download Hindi dataset
python scripts/prepare_hindi_dataset.py --source hf --output data/hindi

# 4. Start training
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# 5. Test the result in the demo
python scripts/demo.py --adapter checkpoints/qwen3-hindi/checkpoint-best
```

---

## Config Reference (`configs/qwen3_tts_hindi.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `model_id` | `Qwen3-TTS-12Hz-0.6B-Base-8bit` | HuggingFace model ID |
| `model` | `model_type` | `qwen3_tts` | Determines processor + loss fn |
| `lora` | `rank` | `8` | LoRA rank (higher = more capacity) |
| `lora` | `alpha` | `16.0` | LoRA scaling factor (`scale = alpha/rank`) |
| `lora` | `dropout` | `0.05` | Dropout on LoRA input |
| `data` | `train_jsonl` | `./data/hindi/train.jsonl` | Training data path |
| `data` | `max_duration` | `15.0` | Skip clips longer than this (seconds) |
| `trainer` | `batch_size` | `4` | Per-step batch size |
| `trainer` | `grad_accumulation` | `8` | Steps before optimizer update (effective batch = 32) |
| `trainer` | `learning_rate` | `2e-4` | Peak LR |
| `trainer` | `warmup_steps` | `50` | Linear LR warmup steps |
| `trainer` | `lr_schedule` | `cosine` | `cosine` / `linear` / `constant` |
| `trainer` | `label_smoothing` | `0.1` | Helps with Hindi character diversity |

---

## Supported Models

| Model | `model_type` | Status |
|-------|-------------|--------|
| Qwen3-TTS 0.6B Base 8bit | `qwen3_tts` | ✅ Working (231 LoRA layers, 1.12% trainable) |
| Qwen3-TTS 1.7B Base 8bit | `qwen3_tts` | ✅ Same pipeline, change `model_id` in config |
| CSM (Sesame) | `csm` | ✅ Processor + loss implemented |
| Kokoro | — | Planned |
| Chatterbox | — | Planned |

---

## How Voice Cloning Works (Qwen3-TTS)

Qwen3-TTS has three modes controlled by which model checkpoint is loaded:

1. **Base** — No speaker conditioning → random voice every generation
2. **CustomVoice** — Takes a 3–10 sec reference audio clip → extracts a 256-dim speaker embedding via the built-in speaker encoder → injects it into the LM context → output speech matches the reference speaker's voice
3. **VoiceDesign** — Describe the voice in text (e.g. "a young woman with a calm tone")

The `demo.py` uses modes 1 and 2 side-by-side. After Hindi finetuning, the cloned voice should speak Hindi more naturally while still sounding like the reference speaker.
