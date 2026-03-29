# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-based LoRA/QLoRA finetuning pipeline for TTS models on Apple Silicon. Two distinct training workflows:

1. **Language Adaptation (Pipeline 1)**: Teaches model new language/accent via LoRA — no ref_audio needed at inference, voice identity comes from ref_audio at generation time.
2. **Speaker Voice Cloning (Pipeline 2)**: Permanently bakes a specific voice into the model via LoRA + speaker embedding injection — no ref_audio needed at inference.

Supported models: Qwen3-TTS 0.6B/1.7B (working), CSM/Sesame (implemented), Kokoro/Chatterbox (planned).

## Common Commands

```bash
# Install dependencies
pip install mlx-audio soundfile scipy datasets transformers gradio pyyaml safetensors

# Pre-tokenize dataset (run once before training — avoids OOM and speeds up training)
# Output is auto-written to data/train_codes.jsonl (same dir, _codes suffix)
python scripts/preprocess_dataset.py \
  --input data/train.jsonl \
  --model-id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit

# Train (Pipeline 1: language adaptation)
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# Train (Pipeline 2: speaker voice cloning)
python scripts/train.py --config configs/qwen3_tts_speaker.yaml

# Post-training: bake speaker embedding into model (Pipeline 2 only)
python scripts/bake_speaker_embedding.py \
  --config   configs/qwen3_tts_speaker.yaml \
  --checkpoint checkpoints/qwen3-speaker/checkpoint-final \
  --output   checkpoints/qwen3-speaker/custom_voice_model

# Monitor training
python scripts/watch_training.py --log checkpoints/qwen3-hindi/train_log.jsonl

# Demo UI
python scripts/demo.py --adapter checkpoints/qwen3-speaker/custom_voice_model
```

## Architecture

```
scripts/train.py          # Entry point: loads config, applies LoRA, runs training
train/trainer.py          # Training loop: AdamW, grad accumulation, LR schedule, checkpointing
train/lora.py             # LoRA/QLoRA: LoRALinear, QLoRALinear, recursive model patching
train/losses/codec_loss.py # Loss functions: masked cross-entropy on codec token prediction
data/base_dataset.py      # JSONL dataset loader, TTSSample, BatchIterator
data/audio_utils.py       # Audio I/O, resampling, loudness norm, silence trimming
data/processors/qwen3_tts.py  # Qwen3-TTS tokenization: audio→codec_ids, text→text_ids
data/processors/csm.py        # CSM/Sesame processor
```

### Data Flow

1. JSONL dataset (`{"audio": "...", "text": "...", "ref_audio": "..."}`)
2. `base_dataset.py` loads/resamples audio, validates duration
3. `processors/qwen3_tts.py` tokenizes: audio → codec_ids (12Hz), text → text_ids, ref_audio → mel spectrogram
4. `BatchIterator` length-sorts and prefetches batches
5. `codec_loss.py` runs forward pass: text+codec embeds → talker transformer → logits → cross-entropy
6. Pipeline 2 only: speaker_encoder extracts embedding from ref_mel, injected into codec embeddings
7. `trainer.py` accumulates gradients over LoRA params only, clips, steps AdamW

### LoRA Implementation

- Only `lora_a`/`lora_b` matrices are trainable — extracted via `get_trainable_params()` in `lora.py`
- `QLoRALinear`: 8-bit quantized base weights + bf16 LoRA delta (memory-efficient)
- LoRA is applied recursively to attention + MLP layers specified in config under `lora_target_modules`
- Checkpoints save only LoRA adapters as safetensors (~46 MB for rank-8), not the full model

### Loss Functions (codec_loss.py)

- **Language adaptation**: `loss = main_codec_loss + 0.3 × sub_talker_loss`
- **Speaker cloning**: Same structure but speaker_embed is added to codec embeddings before forward pass
- Both use masked cross-entropy (ignoring padding tokens)

### Gradient Management

`trainer.py` uses a custom `value_and_grad_fn` with `_strip_empty()` to handle frozen submodules (speech_tokenizer has `gc_func` that triggers graph traversal issues). Gradient norm is computed in a single batched operation across all LoRA layers.

### Pre-tokenization Strategy

Running `preprocess_dataset.py` before training saves codec IDs to `.codec.npy` files alongside audio. The processor then loads these instead of running speech_tokenizer during training — eliminates the main source of OOM errors and dramatically speeds up training.

## Config Parameters

Key YAML fields that affect behavior:

| Field | Purpose |
|-------|---------|
| `include_ref_mel` | `true` = Pipeline 2 (speaker cloning), `false` = Pipeline 1 |
| `lora_rank` | LoRA capacity: 8 for language, 16 for speaker |
| `label_smoothing` | 0.1 for language diversity, 0.0 for sharp speaker predictions |
| `grad_accumulation` | Effective batch = `batch_size × grad_accumulation` |
| `max_codec_length` | Max codec sequence length in tokens (12Hz → ~100s at 1200) |

## Dataset Format

JSONL with one sample per line:
```json
{"audio": "path/to/file.wav", "text": "transcription", "ref_audio": "path/to/ref.wav"}
```

- `ref_audio` is required for Pipeline 2, optional for Pipeline 1
- Audio preprocessing: resampled to 24kHz, loudness-normalized to -23 dBFS, silence-trimmed
- After `preprocess_dataset.py`: codec_path field added pointing to `.codec.npy`
