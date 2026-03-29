# Model Card — Qwen3-TTS Indian Languages (MLX LoRA)

## Model Summary

LoRA fine-tuned adapters for [Qwen3-TTS-0.6B](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) targeting Indian language speech synthesis, trained entirely on Apple Silicon via the [MLX](https://github.com/ml-explore/mlx) framework.

Two training pipelines are released:

| Pipeline | Purpose | Adapter size | Ref audio at inference? |
|----------|---------|-------------|------------------------|
| **Pipeline 1 — Language Adaptation** | Teaches the model a new language/accent | 23 MB (rank-8) | Yes — any voice |
| **Pipeline 2 — Speaker Voice Cloning** | Bakes a specific speaker's identity into the model | 45 MB (rank-16) | No |

---

## Base Model

| Field | Value |
|-------|-------|
| Base model | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` |
| Architecture | Qwen3-TTS (LLM talker + speech tokenizer + speaker encoder) |
| Codec | 12Hz discrete codec (VQ-based) |
| Parameters | 366M total, ~12–13M trainable (LoRA only) |

---

## Supported Languages

New language token IDs added to the codec embedding table (slots were reserved but unused in the base model):

| Language  | Code | Token ID | Status |
|-----------|------|----------|--------|
| Hindi     | `hi` | 2051 | Trained |
| Tamil     | `ta` | 2052 | Planned |
| Telugu    | `te` | 2056 | Planned |
| Bengali   | `bn` | 2057 | Planned |
| Kannada   | `kn` | 2059 | Planned |
| Malayalam | `ml` | 2060 | Planned |
| Marathi   | `mr` | 2062 | Planned |
| Gujarati  | `gu` | 2063 | Planned |
| Punjabi   | `pa` | 2065 | Planned |

The base model natively supports: English (2050), Chinese (2055), German (2053), Spanish (2054), French (2061), Italian (2070), Portuguese (2071), Japanese (2058), Korean (2064), Russian (2069).

---

## Training Details

### Architecture — LoRA

| Parameter | Pipeline 1 | Pipeline 2 |
|-----------|-----------|-----------|
| Rank | 8 | 16 |
| Alpha | 16.0 | 32.0 |
| Dropout | 0.05 | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj | q/k/v/o/gate/up/down_proj |
| Scope | `talker` (avoids speech tokenizer) | `talker` |
| Trainable params | ~11.9M (3.24%) | ~13.3M (3.62%) |

### Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| LR schedule | Cosine decay |
| Warmup steps | 50 (P1) / 20 (P2) |
| Grad clip | 1.0 |
| Effective batch size | 32 (P1: batch=2 × accum=16) / 16 (P2: batch=2 × accum=8) |

### Training Duration (Hindi)

| Pipeline | Epochs | Steps | Final val loss | Training time |
|----------|--------|-------|----------------|---------------|
| P1 — Language | 15 | ~655+ | ~7.55 | ~4641s |
| P2 — Speaker | 3 | 354 | 7.54 | ~2828s |

*Training hardware: Apple Silicon M-series (MLX, GPU backend)*

---

## Datasets

### Hindi (current)

| Dataset | Hours | Speakers | License | Source |
|---------|-------|----------|---------|--------|
| Custom Hindi TTS corpus | ~2.6h (1900 train / 100 val samples) | 1 | — | Private |

### Planned — Open-Source Indian Language Datasets

| Language | Dataset | Hours | Speakers | License |
|----------|---------|-------|----------|---------|
| All 9 languages | IndicVoices-R (ai4bharat) | 1,704 total | Multi | CC-BY-4.0 |
| Hindi | IndicTTS-Hindi (SPRINGLab) | 10.3 | 2 (1M+1F) | IndicTTS* |
| Tamil | Mozilla Common Voice | 220+ | Multi | CC-0 |
| Tamil | IndicTTS-Tamil | ~10 | 2 | IndicTTS* |
| Telugu | OpenSLR/66 | ~10 | Multi | CC-BY-SA-4.0 |
| Bengali | OpenSLR/37 | ~20 | Multi | CC-BY-SA-4.0 |
| Kannada | OpenSLR/63 | ~10 | 24 | CC-BY-SA-4.0 |
| Malayalam | OpenSLR/63 | ~10 | Multi | CC-BY-SA-4.0 |
| Marathi | OpenSLR/64 | ~99 | 31 | CC-BY-SA-4.0 |
| Gujarati | OpenSLR/78 | ~51 | 204 | CC-BY-SA-4.0 |
| Punjabi | IndicVoices-R | 9–175 | Multi | CC-BY-4.0 |

*IndicTTS: requires accepting license at https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf

---

## Loss Function

```
total_loss = main_codec_loss + 0.3 × sub_talker_loss
```

- **main_codec_loss**: masked cross-entropy on codec token predictions from the talker transformer
- **sub_talker_loss**: masked cross-entropy on the 16-codebook code predictor auxiliary head
- Padding positions are masked out (not counted in loss)
- Pipeline 1: label_smoothing=0.1 (language diversity)
- Pipeline 2: label_smoothing=0.0 (sharp speaker identity)

---

## Usage

### Inference with Pipeline 1 adapter (language adaptation)

```python
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio
from train.lora import apply_lora, load_adapters, LoRAConfig

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
apply_lora(model, LoRAConfig(model_type="qwen3_tts", rank=8))
load_adapters(model, "checkpoints/qwen3-hindi/checkpoint-best/adapters.safetensors")

generate_audio(
    text="नमस्ते! आज का दिन बहुत अच्छा है।",
    model=model,
    output_path="./output",
    lang_code="hi",
    ref_audio="path/to/reference_speaker.wav",
)
```

### Demo UI

```bash
python scripts/demo.py  # launches Gradio at localhost:7860
```

Select a checkpoint from the dropdown to compare base vs adapted models.

---

## Limitations

- Current Hindi model trained on ~2.6 hours from a single speaker — limited speaker diversity
- May produce unnatural prosody on text mixing Hindi and English (code-switching)
- Not production-grade; intended for research and further fine-tuning
- Model may hallucinate or skip tokens on very long inputs (>10 seconds equivalent)
- Pipeline 2 voice cloning quality depends on reference audio quality (5–10s, clean, no background noise)

---

## Reproducing Training

```bash
# Install dependencies
pip install mlx-audio soundfile scipy datasets transformers gradio pyyaml safetensors

# Pre-tokenize dataset (run once)
python scripts/preprocess_dataset.py \
  --input data/hindi/train.jsonl data/hindi/val.jsonl \
  --model-id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit

# Stamp language code
python scripts/add_lang_code.py --input data/hindi/train_codes.jsonl --lang hi

# Train Pipeline 1 (language adaptation)
python scripts/train.py --config configs/qwen3_tts_hindi.yaml

# Train Pipeline 2 (speaker voice cloning)
python scripts/train.py --config configs/qwen3_tts_speaker.yaml
```

Loss curves are saved as JSONL at `checkpoints/*/train_log.jsonl`.

---

## License

- **Code**: Apache 2.0
- **Model weights (adapters)**: Apache 2.0 (derived from base model weights — check Qwen3-TTS license)
- **Training data**: see dataset licenses above; IndicVoices-R (CC-BY-4.0) and OpenSLR datasets (CC-BY-SA-4.0) are fully open

---

## Citation

If you use this work, please cite the base model:

```bibtex
@misc{qwen3tts2025,
  title  = {Qwen3-TTS: A Text-to-Speech Model},
  author = {Qwen Team},
  year   = {2025},
  url    = {https://huggingface.co/Qwen/Qwen3-TTS}
}
```
