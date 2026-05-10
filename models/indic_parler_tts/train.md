# Indic Parler TTS — Finetuning Plans

Two experiments: **Plan A** adds a new speaker to the model's existing 69, **Plan B** adds a new language (Bhojpuri) that the model has never seen.

Both use the **SYSPIN corpus** (CC-BY-4.0, 48kHz studio quality).  
Download: https://spiredatasets.ee.iisc.ac.in/syspincorpus

---

## Architecture Recap (relevant to training)

```
description text  →  T5 encoder (frozen)  →  enc_hidden [B, T_desc, 1024]
prompt text       →  embed_prompts (frozen) →  prompt_emb [B, T_prompt, 1024]
                                                      ↓
                          ParlerDecoder  ←  cross-attends to enc_hidden
                                                      ↓
                      9-codebook tokens (delay pattern, 86Hz)
                                                      ↓
                          DAC decoder (frozen)  →  audio @ 44100Hz
```

**Where speaker identity lives**: the decoder's cross-attention weights. When the decoder sees the T5 encoding of `"Divya speaks..."`, cross-attention has learned to route audio token predictions toward Divya's voice distribution. Adding a new speaker = teaching cross-attention one new description→voice mapping via LoRA.

**Where language phoneme patterns live**: decoder self-attention and MLP layers. These control the sequence of codec tokens that encode phoneme shapes. New language adaptation = LoRA on self-attention + FFN.

---

## What Needs to Be Built (not yet in this repo)

The existing pipeline targets Qwen3-TTS (12Hz SNAC codec). For Parler TTS you need:

| Component | File to create | Notes |
|-----------|---------------|-------|
| DAC encoder preprocessor | `data/processors/indic_parler.py` | Use `mlx_audio.codec.models.descript.dac.DAC.encode()` — DAC at 86Hz, 9 codebooks |
| Delay-pattern loss | `train/losses/parler_loss.py` | 9-codebook staggered CE loss matching `model.py` generation pattern |
| Description injection | extend `train/trainer.py` | Pass fixed/per-clip description through frozen T5 encoder at each step |
| LoRA target mapping | `configs/indic_parler_*.yaml` | Map module names to Parler's `cross_attn.q`, `self_attn.q`, `fc1`, `fc2` |

---

## Plan A — Speaker Adaptation (SYSPIN Hindi Female)

**Goal**: Add a new speaker (SYSPIN Hindi Female studio artist) as speaker 70.  
At inference, her voice is produced by a fixed description string — no ref audio needed.

### Dataset

- **Source**: SYSPIN corpus, Hindi Female speaker
- **Subset**: Take 3–5h (full 20h risks overfitting to one voice)
- **Filter**: Keep utterances 2–12s; discard very short/long clips
- **Quality**: 48kHz/24-bit studio, no background noise

### Step 1: Generate Accurate Speaker Description

Run `dataspeech` on all clips to measure actual audio properties:

```bash
pip install dataspeech

python -m dataspeech.main \
    --dataset_name ./data/syspin_hindi_female \
    --audio_column audio \
    --text_column text \
    --output_dir ./data/syspin_hindi_female_with_desc
```

This measures pitch, speaking rate, SNR, gender per clip. Find the **modal description** — the property combination covering ~80% of clips. Then craft one fixed string:

```
SPEAKER_DESC = (
    "A female speaker delivers Hindi speech at a moderate pace. "
    "The recording is of very high quality, with the speaker's voice "
    "sounding clear and very close up."
)
```

**Why fixed**: a single consistent description string creates a stable T5 encoding → one point in enc_hidden space → one speaker identity in cross-attention. Per-clip descriptions would scatter the T5 encoding and prevent a coherent voice from forming.

**Why accuracy matters**: if the description says "deep baritone" but the audio is high-pitched, the cross-attention gradient is contradictory and the LoRA learns a nonsense mapping. Always derive the description from measured audio properties.

### Step 2: Prepare JSONL

```jsonl
{"audio": "syspin/hi_female_0001.wav", "text": "यह पहला वाक्य है।"}
{"audio": "syspin/hi_female_0002.wav", "text": "आज मौसम बहुत अच्छा है।"}
```

No `description` field — it is hardcoded in the training config and passed to T5 at every step.

### Step 3: Config

```yaml
# configs/indic_parler_speaker.yaml

model:
  model_type: indic_parler_tts
  hf_repo: ai4bharat/indic-parler-tts

speaker_description: >
  A female speaker delivers Hindi speech at a moderate pace.
  The recording is of very high quality, with the speaker's voice
  sounding clear and very close up.

lora:
  rank: 4                         # low rank — protects existing 69 speakers
  alpha: 4                        # scale = alpha/rank = 1.0 (conservative)
  target_modules:
    - cross_attn.q                # cross-attention is the speaker identity pathway
    - cross_attn.k
    - cross_attn.v
    - cross_attn.out
    - self_attn.q                 # self-attention for temporal audio coherence
    - self_attn.k
    - self_attn.v
    - self_attn.out

frozen:
  - text_encoder                  # T5 fully frozen — already handles descriptions
  - embed_prompts                 # prompt tokenizer embedding frozen
  - dac                           # DAC decoder frozen

data:
  train_jsonl: data/syspin_hindi_female/train.jsonl
  val_jsonl:   data/syspin_hindi_female/val.jsonl
  max_audio_length_s: 12.0

trainer:
  learning_rate: 5e-6             # very low — decoder is pretrained, T5 frozen
  num_epochs: 2                   # stop early, don't overfit one voice
  batch_size: 1
  grad_accumulation: 4
  label_smoothing: 0.0            # sharp speaker predictions
  eval_every_n_steps: 50
```

### Step 4: Stopping Criterion

Do NOT stop on loss alone. Every 50 steps run inference on:

1. The new speaker description → quality should improve
2. 3–4 existing speaker descriptions (e.g. `"Divya's voice..."`, `"Rahul speaks..."`) → must NOT degrade

Stop when the new speaker sounds correct AND existing speakers are unaffected. This is the true stopping criterion.

### Catastrophic Forgetting Risk

LoRA at r=4 limits weight updates to 4 directions per layer — existing 69 speakers live in other directions and are largely protected. Risk factors that increase forgetting:

| Risk Factor | Mitigation |
|-------------|-----------|
| High learning rate | Use 5e-6, not 1e-4 |
| Many epochs | Cap at 2 epochs |
| High rank | Keep r=4, not r=16 |
| No replay data | Mix in 200–300 original Hindi samples if available — most effective protection |

---

## Plan B — New Language Adaptation (Bhojpuri)

**Goal**: Teach the model to synthesize Bhojpuri (~80M speakers, completely absent from the model).  
Voice quality and prosody are still controlled by description strings.

### Why Bhojpuri

- Not in indic-parler-tts's 21 supported languages
- Devanagari script (same as Hindi) — T5 tokenizer and prompt tokenizer already have full vocabulary coverage, near-zero OOV risk
- SYSPIN provides 20h female + 20h male at 48kHz studio quality
- Same corpus as Plan A — identical preprocessing pipeline

### Dataset

- **Source**: SYSPIN corpus, Bhojpuri section
- **Subset**: 5–10h (use female speaker; optionally mix male for generalization)
- **Filter**: same as Plan A (2–12s clips)

### Step 0: Bhojpuri Speaker Check (run before any training)

Before touching the model, verify two things: (a) what the pretrained model produces on Bhojpuri text, and (b) which SYSPIN Bhojpuri speakers are in the corpus and which one to train on.

**0a — Baseline inference check**

Run the stock `ai4bharat/indic-parler-tts` on a Bhojpuri sentence with a neutral description. This is your pre-training baseline — record the output file to compare against during training.

```python
from mlx_audio.tts.generate import generate_audio

BHOJPURI_TEST = "रउरा के राम राम। आज हम एही गाँव में रहीला।"  # simple greeting + statement

audio = generate_audio(
    text=BHOJPURI_TEST,
    model_id="mlx-community/indic-parler-tts",
    description="A female speaker delivers speech at a moderate pace. "
                "The recording is of very high quality.",
    output_path="baselines/bhojpuri_pretrained_baseline.wav",
)
```

Listen for: Hindi phonemes substituted for Bhojpuri ones (e.g., /r/ for retroflex, vowel shortening). This tells you how much the model needs to learn vs. already knows.

**0b — SYSPIN Bhojpuri corpus check**

```bash
# After downloading SYSPIN:
ls data/syspin_bhojpuri/
# Expected: separate subdirs per speaker (e.g., BH_F1/, BH_M1/)

# Count clips and total duration per speaker
for spk in data/syspin_bhojpuri/*/; do
  count=$(ls "$spk"*.wav 2>/dev/null | wc -l)
  # soxi -D sums duration in seconds
  dur=$(soxi -D "$spk"*.wav 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s/3600}')
  echo "$(basename $spk): $count clips, ~${dur}h"
done
```

Pick the speaker with the cleanest recordings and most data. Typically `BH_F1` (female speaker 1) is the studio-quality track in SYSPIN. Use that speaker's clips exclusively for training — mixing speakers here defeats the language-learning goal; you want phoneme patterns from one consistent voice.

**0c — T5 tokenizer OOV check**

Bhojpuri uses Devanagari but has some script-unique characters. Verify zero OOV before training:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

BHOJPURI_SAMPLES = [
    "रउरा के राम राम।",
    "ई बहुत नीमन बा।",
    "हम कल जाइब।",
]
for s in BHOJPURI_SAMPLES:
    ids = tok.encode(s)
    decoded = tok.decode(ids, skip_special_tokens=True)
    print(f"{'OK' if decoded.strip() == s.strip() else 'OOV!'}: {s}")
```

If any line prints `OOV!`, the T5 vocabulary is missing that character — stop and investigate before training. In practice Devanagari is fully covered so this should all print `OK`.

---

### Step 1: Generate Per-Clip Descriptions

Unlike Plan A, use **varied per-clip descriptions** from dataspeech output. Do not fix one description.

**Why varied**: you want the model to learn Bhojpuri phoneme patterns independently of any specific voice style. If you fix one description, the model learns a new speaker (who happens to speak Bhojpuri) rather than learning the language. Varied descriptions teach: description controls prosody, Bhojpuri text controls phoneme content.

```bash
python -m dataspeech.main \
    --dataset_name ./data/syspin_bhojpuri \
    --audio_column audio \
    --text_column text \
    --output_dir ./data/syspin_bhojpuri_with_desc
```

### Step 2: Prepare JSONL with Descriptions

```jsonl
{"audio": "bh_001.wav", "text": "भोजपुरी पाठ...", "description": "A female speaker delivers speech at a moderate pace. The recording is very clean."}
{"audio": "bh_002.wav", "text": "दूसरा वाक्य...",  "description": "A female speaker delivers speech at a slightly fast pace. The recording is very clean."}
```

### Step 3: Config

```yaml
# configs/indic_parler_bhojpuri.yaml

model:
  model_type: indic_parler_tts
  hf_repo: ai4bharat/indic-parler-tts

# No speaker_description field — use per-clip descriptions from JSONL

lora:
  rank: 8                         # higher rank — learning a new language needs more capacity
  alpha: 16                       # scale = 2.0, allow larger updates
  target_modules:
    - self_attn.q                 # self-attention drives phoneme sequence learning
    - self_attn.k
    - self_attn.v
    - self_attn.out
    - fc1                         # MLP layers critical for new phoneme distributions
    - fc2
    - cross_attn.q                # keep cross-attn so description conditioning works
    - cross_attn.v

frozen:
  - text_encoder                  # T5 frozen — handles any description string already
  - embed_prompts
  - dac

data:
  train_jsonl: data/syspin_bhojpuri/train.jsonl
  val_jsonl:   data/syspin_bhojpuri/val.jsonl
  max_audio_length_s: 12.0

trainer:
  learning_rate: 1e-5             # slightly higher than speaker experiment
  num_epochs: 3
  batch_size: 1
  grad_accumulation: 4
  label_smoothing: 0.1            # language diversity — not sharp single-voice predictions
  eval_every_n_steps: 100
```

### Step 4: Bhojpuri-Specific Risk — Hindi Accent Bleed

Because Bhojpuri uses Devanagari and is phonologically close to Hindi, the pretrained decoder's strong Hindi priors may cause early training to produce Hindi-accented Bhojpuri. Mitigation:

- Mix 200–300 Hindi training samples into the Bhojpuri data — anchors the decoder while it learns Bhojpuri patterns without catastrophically shifting its Hindi behavior
- Evaluate by listening for distinctive Bhojpuri phonemes (retroflex sounds, vowel length distinctions) vs. Hindi approximations

### Step 5: Stopping Criterion

Every 100 steps run inference on:
1. A Bhojpuri test sentence → listen for correct phoneme patterns
2. A Hindi, Tamil, and Telugu test sentence → must remain intelligible
3. Val loss on Bhojpuri held-out set

---

## Side-by-Side Comparison

| | Plan A: Speaker Adaptation | Plan B: Language Adaptation |
|--|---|---|
| **Dataset** | SYSPIN Hindi Female, 3–5h | SYSPIN Bhojpuri Female, 5–10h |
| **Description** | Fixed per speaker | Varied per clip (dataspeech) |
| **LoRA rank** | r=4 | r=8 |
| **LoRA alpha** | 4 (scale=1.0) | 16 (scale=2.0) |
| **Primary LoRA targets** | cross_attn (speaker pathway) | self_attn + FFN (phoneme pathway) |
| **Learning rate** | 5e-6 | 1e-5 |
| **Epochs** | 1–2 | 3 |
| **label_smoothing** | 0.0 (sharp voice) | 0.1 (language diversity) |
| **Stop criterion** | Existing speakers don't degrade | Bhojpuri phonemes correct, others intact |
| **Main risk** | Forgetting existing 69 speakers | Hindi accent bleeding into Bhojpuri |

---

## Inference After Training

### Plan A — New Speaker
```python
audio = generate(
    model, tokenizers,
    description="A female speaker delivers Hindi speech at a moderate pace. "
                "The recording is of very high quality, with the speaker's voice "
                "sounding clear and very close up.",   # exact training description
    text="आपका स्वागत है।"
)
```

### Plan B — Bhojpuri
```python
audio = generate(
    model, tokenizers,
    description="A female speaker delivers speech at a moderate pace. "
                "The recording is of very high quality.",
    text="रउरा स्वागत बा।"   # Bhojpuri text in Devanagari
)
```

---

## Data Sources

| Resource | URL | License |
|----------|-----|---------|
| SYSPIN corpus (Hindi, Bhojpuri, 9 languages) | https://spiredatasets.ee.iisc.ac.in/syspincorpus | CC-BY-4.0 |
| SPRINGLab/IndicTTS-Hindi (HuggingFace) | huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi | CC-BY-4.0 |
| dataspeech (description generation) | github.com/huggingface/dataspeech | Apache-2.0 |
| IndicVoices-R (22 languages, 1700h) | huggingface.co/datasets/ai4bharat/IndicVoices-R | CC-BY-4.0 |
