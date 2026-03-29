# Training Pipelines — What They Do and Why

This repo supports three distinct ways to finetune Qwen3-TTS. Each trains a different capability into the model and produces a different kind of adapter. Understanding which one to use and why matters a lot for the quality of your output.

---

## Pipeline 1 — Language / Accent Adaptation

**Config**: `configs/qwen3_tts_hindi.yaml`
**Model type**: `qwen3_tts`
**Use when**: You want the model to speak a new language or accent it wasn't pretrained on.

### What it trains

LoRA adapters on the attention and MLP layers of the `talker` transformer. The model learns the phonetics, prosody, and rhythm of your target language by seeing many examples of (text, audio codec) pairs in that language.

### How it conditions language (two modes)

**Mode A — `lang_code: auto` (nothink prefix)**

The codec prefix is:
```
[nothink, think_bos, think_eos, pad, bos]
```
No dedicated language token. The model learns the language purely from the LoRA weight updates. Weaker signal — the model has to figure out "this is Hindi" from the audio patterns alone.

**Mode B — `custom_lang_ids` + `lang_code: hi` (language token)**

The codec prefix is:
```
[think, think_bos, lang_id(2051), think_eos, pad, bos]
```
A dedicated token ID is inserted into the codec embedding table and used as an explicit language conditioning signal before generation starts. The transformer attends to this token throughout generation — much stronger signal than Mode A.

The base Qwen3-TTS model reserves ID 2050 for English (`en`). Indian language tokens use the unused slots starting at 2051:

| Language | Code | Token ID |
|----------|------|----------|
| Hindi    | `hi` | 2051     |
| Tamil    | `ta` | 2052     |
| Telugu   | `te` | 2056     |
| Kannada  | `kn` | 2059     |
| Marathi  | `mr` | 2062     |

To register a new language, add it to `custom_lang_ids` in your config — no code changes needed.

### What you get

- An adapter that makes the model speak your language
- Voice identity at inference still comes from `ref_audio` (ICL mode)
- Adapter size: 23 MB (rank-8)

### Effect on inference

```
Without ref_audio → random voice, target language
With ref_audio    → cloned voice from the clip, target language
```

---

## Pipeline 2 — Speaker Voice Cloning

**Config**: `configs/qwen3_tts_speaker.yaml`
**Model type**: `qwen3_tts_speaker`
**Use when**: You want to permanently bake a specific person's voice into the model so no reference audio is needed at inference.

### What it trains

Same LoRA as Pipeline 1 (rank-16 instead of rank-8 for more capacity), **plus** a speaker conditioning path:

1. Every training sample must have a `ref_audio` field pointing to a clip of the target speaker
2. A frozen **speaker encoder** (built into the mlx_audio Qwen3-TTS model) extracts a speaker embedding from the ref audio mel spectrogram
3. This speaker embedding is inserted as a **single token** into the codec prefix — immediately after the think/lang section and before `[pad, bos]`:

```
auto+speaker:  [nothink, think_bos, think_eos, spk_embed, pad, bos]
lang+speaker:  [think,  think_bos, lang_id,   think_eos, spk_embed, pad, bos]
```

This matches the official Qwen3-TTS `sft_12hz.py` approach exactly — the speaker token is in the transformer's context before any codec token is generated, so every prediction is conditioned on the speaker identity.

### Why positional injection matters

An earlier implementation broadcast-added the speaker embedding across all codec positions (additive conditioning). That's out-of-distribution — the model was pretrained with speaker identity at a specific sequence position, not spread across the codec. Positional injection matches the pretraining format, requiring far less LoRA capacity to converge.

### Baking the voice post-training

After training, `bake_speaker_embedding.py` computes the mean speaker embedding from all training reference clips and writes it into `codec_embedding.weight[3000]` — slot 3000 is the reserved custom-voice token. The model config is updated to `tts_model_type: custom_voice`. From this point, the model generates the target voice without any ref_audio.

### What you get

- An adapter tied to one specific speaker's voice
- No ref_audio needed at inference
- Adapter size: 45 MB (rank-16)

### Effect on inference

```
Without ref_audio → the baked speaker's voice (always)
With ref_audio    → ref_audio is ignored (voice is already baked)
```

---

## Pipeline 3 — Multilingual Training

**Config**: `configs/qwen3_tts_multilingual.yaml`
**Model type**: `qwen3_tts`
**Use when**: You want a single adapter that covers multiple languages, each with their own language token conditioning.

### What it trains

Same as Pipeline 1 but with multiple languages mixed in a single training run. Each sample in the JSONL has its own `lang_code` field, so the model sees the correct language token per sample. This is why per-sample `lang_code` flows all the way from the JSONL through `TTSSample` → `collate_qwen3` → `batch["lang_codes"]` → `_build_codec_prefix`.

### Why mixing languages works

Each language gets a distinct token ID in the codec embedding table. The transformer learns to associate each token with its language's phonetic space. With 5 languages in one run:

- Language diversity prevents the adapter from collapsing to one accent
- `label_smoothing: 0.1` reduces overconfidence (important with varied phonetics)
- Longer warmup (100 steps) stabilises learning across different loss scales per language

### What you get

- A single adapter covering all trained languages
- Stronger per-language quality than training each language separately (cross-lingual regularisation effect)
- At inference, pass `lang_code="hi"` / `"kn"` / `"ta"` etc. to get the right language

### Effect on inference

```
lang_code="hi" + ref_audio → Hindi speech in the cloned voice
lang_code="kn" + ref_audio → Kannada speech in the cloned voice
lang_code="auto"           → falls back to base English behaviour
```

---

## Comparison

| | Pipeline 1 | Pipeline 2 | Multilingual |
|---|---|---|---|
| Goal | New language/accent | Specific speaker voice | Multiple languages |
| LoRA rank | 8 | 16 | 8 |
| Speaker encoder | Not used | Used (frozen, from mlx_audio) | Not used |
| `ref_audio` at training | No | Yes (required) | No |
| `ref_audio` at inference | Optional (for voice cloning) | Not needed | Optional |
| Language tokens | Yes (`custom_lang_ids`) | Yes | Yes (per-sample) |
| label_smoothing | 0.1 | 0.0 | 0.1 |
| Adapter size | 23 MB | 45 MB | 23 MB |

---

## Choosing the Right Pipeline

```
Do you want to add a new language?
  └─ Yes → Pipeline 1 or Multilingual
       └─ Multiple languages at once? → Multilingual
       └─ Single language?            → Pipeline 1

Do you want to clone a specific voice?
  └─ Need ref_audio at inference?  → Pipeline 1 (use ICL mode)
  └─ No ref_audio at inference?    → Pipeline 2 (bake the voice)
```

---

## Data Format

All pipelines use JSONL with one sample per line.

**Pipeline 1 / Multilingual:**
```json
{"audio": "path/to/clip.wav", "text": "transcription", "lang_code": "hi"}
```

**Pipeline 2:**
```json
{"audio": "path/to/clip.wav", "text": "transcription", "lang_code": "hi", "ref_audio": "path/to/ref.wav"}
```

The `ref_audio` should always be a clip of the **target speaker** (the voice you want to bake in). Using the same 5–10 second clip for every sample gives the strongest, most consistent speaker identity signal.
