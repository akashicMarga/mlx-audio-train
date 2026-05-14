# MiniMind-O (MLX)

MLX port of [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) — a compact speech-to-speech model with optional vision input, running natively on Apple Silicon.

## Config-driven architecture

**Every architectural decision comes from a YAML config file. No code changes needed to swap components.**

```yaml
# configs/minimind_o/mms_phase1a.yaml
model:
  audio_encoder_type: mms-300m    # ← change this one line to swap the encoder
  audio_encoder_path: facebook/mms-300m
  hidden_size: 768

training:
  mode: audio_proj                # ← change this to control what trains
  freeze_backbone: all
  epochs: 5
  data_path: ./data/indic_train.parquet
```

```bash
# Run any configuration — no code changes
python scripts/minimind_o_train.py --config configs/minimind_o/mms_phase1a.yaml

# Override individual fields from CLI
python scripts/minimind_o_train.py --config configs/minimind_o/mms_phase1a.yaml --epochs 10 --lr 2e-4
```

**What is config-controlled:**

| Config field | Effect |
|---|---|
| `model.audio_encoder_type` | Which encoder loads: `sensevoice` / `wav2vec2-base` / `wav2vec2-large` / `mms-300m` / `mms-1b` |
| `model.audio_hidden_size` | Projector `in_dim` — auto-resolved from encoder type if `0` |
| `training.mode` | What trains: `audio_proj` / `vision_proj` / `full` |
| `training.freeze_backbone` | Thinker freeze: `all` / `none` / `last1` / `last2` |
| `training.freeze_talker` | Whether Talker is frozen |
| `training.learning_rate` | Optimizer LR |
| `training.epochs` | Training duration |

**Adding a new encoder** requires only two steps — no model code changes:
1. Add its hidden size to `AUDIO_ENCODER_HIDDEN_SIZES` in `config.py`
2. Add a loader function in `speech_encoder.py` and a branch in `load_audio_encoder()`

---

## Architecture

```
[Mic / Audio file]
    ↓
SenseVoice (MLX, mlx-community/SenseVoiceSmall)   — speech encoder, 16kHz fbank input
    ↓ (T, 512)
MMAudioProjector                                   — 3-layer MLP, projects to Thinker dim
    ↓ injected at <|audio_pad|> positions
┌──────────────────────────────────────────────────────────┐
│  Thinker  —  8-layer causal transformer LM, 768 hidden   │
│  GQA attention (8 heads / 4 kv), SwiGLU MLP, RoPE        │
│  Optional MoE (4 experts, top-1)                         │
└──────────────────┬───────────────────────────────────────┘
                   │ bridge-layer hidden states (layer 3)
                   ↓
┌──────────────────────────────────────────────────────────┐
│  Talker  —  4-layer transformer, same hidden dim         │
│  TalkerEmbedding: 8 parallel codebook embeddings         │
│  TalkerHead: shared base + 8 per-codebook adapter heads  │
└──────────────────────────────────────────────────────────┘
    ↓ 8 Mimi codebook logits per step
Mimi decoder (mlx-audio, kyutai/moshiko-mlx-bf16) — 24kHz audio output
```

**Optional vision path:**
```
[Image]
    ↓
SigLIP2-base-p32-256 (MLX-native)   — ViT, 256×256 → 64 patch tokens (768d)
    ↓ (1, 64, 768)
MMVisionProjector                    — 3-layer MLP
    ↓ injected at <|image_pad|> × 64 positions in prompt
Thinker → Talker  (same as above)
```

## Model Size

| Component | Params | Frozen |
|-----------|--------|--------|
| Thinker (8-layer transformer) | ~85M | No |
| Talker (4-layer + adapters) | ~33M | No |
| MMAudioProjector | ~1.2M | No |
| MMVisionProjector | ~1.2M | No |
| SenseVoice encoder | ~230M | Yes (external) |
| SigLIP2-p32-256 | ~94M | Yes (external) |

**Total trainable: 118M**

## Files

```
models/minimind_o/
    __init__.py          — exports MiniMindOmni, OmniConfig
    config.py            — MiniMindConfig + OmniConfig dataclasses
    thinker.py           — transformer backbone (GQA, SwiGLU, MoE, RoPE)
    talker.py            — acoustic head (TalkerHead, TalkerEmbedding)
    projectors.py        — MMAudioProjector, MMVisionProjector
    speech_encoder.py    — SenseVoice PyTorch wrapper (funasr)
    vision_encoder.py    — MLX-native SigLIP2 ViT + image preprocessor
    model.py             — MiniMindOmni: full model, generation, weight loading
    dataset.py           — OmniDataset: Parquet loader for training
    vad.py               — SileroVAD + RealtimeSession (always-on streaming)
```

## Web Demo

An interactive browser UI with streaming text + audio playback, mic recording, image upload, and 6 built-in sample images for quick vision testing.

```bash
# Install server deps (one-time)
pip install flask flask-cors flask-sock

# Launch (auto-detects weights, SenseVoice, Mimi, SigLIP2)
python scripts/minimind_o_web_demo.py

# Open in browser
open http://localhost:7860
```

**Optional flags:**

```bash
# Disable SenseVoice (text + image only, faster startup)
python scripts/minimind_o_web_demo.py --sensevoice ""

# Custom port
python scripts/minimind_o_web_demo.py --port 8080

# Tune streaming (larger chunk = less CPU, more latency)
python scripts/minimind_o_web_demo.py --audio_chunk_frames 8
```

**Features:**
- **Sidebar sample images** — 6 pre-loaded images (cat, dog, pizza, Eiffel Tower, mountain, coffee). Click one to attach it and auto-fill a suggested question.
- **Mic recording** — click the mic button to start/stop; audio is sent to SenseVoice for encoding.
- **Image upload** — attach any image from disk via the image button.
- **Streaming audio** — Mimi codes are decoded and played in real-time as they arrive (no waiting for the full response).
- **Multi-turn history** — conversation context is maintained across turns (configurable via `--max_history_turns`).
- **Settings panel** — temperature, top-p, max tokens, adjustable per session.

**Status badges** in the header show which components loaded (green = active):

| Badge | What it means |
|-------|---------------|
| SenseVoice | Mic input enabled — speech is encoded to Thinker features |
| Mimi | Audio output enabled — responses include synthesised speech |
| Vision | Image input enabled — SigLIP2-p32-256 loaded |

If Vision is grey, download the vision encoder first:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/siglip2-base-patch32-256', local_dir='./model/siglip2_p32')
"
```

---

## Scripts

```bash
# 1. Download weights from HF + text smoke test (runs this first)
python scripts/minimind_o_test_text.py
python scripts/minimind_o_test_text.py --prompt "What is the capital of France?"

# 2. Verify weight conversion is numerically correct
python scripts/minimind_o_verify_alignment.py
python scripts/minimind_o_verify_alignment.py --verbose   # print every key

# 3. Interactive mic demo (push-to-talk, speech → speech)
python scripts/minimind_o_mic_demo.py
python scripts/minimind_o_mic_demo.py --list-devices      # pick mic
python scripts/minimind_o_mic_demo.py --device 2

# 4. Batch inference (text / audio / voice-clone / multi-turn modes)
python scripts/minimind_o_eval.py --model_dir out/ --mode 0   # text → speech
python scripts/minimind_o_eval.py --model_dir out/ --mode 2   # audio → speech
python scripts/minimind_o_eval.py --model_dir out/ --mode 3   # voice clone
python scripts/minimind_o_eval.py --model_dir out/ --mode 0,1,2,3  # all modes

# 5. Convert weights (local .pth or HuggingFace repo → .npz)
python scripts/minimind_o_convert_weights.py --hf_repo jingyaogong/minimind-3o --output out/minimind_3o_mlx.pth
python scripts/minimind_o_convert_weights.py --input out/sft_omni_768.pth --output out/sft_omni_mlx.npz --verify

# 6. Fine-tune
python scripts/minimind_o_train.py \
    --data_path dataset/train_t2a.parquet \
    --tokenizer_path ./model \
    --mode all --epochs 15
```

## Weights

Weights are not included in the repo. `minimind_o_test_text.py` downloads and converts them automatically on first run:

```
HuggingFace: jingyaogong/minimind-3o
Cached to:   out/minimind_3o/weights.npz  (gitignored)
```

To convert a local `.pth` checkpoint manually:
```bash
python scripts/minimind_o_convert_weights.py --input your_checkpoint.pth --output out/mlx_weights.npz
```

## Vision (Image Q&A)

The vision path requires `google/siglip2-base-patch32-256` (patch=32, 64 tokens):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/siglip2-base-patch32-256', local_dir='./model/siglip2_p32')
"
```

Then use `load_siglip2_mlx` in your code:

```python
from models.minimind_o.vision_encoder import load_siglip2_mlx
from PIL import Image

vision_enc, vision_proc = load_siglip2_mlx("./model/siglip2_p32")

img = Image.open("photo.jpg").convert("RGB")
vision_feats = vision_enc(vision_proc(img)).last_hidden_state  # (1, 64, 768)

# Pass to model via mlx_vision_feats
out = model.generate(..., mlx_vision_feats=vision_feats)
```

**Note:** Use the **p32** variant, not p16. p16 gives 256 patches but the model was trained with 64 tokens (`image_token_len=64`). Using the wrong variant degrades vision quality significantly.

## Capabilities & Limitations

| Task | Quality |
|------|---------|
| Text → speech | Good — coherent English responses at 120–180 tok/s |
| Speech → speech | Good — SenseVoice transcribes, Thinker+Talker responds |
| Image → text/speech | Basic — identifies main objects/scenes, struggles with details |
| Math / reasoning | Weak — 118M LM, expect errors on arithmetic |
| Code | Not trained for it |

The README for the upstream model notes: *"usually captures the main object and the rough scene, but fine-grained spatial relations, counts and attributes are still often wrong."*

## Generation Parameters

| Param | Default | Notes |
|-------|---------|-------|
| `temperature` | 0.75 | Lower = more focused, higher = more creative |
| `top_p` | 0.90 | Nucleus sampling |
| `max_new_tokens` | 512 | Per turn |
| `use_cache` | True | KV cache for incremental decoding |
| `return_audio_codes` | False | Set True to get Mimi codes alongside text |

## Training Data Format

The training script (`minimind_o_train.py`) reads Parquet files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `conversations` | JSON string | Chat turns `[{role, content}]` |
| `question_audios` | list of bytes | Raw audio per user turn |
| `answer_audios` | list of ints | Mimi codes (interleaved 8 codebooks) |
| `image_bytes` | bytes | Optional image for vision turns |
| `ref_audios` | list of ints | Optional reference audio codes (voice clone) |
| `spk_emb` | list of floats | Optional CAM++ speaker embedding (192d) |

## VAD (Always-on Streaming)

`vad.py` provides `SileroVAD` and `RealtimeSession` for a fully hands-free loop where speech start/end is detected automatically. This requires the Silero ONNX model:

```bash
# Download silero_vad.onnx from https://github.com/snakers4/silero-vad
```

The mic demo currently uses push-to-talk (ENTER to start/stop). VAD-based mode is available for integration into a custom streaming loop.

## Key Config Fields

```python
OmniConfig(
    hidden_size=768,             # Thinker hidden dim
    num_hidden_layers=8,         # Thinker depth
    num_talker_hidden_layers=4,  # Talker depth
    audio_vocab_size=2112,       # 2048 Mimi codes + 64 specials
    audio_pad_token=2049,
    audio_stop_token=2050,
    audio_spk_token=2051,        # speaker embedding injection position
    image_token_len=64,          # patch tokens per image
    bridge_layer=3,              # which Thinker layer feeds the Talker
    spk_emb_size=192,            # CAM++ embedding dim
)
```

---

## Roadmap — Multilingual → Full-Duplex S2S

The goal is to evolve MiniMind-O from a Chinese-first single-turn speech model into a multilingual, full-duplex Indian language speech assistant. Four phases, each buildable independently.

---

### Phase 1a — Swap ASR Encoder (SenseVoice → MMS / wav2vec2)

**Goal:** Validate multilingual Indian language understanding with minimal training (projector-only, ~1.2M params). No backbone changes.

**Why:** SenseVoice is strong on Chinese/Japanese but weak on Indian languages. Meta's MMS supports 1000+ languages including all major Indian ones; wav2vec2-large has strong Indic representations.

| File | Change |
|------|--------|
| `models/minimind_o/speech_encoder.py` | Add `load_wav2vec2(path)` and `load_mms(path)` — freeze all encoder weights. Add `Wav2Vec2AudioProcessor` wrapper (16kHz → features, output dim 1024). Keep `load_sensevoice()` — encoder type switchable. |
| `models/minimind_o/config.py` | Add `audio_encoder_type: str = "sensevoice"` (options: `sensevoice`, `mms`, `wav2vec2`). Drive `audio_hidden_size` from encoder type (512 for SenseVoice, 1024 for MMS-large). |
| `models/minimind_o/projectors.py` | Read `in_dim` from config instead of hardcoded 512. |
| `models/minimind_o/model.py` | Switch encoder load based on `config.audio_encoder_type`. Everything downstream unchanged. |
| `configs/minimind_o_mms.yaml` | `audio_encoder_type: mms`, `audio_hidden_size: 1024`, `freeze_backbone: all`, `mode: audio_proj` |
| `scripts/minimind_o_train.py` | Add `--audio-encoder` CLI arg to override config. |

**Training:** Projector only. ~100–500 steps, minutes on M2.

---

### Phase 1b — Indian Language Data Alignment

**Goal:** Fine-tune Thinker top layers on Indian language audio-text pairs so the model can reason and respond in Indic languages.

| File | Change |
|------|--------|
| `scripts/minimind_o_train.py` | Add `freeze_backbone: last2` mode — unfreeze last 2 Thinker layers + norm. |
| `models/minimind_o/dataset.py` | Support AI4Bharat / IndicTTS dataset format. Handle multiple language sources in one JSONL. |
| `scripts/prepare_indic_dataset.py` | Download + convert AI4Bharat IndicTTS → MiniMind-O Parquet format. Output: `data/indic_train.parquet`. |

**Training:** Projector + last 2 Thinker layers. ~1 day on M2 Max with 10K samples.

**Data:** [AI4Bharat IndicTTS](https://github.com/AI4Bharat/IndicTTS) — 10K+ samples across Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali.

---

### Phase 2a — Mimi as Input Encoder (Single Codec In + Out)

**Goal:** Replace SenseVoice with the causal Mimi encoder — same codec for input and output. Enables true streaming input (process audio as it arrives, no buffering), lower latency, and architectural simplicity. The Thinker already understands Mimi code space from the Talker side — convergence should be fast.

| File | Change |
|------|--------|
| `models/minimind_o/model.py` | Add Mimi encoder path: raw audio → `mimi.encode()` → 8 codebook IDs → embedding lookup → Thinker. Remove `audio_encoder` + `audio_proj` when `use_mimi_input=True`. |
| `models/minimind_o/config.py` | Add `use_mimi_input: bool = False`. |
| `scripts/preprocess_mimi_input.py` | Pre-encode user-side audio to Mimi codes (like `preprocess_dataset.py` does for Qwen3-TTS). Saves `.mimi_input.npy` alongside audio. |
| `models/minimind_o/dataset.py` | Load pre-encoded Mimi input codes when `use_mimi_input=True`. Pass as `user_audio_codes`. |
| `models/minimind_o/vad.py` | Add chunk-streaming path — process Mimi frames as they arrive (80ms chunks). |

**Training:** Input embedding layer + Thinker. ~1–2 days.

---

### Phase 2b — Full Duplex (Moshi-style, No VAD)

**Goal:** Interleaved dual-stream training — model speaks while listening, no explicit turn-taking, no VAD gating. Closest to how humans actually converse.

| File | Change |
|------|--------|
| `models/minimind_o/model.py` | New `forward_duplex()`: two simultaneous Mimi streams (user + model). Interleaved sequence: `[u_t0, m_t0, u_t1, m_t1, ...]` at 12.5Hz. Model predicts next `m_t` conditioned on all prior tokens. |
| `models/minimind_o/dataset.py` | New `DuplexDataset` — loads two-channel conversation Parquet. Each row: `{user_mimi_codes, model_mimi_codes, text}` frame-aligned. |
| `scripts/minimind_o_duplex_train.py` | Training loop with `forward_duplex()`. Loss on model-stream only — no backprop through user stream. |
| `models/minimind_o/vad.py` | Simplify `RealtimeSession` — remove VAD gating, process every 80ms unconditionally. |
| `scripts/prepare_duplex_dataset.py` | Convert single-speaker pairs to interleaved duplex format. Synthetic duplex: TTS question + TTS answer, interleave with overlap simulation. |

**Training:** Full model on duplex data. Days to weeks depending on dataset size.

**Data (hardest part):** No large Indian language duplex corpus exists publicly. Realistic path: synthetic generation — TTS the question in Speaker A's voice, TTS the answer in Speaker B's voice, interleave with 200–500ms overlap at turn boundaries. Target 50K+ pairs for Phase 2b.

---

## Data Requirements per Phase

| Phase | What you need | Source | Min size |
|-------|--------------|--------|----------|
| 1a | Any multilingual audio-text pairs | AI4Bharat, IndicSUPERB, CommonVoice | ~1K samples |
| 1b | Indian language TTS pairs (text + audio) | AI4Bharat IndicTTS | ~10K samples |
| 2a | Same as 1b, Mimi pre-encoded | Same | Same |
| 2b | **Duplex conversations** (two simultaneous streams) | Synthetic or telephone corpus | 50K+ pairs |

---

## Scratch Training vs Fine-tuning

The upstream minimind-o was trained from scratch on a 3090 (24GB VRAM). On Apple Silicon the equivalent is M2 Max (32–96GB unified memory). Key differences:

| | From scratch | Fine-tune from upstream checkpoint |
|---|---|---|
| Data needed | 100K–1M samples | 1K–10K samples |
| Time (M2 Max) | Weeks | Hours–days |
| Control | Full — your architecture choices | Constrained to upstream design |
| Best for | Phase 2b (duplex requires it) | Phase 1a, 1b, 2a |

For Phases 1a–2a, **fine-tuning from the upstream checkpoint is strongly recommended**. Scratch training only makes sense for Phase 2b where the architecture changes fundamentally (duplex forward pass, no SenseVoice dependency).

---

## Branch Strategy

```
main
 └─ phase-1a-mms-encoder       # swap SenseVoice → MMS, train projector
     └─ phase-1b-indic-align   # unfreeze last 2 Thinker layers, Indic data
         └─ phase-2a-mimi-in   # Mimi input codec, streaming
             └─ phase-2b-duplex # full duplex, new forward pass
```
```
