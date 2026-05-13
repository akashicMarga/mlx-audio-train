# MiniMind-O (MLX)

MLX port of [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) — a compact speech-to-speech model with optional vision input, running natively on Apple Silicon.

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
