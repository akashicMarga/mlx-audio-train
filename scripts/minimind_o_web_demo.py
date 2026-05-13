"""
MiniMind-O Web Demo — MLX backend (Flask + SSE).

Faithful port of jingyaogong/minimind-o webui/web_demo.py adapted for:
  - MLX model (no PyTorch inference)
  - MLX-native SenseVoice encoder (mlx-audio)
  - MLX-native Mimi decoder (mlx-audio)
  - MLX-native SigLIP2 vision encoder (models/minimind_o/vision_encoder.py)

Upstream: https://github.com/jingyaogong/minimind-o

Usage:
    pip install flask flask-cors flask-sock
    python scripts/minimind_o_web_demo.py
    # open http://localhost:7860
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import base64
import glob
import io
import json
import logging
import sys
import threading
import time

import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from models.minimind_o import MiniMindOmni, OmniConfig
from models.minimind_o.vision_encoder import load_siglip2_mlx

logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "minimind_o_webui"))
CORS(app)
sock = Sock(app)

M = {}          # model, tokenizer, mimi, sensevoice, vision_enc/proc, config
GEN_LOCK = threading.Lock()

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimind_o_webui")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    import mlx.core as mx

    # ── Weights ──────────────────────────────────────────────────────────────
    weights_path = args.weights
    if not weights_path:
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out", "minimind_3o", "weights.npz"
        )
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        print("Run: python scripts/minimind_o_test_text.py")
        sys.exit(1)

    # ── MiniMind-O ───────────────────────────────────────────────────────────
    print("Loading MiniMind-O ...", end=" ", flush=True)
    config = OmniConfig()
    model  = MiniMindOmni(config, audio_encoder_path="")
    weights = [(k, mx.array(v)) for k, v in np.load(weights_path, allow_pickle=False).items()]
    model.load_weights(weights, strict=False)
    model.eval()
    M["model"]  = model
    M["config"] = config
    print("OK")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    snap_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots/*/tokenizer.json"
    )
    snaps = sorted(glob.glob(snap_pattern))
    if not snaps:
        print("Tokenizer not found. Run minimind_o_test_text.py first.")
        sys.exit(1)
    tok_dir = os.path.dirname(snaps[-1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    jinja = os.path.join(tok_dir, "chat_template.jinja")
    if not tok.chat_template and os.path.exists(jinja):
        tok.chat_template = open(jinja).read()
    M["tokenizer"] = tok
    print(f"Tokenizer OK  (vocab={len(tok)})")

    # ── MLX Mimi decoder ─────────────────────────────────────────────────────
    try:
        from mlx_audio.codec import Mimi
        M["mimi"] = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")
        print(f"Mimi OK  ({M['mimi'].sample_rate}Hz)")
    except Exception as e:
        M["mimi"] = None
        print(f"Mimi MISSING: {e} — audio output disabled")

    # ── MLX SenseVoice (speech encoder) ─────────────────────────────────────
    if args.sensevoice:
        try:
            from mlx_audio.stt.utils import load_model as load_sv
            from mlx_audio.utils import get_model_path
            from huggingface_hub import hf_hub_download
            from pathlib import Path
            sv = load_sv(args.sensevoice)
            snap_dir = Path(get_model_path(args.sensevoice))
            if sv._cmvn_means is None:
                mvn = snap_dir / "am.mvn"
                if not mvn.exists():
                    hf_hub_download(args.sensevoice, "am.mvn", local_dir=str(snap_dir))
                if mvn.exists():
                    from mlx_audio.stt.models.sensevoice.sensevoice import _parse_am_mvn
                    means, istd = _parse_am_mvn(str(mvn))
                    sv._cmvn_means = mx.array(means)
                    sv._cmvn_istd  = mx.array(istd)
            sv.eval()
            M["sensevoice"] = sv
            print(f"SenseVoice OK  ({args.sensevoice})")
        except Exception as e:
            M["sensevoice"] = None
            print(f"SenseVoice MISSING: {e} — mic input disabled")
    else:
        M["sensevoice"] = None

    # ── MLX SigLIP2 vision encoder ───────────────────────────────────────────
    if args.vision_dir and os.path.exists(args.vision_dir):
        try:
            vision_enc, vision_proc = load_siglip2_mlx(args.vision_dir)
            M["vision_enc"]  = vision_enc
            M["vision_proc"] = vision_proc
            print("SigLIP2 OK")
        except Exception as e:
            M["vision_enc"] = M["vision_proc"] = None
            print(f"SigLIP2 MISSING: {e} — image input disabled")
    else:
        M["vision_enc"] = M["vision_proc"] = None

    M["args"] = args
    print("\nAll models loaded. Starting server ...\n")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def encode_audio_mlx(wav_np: np.ndarray) -> mx.array:
    """wav (16kHz float32 numpy) → (T, hidden) MLX audio features.

    Shape contract: returns (T, hidden_size).
    - T   = number of encoder frames (depends on audio length, typically 20–200)
    - hidden_size = 768 (Thinker hidden dim after audio_proj)
    Caller uses shape[0] (T) to build the <|audio_pad|> prompt tokens.
    """
    sv = M["sensevoice"]
    audio_mx = mx.array(wav_np.astype(np.float32))
    feats   = sv._extract_features(audio_mx)        # (T_fbank, 560)
    enc_out = sv.encoder(feats[None])               # (1, T, 512)
    mx.eval(enc_out)
    projected = M["model"].audio_proj(enc_out)[0]   # drop batch dim → (T, 768)
    mx.eval(projected)
    assert projected.ndim == 2, f"encode_audio_mlx: expected (T, hidden), got {projected.shape}"
    return projected


def encode_image_mlx(image_b64: str) -> mx.array:
    """Base64 image → (1, 64, 768) MLX vision features."""
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    out = M["vision_enc"](M["vision_proc"](img))
    mx.eval(out.last_hidden_state)
    return out.last_hidden_state  # (1, 64, 768)


def mimi_decode_frames(frames: list) -> np.ndarray | None:
    """List of 8-code frames → float32 PCM at 24kHz."""
    if not frames or M["mimi"] is None:
        return None
    codes = np.clip(np.array(frames, dtype=np.int32), 0, 2047)  # (n, 8)
    codes_mx = mx.array(codes.T[None, ...])                      # (1, 8, n)
    audio_mx = M["mimi"].decode(codes_mx)                        # (1, 1, samples)
    mx.eval(audio_mx)
    return np.array(audio_mx[0, 0].tolist(), dtype=np.float32)


def pcm_bytes(wav: np.ndarray, overlap_samples: int = 0) -> bytes:
    """float32 wav → 16-bit signed PCM bytes (with optional overlap skip)."""
    if overlap_samples > 0:
        wav = wav[overlap_samples:]
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16).tobytes()


def sse(d: dict) -> str:
    return f"data: {json.dumps(d)}\n\n"


IMAGE_TOKEN = ""  # filled after model load


def build_prompt(text: str, has_audio: bool, has_image: bool) -> str:
    parts = []
    if has_image:
        parts.append(IMAGE_TOKEN)
    if text:
        parts.append(text)
    if has_audio and not text:
        parts.append("")
    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEBUI_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEBUI_DIR, path)


@app.route("/status")
def status():
    return json.dumps({
        "model": "MiniMind-O (118M)",
        "sensevoice": M.get("sensevoice") is not None,
        "mimi": M.get("mimi") is not None,
        "vision": M.get("vision_enc") is not None,
    })


@app.route("/chat", methods=["POST"])
def chat():
    d = request.json or {}
    text       = d.get("text", "").strip()
    audio_b64  = d.get("audio")    # base64 WAV/webm from browser mic
    image_b64  = d.get("image")    # base64 image
    history    = d.get("history", [])
    max_tokens  = int(d.get("max_tokens", 512))
    temperature = float(d.get("temperature", 0.75))
    top_p       = float(d.get("top_p", 0.90))
    rep_penalty = float(d.get("rep_penalty", 1.3))   # repetition penalty — default 1.3 prevents loops
    args       = M["args"]

    def generate():
        tok    = M["tokenizer"]
        model  = M["model"]
        config = M["config"]

        # ── Pre-process inputs ───────────────────────────────────────────────
        mlx_audio_feats  = None
        mlx_vision_feats = None
        prompt_text = text
        transcript = ""   # speech-to-text transcript for multi-turn history

        if audio_b64 and M.get("sensevoice"):
            try:
                # Browser sends WebM/Opus — pydub handles any format via ffmpeg.
                # soundfile alone can't decode WebM, which was causing silent failures
                # and the model falling back to generic "How can I help you?" greetings.
                from pydub import AudioSegment
                raw = base64.b64decode(audio_b64)
                seg = (AudioSegment.from_file(io.BytesIO(raw))
                       .set_channels(1)
                       .set_frame_rate(16000)
                       .set_sample_width(2))
                wav = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Normalize to consistent RMS level
                rms = np.sqrt(np.mean(wav ** 2))
                if rms > 1e-6:
                    wav = wav * (0.1 / rms)

                # Get SenseVoice transcript — used for multi-turn history so the
                # model has a text record of what was said in previous turns
                sv = M.get("sensevoice")
                if sv is not None:
                    try:
                        audio_mx = mx.array(wav)
                        result = sv.generate(audio_mx, language="auto", verbose=False)
                        transcript = result.text.strip()
                    except Exception:
                        transcript = ""

                mlx_audio_feats = [encode_audio_mlx(wav)]
                n_feat = mlx_audio_feats[0].shape[0]  # T audio frames — NOT shape[1] (=768=hidden)
                print(f"[audio] wav={len(wav)/16000:.1f}s  frames={n_feat}", flush=True)

                # Prompt sent to the model this turn:
                # <|audio_pad|> * T  → actual speech features injected here
                # + typed text if any
                # NOTE: we deliberately do NOT append the transcript here.
                # SenseVoice English ASR is noisy (e.g. "Eiffel" → "i felt"), so
                # putting the garbled transcript into the model prompt causes
                # repetition loops. The transcript is only shown to the user in the UI.
                audio_prompt = config.audio_special_token * n_feat
                prompt_text = audio_prompt + (f"\n\n{text}" if text else "")

                yield sse({"type": "transcript", "content": transcript})
            except Exception as e:
                yield sse({"type": "error", "content": f"Audio encode failed: {e}"})

        if image_b64 and M.get("vision_enc"):
            try:
                mlx_vision_feats = encode_image_mlx(image_b64)
                prompt_text = IMAGE_TOKEN + ("\n" + prompt_text if prompt_text else "")
            except Exception as e:
                yield sse({"type": "error", "content": f"Image encode failed: {e}"})

        # ── Build input IDs ──────────────────────────────────────────────────
        # History entries from the frontend use typed text only (never raw transcripts).
        # Garbled ASR text in history causes the model to loop and echo it back.
        n_hist = args.max_history_turns
        hist = history[-n_hist:] if n_hist > 0 else []
        messages = hist + [{"role": "user", "content": prompt_text}]
        try:
            prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_str = f"<s>user\n{prompt_text}</s><s>assistant\n"

        input_ids = mx.array(tok(prompt_str).input_ids, dtype=mx.int32)[None, :]

        # ── Generate ─────────────────────────────────────────────────────────
        frames: list = []
        full_text = ""
        n_tokens = 0
        CHUNK = args.audio_chunk_frames
        OVERLAP_FRAMES = args.audio_overlap
        SAMPLE_RATE = 24000
        SAMPLES_PER_FRAME = int(SAMPLE_RATE / 12.5)  # Mimi: 12.5fps → 1920 samples/frame
        t0 = time.time()
        text_ttft = audio_ttft = None

        with GEN_LOCK:
            for y, af in model.generate(
                input_ids,
                eos_token_id=tok.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                rp=rep_penalty,
                stream=True,
                return_audio_codes=True,
                mlx_audio_feats=mlx_audio_feats,
                mlx_vision_feats=mlx_vision_feats,
            ):
                # Text token
                if y is not None:
                    n_tokens = y.shape[1]  # (batch=1, seq_len) — seq_len = tokens generated
                    if text_ttft is None:
                        text_ttft = (time.time() - t0) * 1000
                        yield sse({"type": "ttft", "text_ttft": round(text_ttft, 1)})
                    decoded = tok.decode(y[0].tolist(), skip_special_tokens=True)
                    if decoded and len(decoded) > len(full_text):
                        delta = decoded[len(full_text):]
                        full_text = decoded
                        yield sse({"type": "text", "content": delta})

                # Audio frame
                if af and all(c < config.audio_stop_token for c in af):
                    if audio_ttft is None:
                        audio_ttft = (time.time() - t0) * 1000
                        yield sse({"type": "ttft", "audio_ttft": round(audio_ttft, 1)})
                    frames.append(af)

                    # Emit a chunk every CHUNK frames
                    n = len(frames)
                    if n >= CHUNK and n % CHUNK == 0:
                        ov_frames = min(OVERLAP_FRAMES, n - CHUNK)
                        wav = mimi_decode_frames(frames[-(CHUNK + ov_frames):])
                        if wav is not None:
                            ov_samples = ov_frames * SAMPLES_PER_FRAME
                            pcm = pcm_bytes(wav, ov_samples)
                            b64 = base64.b64encode(pcm).decode()
                            for i in range(0, len(b64), 3000):
                                chunk = b64[i:i + 3000]
                                done  = (i + 3000) >= len(b64)
                                yield sse({"type": "pcm", "c": chunk, "d": done})

        # Flush remaining frames
        rem = len(frames) % CHUNK
        if rem:
            ov_frames = min(OVERLAP_FRAMES, len(frames) - rem)
            wav = mimi_decode_frames(frames[-(rem + ov_frames):])
            if wav is not None:
                ov_samples = ov_frames * SAMPLES_PER_FRAME
                pcm = pcm_bytes(wav, ov_samples)
                b64 = base64.b64encode(pcm).decode()
                for i in range(0, len(b64), 3000):
                    chunk = b64[i:i + 3000]
                    done  = (i + 3000) >= len(b64)
                    yield sse({"type": "pcm", "c": chunk, "d": done})

        elapsed = time.time() - t0
        tok_per_sec = round(n_tokens / elapsed, 1) if elapsed > 0 else 0
        audio_dur = round(len(frames) * 0.08, 1)
        yield sse({
            "type": "done",
            "full_text": full_text,
            "transcript": transcript,   # for frontend to store in history
            "audio_frames": len(frames),
            "audio_dur_s": audio_dur,
            "elapsed_s": round(elapsed, 2),
            "n_tokens": n_tokens,
            "tok_per_sec": tok_per_sec,
            "text_ttft_ms": round(text_ttft, 1) if text_ttft else None,
            "audio_ttft_ms": round(audio_ttft, 1) if audio_ttft else None,
        })

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p = argparse.ArgumentParser(description="MiniMind-O Web Demo (MLX)")
    p.add_argument("--weights", default="", help="Path to weights.npz (auto-detected if empty)")
    p.add_argument("--sensevoice", default="mlx-community/SenseVoiceSmall",
                   help="SenseVoice HF repo for mic input (pass '' to disable)")
    p.add_argument("--vision_dir", default=os.path.join(repo_root, "model", "siglip2_p32"),
                   help="Path to SigLIP2-p32-256 for image input")
    p.add_argument("--port", default=7860, type=int)
    p.add_argument("--audio_chunk_frames", default=4, type=int,
                   help="Mimi frames per audio chunk (4 ≈ 320ms)")
    p.add_argument("--audio_overlap", default=2, type=int,
                   help="Overlap frames between chunks (smooths boundaries)")
    p.add_argument("--max_history_turns", default=3, type=int,
                   help="Number of conversation turns to keep in context")
    args = p.parse_args()

    load_models(args)

    # Fill IMAGE_TOKEN after config is loaded
    global IMAGE_TOKEN
    IMAGE_TOKEN = M["config"].image_special_token * M["config"].image_token_len

    os.makedirs(WEBUI_DIR, exist_ok=True)

    print(f"Open: http://localhost:{args.port}")
    # threaded=False: all requests run in the main thread, which is where MLX
    # initialised its GPU stream. Spawning worker threads breaks MLX's thread-local
    # stream registry. Single-threaded is fine for a single-user demo with GEN_LOCK.
    app.run(host="0.0.0.0", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
