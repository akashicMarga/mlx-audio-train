"""
MiniMind-O microphone demo — pure MLX, no PyTorch.

True speech-to-speech pipeline:
  mic (sounddevice)
  → MLX fbank + LFR  (mlx_audio.dsp)
  → MLX SenseVoice encoder  (mlx_audio.stt)
  → audio_proj  (MiniMindOmni, MLX)
  → Thinker + Talker  (MiniMindOmni, MLX)
  → mlx-audio Mimi decoder  (MLX)
  → speaker (sounddevice)

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    python scripts/minimind_o_mic_demo.py
    python scripts/minimind_o_mic_demo.py --sensevoice_model mlx-community/SenseVoiceSmall
"""

import argparse
import glob
import os
import sys
import termios
import threading
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimind_o import MiniMindOmni, OmniConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEIGHTS_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out", "minimind_3o", "weights.npz",
)
HF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots"
)
MIC_SR   = 16_000   # SenseVoice input sample rate
PLAY_SR  = 24_000   # Mimi output sample rate
AUDIO_STOP_TOKEN = 2050
AUDIO_PAD_TOKEN  = 2049


def _flush_stdin() -> None:
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------

def load_tokenizer():
    snaps = sorted(glob.glob(os.path.join(HF_SNAP, "*/tokenizer.json")))
    if not snaps:
        raise FileNotFoundError(
            "Tokenizer not found. Run scripts/minimind_o_test_text.py first."
        )
    tok_dir = os.path.dirname(snaps[-1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if not tok.chat_template:
        jinja = os.path.join(tok_dir, "chat_template.jinja")
        if os.path.exists(jinja):
            tok.chat_template = open(jinja).read()
    return tok


def load_omni_model() -> MiniMindOmni:
    if not os.path.exists(WEIGHTS_NPZ):
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS_NPZ}.\nRun scripts/minimind_o_test_text.py first."
        )
    config = OmniConfig()
    model = MiniMindOmni(config, audio_encoder_path="")
    weights = list(np.load(WEIGHTS_NPZ, allow_pickle=False).items())
    model.load_weights([(k, mx.array(v)) for k, v in weights], strict=False)
    model.eval()
    return model


def load_sensevoice_mlx(repo: str = "mlx-community/SenseVoiceSmall"):
    """Load the MLX-native SenseVoice encoder, patching in CMVN + BPE if missing."""
    from mlx_audio.stt.utils import load_model
    from mlx_audio.utils import get_model_path
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    sv = load_model(repo)
    snap_dir = Path(get_model_path(repo))

    if sv._cmvn_means is None:
        mvn_path = snap_dir / "am.mvn"
        if not mvn_path.exists():
            hf_hub_download(repo, "am.mvn", local_dir=str(snap_dir))
        if mvn_path.exists():
            from mlx_audio.stt.models.sensevoice.sensevoice import _parse_am_mvn
            means, istd = _parse_am_mvn(str(mvn_path))
            sv._cmvn_means = mx.array(means)
            sv._cmvn_istd  = mx.array(istd)
            print("  (CMVN loaded)")

    if sv._tokenizer is None:
        bpe_path = snap_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
        if not bpe_path.exists():
            hf_hub_download(repo, "chn_jpn_yue_eng_ko_spectok.bpe.model", local_dir=str(snap_dir))
        if bpe_path.exists():
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(str(bpe_path))
                sv._tokenizer = sp
                print("  (BPE tokenizer loaded)")
            except ImportError:
                pass

    sv.eval()
    return sv


def load_mimi_mlx():
    from mlx_audio.codec import Mimi
    return Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")


# ---------------------------------------------------------------------------
# MLX audio feature extraction
# ---------------------------------------------------------------------------

def encode_speech_mlx(
    wav: np.ndarray,
    sv_model,
    omni_model: MiniMindOmni,
    sample_rate: int = MIC_SR,
) -> tuple[list, str]:
    """wav → SenseVoice (MLX) → (audio_proj features, transcript)"""
    audio_mx = mx.array(wav.astype(np.float32))

    feats = sv_model._extract_features(audio_mx)   # (T, 560)

    try:
        result = sv_model.generate(audio_mx, language="auto", verbose=False)
        transcript = result.text.strip()
    except Exception:
        transcript = ""

    enc_out = sv_model.encoder(feats[None])        # (1, T, 512)
    mx.eval(enc_out)

    projected = omni_model.audio_proj(enc_out)[0]  # (T, hidden_size)
    mx.eval(projected)

    return [projected], transcript


# ---------------------------------------------------------------------------
# Push-to-talk recorder
# ---------------------------------------------------------------------------

class PushToTalkRecorder:
    def __init__(self, sample_rate: int = MIC_SR):
        import sounddevice as sd
        self.sd = sd
        self.sr = sample_rate
        self._chunks: list = []
        self._stream = None
        self._lock = threading.Lock()

    def _cb(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def start(self):
        self._chunks = []
        self._stream = self.sd.InputStream(
            samplerate=self.sr, channels=1, dtype=np.float32, callback=self._cb
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(self.sr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_AUDIO_MARKER = "<|audio_pad|>"
_SYSTEM_PROMPT = "You are a helpful voice assistant. Reply in English only."


def build_audio_prompt(tokenizer, audio_feature_len: int, transcript: str = "") -> str:
    user_content = _AUDIO_MARKER * audio_feature_len
    if transcript:
        user_content += f"\n\n[You said: {transcript}]"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def generate_speech_response(
    omni_model: MiniMindOmni,
    tokenizer,
    mlx_audio_feats: list,
    transcript: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.75,
    top_p: float = 0.85,
) -> tuple[str, list]:
    audio_feature_len = mlx_audio_feats[0].shape[0] if mlx_audio_feats[0] is not None else 1

    prompt = build_audio_prompt(tokenizer, audio_feature_len, transcript)
    input_ids = mx.array(tokenizer(prompt).data["input_ids"], dtype=mx.int32)[None, :]

    response_tokens: list[int] = []
    audio_frames: list[list[int]] = []

    for y, frame in omni_model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        return_audio_codes=True,
        mlx_audio_feats=mlx_audio_feats,
    ):
        if y is not None:
            response_tokens = y[0].tolist()
            print(
                f"\r  [Text] {tokenizer.decode(response_tokens, skip_special_tokens=True)}",
                end="", flush=True,
            )
        if frame and all(c < AUDIO_STOP_TOKEN for c in frame):
            audio_frames.append(frame)
        if y is None and not frame:
            break

    print()
    text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return text, audio_frames


# ---------------------------------------------------------------------------
# Mimi decode + playback
# ---------------------------------------------------------------------------

def decode_and_play(mimi, audio_frames: list, sample_rate: int = PLAY_SR):
    if not audio_frames:
        print("  (no audio frames generated)")
        return
    codes = np.clip(np.array(audio_frames, dtype=np.int32), 0, 2047)  # (n_frames, 8)
    codes_mx = mx.array(codes.T[None, ...])    # (1, 8, n_frames)
    audio_mx = mimi.decode(codes_mx)           # (1, 1, samples)
    mx.eval(audio_mx)
    wav = np.array(audio_mx[0, 0].tolist(), dtype=np.float32)
    peak = np.abs(wav).max()
    if peak > 0.01:
        wav = wav / peak * 0.9
    duration = len(wav) / sample_rate
    print(f"  [Audio] {len(audio_frames)} frames → {duration:.1f}s  Playing ...", flush=True)
    import sounddevice as sd
    sd.play(wav, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------

def run_demo(args):
    print("=" * 60)
    print("  MiniMind-O  |  Speech → Speech  |  Pure MLX")
    print("=" * 60)

    print("\nLoading tokenizer ...", end=" ", flush=True)
    tokenizer = load_tokenizer()
    print("OK")

    print("Loading MiniMind-O (Thinker + Talker) ...", end=" ", flush=True)
    omni = load_omni_model()
    from mlx.utils import tree_flatten
    n = sum(p.size for _, p in tree_flatten(omni.parameters())
            if "audio_encoder" not in _ and "vision_encoder" not in _)
    print(f"OK  ({n/1e6:.0f}M params)")

    print(f"Loading SenseVoice encoder [{args.sensevoice_model}] ...", end=" ", flush=True)
    sv = load_sensevoice_mlx(args.sensevoice_model)
    print("OK")

    print("Loading Mimi decoder (mlx-audio) ...", end=" ", flush=True)
    mimi = load_mimi_mlx()
    print(f"OK  ({mimi.sample_rate}Hz, {mimi.frame_rate}fps)")

    recorder = PushToTalkRecorder(sample_rate=MIC_SR)

    import sounddevice as sd
    default_in = sd.query_devices(kind="input")
    print(f"\n  Mic: [{default_in['index']}] {default_in['name']}  "
          f"({default_in['max_input_channels']}ch, {int(default_in['default_samplerate'])}Hz)")
    print("  (set --device N to change, or use System Preferences → Sound → Input)")

    print("\n" + "─" * 60)
    print("  Press ENTER to start/stop recording.")
    print("  Type 'q' + ENTER to quit.")
    print("─" * 60)

    while True:
        _flush_stdin()
        try:
            cmd = input("\n[ENTER to speak | q to quit] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q":
            break

        _flush_stdin()
        print("  Recording ... (press ENTER to stop)", end="", flush=True)
        recorder.start()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        wav = recorder.stop()
        dur = len(wav) / MIC_SR

        rms   = float(np.sqrt(np.mean(wav ** 2)))
        db    = 20 * np.log10(rms + 1e-9)
        bar   = "#" * min(30, max(0, int((db + 60) / 2)))
        level = f"{db:+.0f} dB  [{bar:<30}]"
        print(f"\r  Captured {dur:.1f}s  {level}")

        if dur < 0.5:
            print("  (too short — need at least 0.5s)")
            continue

        if rms < 5e-4:
            print("  Mic appears muted — check System Prefs → Sound → Input")
            continue

        TARGET_RMS = 0.1
        if rms > 1e-6:
            wav = wav * (TARGET_RMS / rms)

        print("  SenseVoice (encode + transcribe) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            mlx_feats, transcript = encode_speech_mlx(wav, sv, omni)
        except Exception as e:
            print(f"\n  ERROR during encoding: {e}")
            continue
        n_feat = mlx_feats[0].shape[0] if mlx_feats[0] is not None else 0
        print(f"OK  ({n_feat} frames, {time.time()-t0:.2f}s)")
        if transcript:
            print(f"  [Heard]  {transcript}")

        print("  Generating response ...", flush=True)
        t0 = time.time()
        text, audio_frames = generate_speech_response(
            omni, tokenizer, mlx_feats,
            transcript=transcript,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"  {len(audio_frames)} audio frames in {time.time()-t0:.1f}s")

        decode_and_play(mimi, audio_frames)
        _flush_stdin()

    print("\nGoodbye!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def list_devices():
    import sounddevice as sd
    print("\nAvailable audio INPUT devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = " <- default" if i == sd.query_devices(kind="input")["index"] else ""
            print(f"  [{i:2d}] {d['name']}  ({d['max_input_channels']}ch){mark}")
    print()


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O speech-to-speech demo (pure MLX)")
    parser.add_argument(
        "--sensevoice_model", default="mlx-community/SenseVoiceSmall",
        help="SenseVoice HF repo (default: mlx-community/SenseVoiceSmall)",
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--device", type=int, default=None,
                        help="Sounddevice input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.device is not None:
        import sounddevice as sd
        sd.default.device[0] = args.device

    run_demo(args)


if __name__ == "__main__":
    main()
