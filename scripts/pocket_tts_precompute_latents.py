#!/usr/bin/env python3
"""
Encode a speech dataset to Pocket TTS Mimi latents (run once before distillation).

Uses the official PyTorch codec from the `pocket-tts` package (pip install pocket-tts),
so latents match what the teacher was trained on exactly. Runs on CPU or MPS; the
encoder is small (~12 s of audio per second on an M-series CPU is typical).

Input JSONL — this repo's usual format, word alignments optional:
    {"audio": "clips/0001.wav", "text": "...", "ref_audio": "clips/0002.wav",
     "words": [{"word": "Hello", "start": 0.12, "end": 0.48}, ...]}
The official manifests (pocket-tts training/scripts/prepare_data.py, e.g. aligned
HiFiTTS-2) work as is: "path"/"transcript" and an optional "start"/"duration"
window inside a longer recording (word times are relative to "start").

Output: <out_dir>/manifest.jsonl + <out_dir>/latents/*.npy (float16, [T, 32] at 12.5 Hz;
~1.6 KB per second of audio, i.e. ~6 GB per 1000 h).

Usage:
    python scripts/pocket_tts_precompute_latents.py \\
        --input data/train.jsonl --out-dir data/pocket_latents/train \\
        --config pocket_tts/config/english_2026-09_24l.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 24000


def load_wav(path: Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    sr = sf.info(str(path)).samplerate
    frames = -1 if duration is None else int(round(duration * sr))
    wav, sr = sf.read(str(path), start=int(round(start * sr)), frames=frames,
                      dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono, no loudness normalisation (matches the official loader)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def build_mimi(config_path: str, device: str):
    import safetensors.torch
    import torch
    from pocket_tts.models.mimi import build_mimi as _build
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    config = load_config(config_path)
    mimi = _build(config.mimi)
    weights = getattr(config.mimi, "weights_path", None) or config.weights_path
    state = safetensors.torch.load_file(str(download_if_necessary(str(weights))))
    mimi.load_state_dict({k.removeprefix("mimi."): v for k, v in state.items()
                          if k.startswith("mimi.")}, strict=True)
    return mimi.eval().to(device), torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", required=True,
                   help="pocket-tts config of the teacher (its Mimi encoder is used)")
    p.add_argument("--audio-root", default=None, help="base dir for relative audio paths "
                   "(default: the input JSONL's directory)")
    p.add_argument("--max-sec", type=float, default=60.0, help="skip clips longer than this")
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    args = p.parse_args()

    import torch

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    mimi, torch = build_mimi(args.config, device)
    in_path = Path(args.input)
    root = Path(args.audio_root) if args.audio_root else in_path.parent
    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    encoded: dict[str, str] = {}

    @torch.no_grad()
    def encode(rel: str, start: float = 0.0, duration: float | None = None) -> str | None:
        key = f"{rel}@{start}+{duration}"
        if key in encoded:
            return encoded[key]
        wav = load_wav(root / rel, start, duration)
        if len(wav) / SAMPLE_RATE > args.max_sec or len(wav) < SAMPLE_RATE // 2:
            return None
        lat = mimi.encode_to_latent(torch.from_numpy(wav)[None, None].to(device))[0]
        name = f"latents/{len(encoded):07d}.npy"
        np.save(out_dir / name, lat.float().cpu().numpy().astype(np.float16))
        encoded[key] = name
        return name

    kept = skipped = 0
    with open(in_path, encoding="utf-8") as fin, open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fout:
        for n, line in enumerate(fin):
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio") or row["path"]
            lat = encode(audio, float(row.get("start", 0.0)),
                         float(row["duration"]) if "start" in row and "duration" in row else None)
            if lat is None:
                skipped += 1
                continue
            out = {"latents": lat, "text": row.get("text") or row["transcript"]}
            if row.get("words"):
                out["words"] = row["words"]
            if row.get("ref_audio") and row["ref_audio"] != audio:
                out["voice_latents"] = encode(row["ref_audio"])
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 200 == 0:
                print(f"  {kept} encoded ({skipped} skipped)", flush=True)
    print(f"[precompute] {kept} rows -> {out_dir / 'manifest.jsonl'} ({skipped} skipped)")


if __name__ == "__main__":
    main()
