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

from models.pocket_tts.codec import SAMPLE_RATE, Mimi, best_device, load_wav  # noqa: E402


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

    mimi = Mimi(args.config, args.device or best_device())
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
        name = f"latents/{len(encoded):07d}.npy"
        np.save(out_dir / name, mimi.encode(wav).astype(np.float16))
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
