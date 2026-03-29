#!/usr/bin/env python3
"""
download_common_voice.py — Download Indian-accented English from Mozilla Common Voice.

Source: mozilla-foundation/common_voice_17_0  (CC-0)
No login required.

Usage:
    python scripts/download_common_voice.py --samples 2000

After downloading, run preprocess_dataset.py:
    python scripts/preprocess_dataset.py \\
        --input data/en/train.jsonl \\
        --output data/en/train_codes.jsonl \\
        --model_id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit

lang_code used: "en" (token ID 2050 — already in pretrained model).
Training on Indian-accented English adapts the existing English token toward
Indian pronunciation without needing a new token ID.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

INDIA_ACCENTS = {"India", "indian", "en-IN", "in"}
MIN_DUR = 1.0
MAX_DUR = 14.0
VAL_FRAC = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000,
                        help="Max samples to collect (default: 2000)")
    parser.add_argument("--out_dir", default="./data/en")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset, Audio as HFAudio
    import io

    print("[download] Streaming Common Voice 17 English ...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        streaming=True,
    )
    ds = ds.cast_column("audio", HFAudio(decode=False))

    out = Path(args.out_dir)
    wav_dir = out / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    seen = 0

    for sample in ds:
        seen += 1
        if seen % 2000 == 0:
            print(f"  scanned {seen}, collected {len(collected)}/{args.samples} ...")

        # Filter to Indian accent
        accent = sample.get("accent", "") or ""
        if not any(a.lower() in accent.lower() for a in INDIA_ACCENTS):
            continue

        # Basic quality: upvotes > downvotes
        if sample.get("down_votes", 0) > sample.get("up_votes", 0):
            continue

        text = (sample.get("sentence") or "").strip()
        if not text:
            continue

        audio_data = sample.get("audio", {})
        if not audio_data or not audio_data.get("bytes"):
            continue

        try:
            arr, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            arr = arr.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
        except Exception:
            continue

        dur = len(arr) / sr
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue

        collected.append({"text": text, "audio_data": audio_data, "duration": dur})
        if len(collected) >= args.samples:
            break

    if not collected:
        print(f"[download] No Indian-accented samples found after scanning {seen} entries.")
        print("[download] Common Voice accent tags may differ — try removing the accent filter.")
        return

    print(f"[download] Collected {len(collected)} Indian English samples. Saving ...")

    rng = random.Random(args.seed)
    rng.shuffle(collected)
    n_val     = max(1, int(len(collected) * VAL_FRAC))
    val_set   = collected[:n_val]
    train_set = collected[n_val:]

    def save_split(items, split_name):
        records = []
        for i, item in enumerate(items):
            fname = f"en_{split_name}_{i:05d}.wav"
            fpath = wav_dir / fname
            arr, sr = sf.read(io.BytesIO(item["audio_data"]["bytes"]))
            arr = arr.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            if sr != 24000:
                try:
                    import scipy.signal as ss
                    arr = ss.resample_poly(arr, 24000, sr).astype(np.float32)
                    sr  = 24000
                except Exception:
                    pass
            sf.write(str(fpath), arr, sr)
            records.append({"audio": str(fpath), "text": item["text"], "lang_code": "en"})
        jsonl = out / f"{split_name}.jsonl"
        with open(jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} → {jsonl}")

    save_split(train_set, "train")
    save_split(val_set,   "val")
    print("[download] Done.")


if __name__ == "__main__":
    main()
