#!/usr/bin/env python3
"""
download_indicvoices.py — Download and prepare IndicVoices-R data for training.

Source: ai4bharat/indicvoices_r (CC-BY-4.0)
Requires: huggingface-cli login  (dataset is gated — accept license at HF first)

Usage:
    # Download 2000 samples of Kannada
    python scripts/download_indicvoices.py --lang kn --samples 2000

    # Download all 5 languages at once
    python scripts/download_indicvoices.py --lang kn mr ta te --samples 2000

    # Smaller test run
    python scripts/download_indicvoices.py --lang kn --samples 200

After downloading, run preprocess_dataset.py per language to generate codec .npy files:
    python scripts/preprocess_dataset.py \\
        --input data/kn/train.jsonl \\
        --output data/kn/train_codes.jsonl \\
        --model_id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit

NOTE — Indian English:
    IndicVoices-R does not include English. For Indian-accented English use
    Mozilla Common Voice filtered to Indian speakers:
        python scripts/download_common_voice.py --lang en --accent india --samples 2000
    (see that script for details)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Language config ────────────────────────────────────────────────────────────

LANG_MAP = {
    # code → IndicVoices-R config name
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

# Minimum SNR for quality filtering (dB). Lower = accept more noise.
MIN_SNR = 20.0
# Duration range matching training config (seconds)
MIN_DUR = 1.0
MAX_DUR = 14.0
# Val split fraction
VAL_FRAC = 0.05


def download_language(lang_code: str, n_samples: int, out_dir: Path, seed: int = 42):
    from datasets import load_dataset

    config_name = LANG_MAP.get(lang_code)
    if not config_name:
        print(f"[download] Unknown lang code: {lang_code}. Supported: {list(LANG_MAP)}")
        return

    print(f"\n[download] {config_name} ({lang_code}) — target {n_samples} samples")
    print(f"[download] Streaming from ai4bharat/indicvoices_r/{config_name} ...")

    try:
        from datasets import Audio as HFAudio
        ds = load_dataset(
            "ai4bharat/indicvoices_r",
            config_name,
            split="train",
            streaming=True,
        )
        # Disable automatic audio decoding — we'll decode bytes with soundfile
        # (avoids torchcodec dependency which requires PyTorch)
        ds = ds.cast_column("audio", HFAudio(decode=False))
    except Exception as e:
        print(f"[download] Failed to load dataset: {e}")
        print("[download] Make sure you have accepted the license and run: huggingface-cli login")
        return

    wav_dir = out_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    seen = 0

    for sample in ds:
        seen += 1
        if seen % 500 == 0:
            print(f"  scanned {seen}, collected {len(collected)}/{n_samples} ...")

        # Duration filter (metadata field — no audio decode needed)
        dur = sample.get("duration", 0.0)
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue

        # SNR quality filter
        snr = sample.get("snr", 0.0)
        if snr is not None and snr < MIN_SNR:
            continue

        text = (sample.get("normalized") or sample.get("text") or "").strip()
        if not text:
            continue

        audio_data = sample.get("audio", {})
        if not audio_data or not audio_data.get("bytes"):
            continue

        collected.append({
            "text":       text,
            "audio_data": audio_data,
            "duration":   dur,
        })

        if len(collected) >= n_samples:
            break

    if not collected:
        print(f"[download] No samples collected after scanning {seen} entries.")
        return

    print(f"[download] Collected {len(collected)} samples. Saving audio ...")

    # Shuffle for random val split
    rng = random.Random(seed)
    rng.shuffle(collected)
    n_val = max(1, int(len(collected) * VAL_FRAC))
    val_set   = collected[:n_val]
    train_set = collected[n_val:]

    def save_split(items, split_name):
        records = []
        for i, item in enumerate(items):
            fname = f"{lang_code}_{split_name}_{i:05d}.wav"
            fpath = wav_dir / fname

            audio_dict = item["audio_data"]
            import io
            raw_bytes = audio_dict.get("bytes")
            arr, sr = sf.read(io.BytesIO(raw_bytes))
            arr = arr.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)

            # Resample to 24kHz if needed
            if sr != 24000:
                try:
                    import scipy.signal as ss
                    arr = ss.resample_poly(arr, 24000, sr).astype(np.float32)
                    sr  = 24000
                except Exception:
                    pass  # keep original sr; audio_utils will resample later

            sf.write(str(fpath), arr, sr)
            records.append({
                "audio":     f"audio/{fname}",  # relative to JSONL location
                "text":      item["text"],
                "lang_code": lang_code,
            })

        jsonl_path = out_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} → {jsonl_path}")
        return jsonl_path

    save_split(train_set, "train")
    save_split(val_set,   "val")
    print(f"[download] Done. Next step:")
    print(f"  python scripts/preprocess_dataset.py \\")
    print(f"      --input {out_dir}/train.jsonl \\")
    print(f"      --output {out_dir}/train_codes.jsonl \\")
    print(f"      --model_id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")


def main():
    parser = argparse.ArgumentParser(description="Download IndicVoices-R data for training")
    parser.add_argument("--lang",    nargs="+", required=True,
                        help="Language code(s): kn mr ta te hi bn gu ml pa")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Max samples per language (default: 2000)")
    parser.add_argument("--out_dir", default="./data",
                        help="Root output directory (default: ./data)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    for lang in args.lang:
        out = Path(args.out_dir) / lang
        download_language(lang, args.samples, out, args.seed)

    print("\n[download] All done.")
    print("Run preprocess_dataset.py for each language, then:")
    print("  python scripts/prepare_multilingual.py  (combines all into one JSONL)")


if __name__ == "__main__":
    main()
