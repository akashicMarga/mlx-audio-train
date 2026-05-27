#!/usr/bin/env python3
"""
prepare_ohf_voice.py — Download and convert OHF-Voice to JSONL for LFM training.

Mirrors the official Liquid4All cookbook preprocessing (examples/voice-assistant).

Dataset: Paulescu/OHF-Voice-audio-20260504  (HuggingFace)
  55,302 (audio, function-call) pairs, 24kHz WAV
  Format: audio_chat field with user (audio) + assistant (text) turns
  Task:   ASR — "Perform ASR." → transcribe audio to function call string

Output JSONL format (one line per sample):
  {"audio": "data/ohf_voice/train/000001.wav", "text": "HassStartTimer|$minutes=5|$name=oven"}

Usage:
    # Requires HuggingFace token (dataset is gated)
    export HF_TOKEN=hf_...

    python scripts/prepare_ohf_voice.py --output-dir data/ohf_voice

    # Limit samples for a quick test run
    python scripts/prepare_ohf_voice.py --output-dir data/ohf_voice --max-samples 500

    # After this, pre-tokenize with Mimi codec:
    python scripts/preprocess_lfm_audio.py \\
        --input  data/ohf_voice/train.jsonl \\
        --model-id mlx-community/LFM2.5-Audio-1.5B-8bit

    # Then train:
    python scripts/train.py --config configs/lfm_audio_asr.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Prepare OHF-Voice dataset for LFM training")
    p.add_argument("--output-dir",  default="data/ohf_voice",
                   help="Directory to save WAV files and JSONL (default: data/ohf_voice)")
    p.add_argument("--dataset-repo", default="Paulescu/OHF-Voice-audio-20260504",
                   help="HuggingFace dataset repo (default: Paulescu/OHF-Voice-audio-20260504)")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of train split for validation (default: 0.05)")
    p.add_argument("--max-samples",  type=int,   default=None,
                   help="Limit total samples (for quick test runs)")
    p.add_argument("--sample-rate",  type=int,   default=24000,
                   help="Output WAV sample rate (default: 24000 — matches Mimi)")
    p.add_argument("--hf-token",     default=None,
                   help="HuggingFace token (also reads HF_TOKEN env var)")
    return p.parse_args()


def extract_sample(row):
    """
    Extract (audio_array, sample_rate, text) from an OHF-Voice row.

    Each row has an 'audio_chat' field: list of {role, content[]} dicts.
    The user turn contains audio, the assistant turn contains the text label.
    """
    audio_arr = None
    sr        = None
    text      = None

    for msg in row.get("audio_chat", []):
        role    = msg.get("role", "")
        content = msg.get("content", [])
        for item in content:
            modality = item.get("modality", item.get("type", ""))
            if role == "user" and modality == "audio":
                audio_data = item.get("audio", {})
                if isinstance(audio_data, dict):
                    audio_arr = audio_data.get("array")
                    sr        = audio_data.get("sampling_rate", 24000)
                elif hasattr(audio_data, "array"):
                    audio_arr = audio_data.array
                    sr        = getattr(audio_data, "sampling_rate", 24000)
            elif role == "assistant" and modality == "text":
                text = item.get("text", "")

    return audio_arr, sr, text


def resample_audio(audio, src_sr: int, dst_sr: int):
    """Resample audio to dst_sr using scipy."""
    if src_sr == dst_sr:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(dst_sr, src_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype("float32")


def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[prepare] Warning: no HF_TOKEN found. Dataset may require authentication.")
        print("[prepare] Set HF_TOKEN=hf_... or pass --hf-token")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[prepare] ERROR: `datasets` package not found. Run: pip install datasets")
        sys.exit(1)

    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("[prepare] ERROR: soundfile/numpy not found. Run: pip install soundfile numpy")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    train_wav_dir = out_dir / "train"
    val_wav_dir   = out_dir / "val"
    train_wav_dir.mkdir(parents=True, exist_ok=True)
    val_wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Loading dataset: {args.dataset_repo}")
    load_kw = dict(trust_remote_code=True)
    if hf_token:
        load_kw["token"] = hf_token

    # Official cookbook uses train split only (prevents test contamination)
    ds = load_dataset(args.dataset_repo, split="train", **load_kw)

    total = len(ds)
    if args.max_samples:
        total = min(total, args.max_samples)
        ds = ds.select(range(total))

    # Deterministic 95/5 split matching the official cookbook
    split = ds.train_test_split(test_size=args.val_fraction, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"[prepare] {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"[prepare] Saving WAVs to: {out_dir}")

    def process_split(dataset, wav_dir: Path, jsonl_path: Path, split_name: str):
        records = []
        skipped = 0
        for idx, row in enumerate(dataset):
            audio_arr, sr, text = extract_sample(row)

            if audio_arr is None or text is None or not str(text).strip():
                skipped += 1
                continue

            audio_np = np.array(audio_arr, dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)

            # Resample to target sample rate
            if sr and sr != args.sample_rate:
                audio_np = resample_audio(audio_np, sr, args.sample_rate)

            wav_path = wav_dir / f"{idx:06d}.wav"
            sf.write(str(wav_path), audio_np, args.sample_rate)

            records.append({"audio": str(wav_path), "text": text.strip()})

            if (idx + 1) % 500 == 0:
                print(f"  [{split_name}] {idx+1}/{len(dataset)} processed ...")

        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"[prepare] {split_name}: {len(records)} saved, {skipped} skipped → {jsonl_path}")
        return records

    train_records = process_split(train_ds, train_wav_dir,
                                  out_dir / "train.jsonl", "train")
    val_records   = process_split(val_ds,   val_wav_dir,
                                  out_dir / "val.jsonl",   "val")

    # Print a few examples so the user can verify the format
    print("\n[prepare] Sample records:")
    for rec in train_records[:3]:
        print(f"  audio={Path(rec['audio']).name}  text={rec['text'][:60]}")

    print(f"\n[prepare] Done.")
    print(f"  Train JSONL: {out_dir}/train.jsonl  ({len(train_records)} samples)")
    print(f"  Val   JSONL: {out_dir}/val.jsonl    ({len(val_records)} samples)")
    print()
    print("Next steps:")
    print(f"  1. Pre-tokenize (saves .codec.npy alongside WAVs — run once):")
    print(f"     python scripts/preprocess_lfm_audio.py \\")
    print(f"         --input {out_dir}/train.jsonl \\")
    print(f"         --model-id mlx-community/LFM2.5-Audio-1.5B-8bit \\")
    print(f"         --save-mel")
    print(f"  2. Train:")
    print(f"     python scripts/train.py --config configs/lfm_audio_asr.yaml")


if __name__ == "__main__":
    main()
