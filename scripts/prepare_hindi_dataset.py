#!/usr/bin/env python3
"""
prepare_hindi_dataset.py — Download and prepare Hindi TTS dataset.

Sources (all free, open, Hindi):
  1. IndicTTS Hindi (IIT Madras) — 10h female + male, studio quality
  2. LJ-Speech Hindi equivalent from OpenSLR
  3. Shrutilipi (CC-BY-4.0) — 6h Hindi broadcast speech
  4. ULCA Hindi TTS dataset — varied speakers

Usage:
    # Download IndicTTS Hindi (recommended, 10h, clean studio)
    python scripts/prepare_hindi_dataset.py --source indictts --output ./data/hindi

    # Download from HuggingFace (easiest)
    python scripts/prepare_hindi_dataset.py --source hf --output ./data/hindi

    # Use your own audio folder
    python scripts/prepare_hindi_dataset.py --source custom \
        --audio-dir /path/to/wavs --output ./data/hindi
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace source (easiest)
# ──────────────────────────────────────────────────────────────────────────────

def download_hf(output_dir: Path, max_samples: int = 2000):
    """
    Download Hindi TTS data from HuggingFace datasets.
    Uses: SPRINGLab/IndicTTS-Hindi — studio quality, IIT Madras TTS corpus
    """
    print("[prepare] Downloading from HuggingFace: SPRINGLab/IndicTTS-Hindi")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return []

    import io
    import soundfile as sf
    import numpy as np

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # cast_column to disable auto-decoding — avoids torchcodec dependency
    from datasets import Audio
    ds = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train", streaming=False)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"[prepare] Dataset size: {len(ds)} samples")

    records = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        # Save audio — item["audio"] is {"bytes": ..., "path": ...} when decode=False
        audio_path = audio_dir / f"hi_{i:05d}.wav"
        if not audio_path.exists():
            raw = item["audio"]
            if raw.get("bytes"):
                # Decode from raw bytes using soundfile
                audio_array, sr = sf.read(io.BytesIO(raw["bytes"]))
            elif raw.get("path") and os.path.exists(raw["path"]):
                audio_array, sr = sf.read(raw["path"])
            else:
                continue
            audio_array = audio_array.astype(np.float32)
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)  # stereo → mono
            sf.write(str(audio_path), audio_array, sr)

        text = item.get("text", item.get("sentence", item.get("transcript", "")))
        if text:
            records.append({
                # Store path relative to output_dir (where the JSONL will live)
                # base_dataset.py resolves relative paths against JSONL's parent dir
                "audio": f"audio/hi_{i:05d}.wav",
                "text":  text.strip(),
            })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{min(max_samples, len(ds))}")

    print(f"[prepare] Downloaded {len(records)} samples")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# IndicTTS source
# ──────────────────────────────────────────────────────────────────────────────

def download_indictts(output_dir: Path):
    """
    IndicTTS Hindi corpus from IIT Madras (OpenSLR 103).
    ~10h, 2 speakers (male + female), studio quality.
    """
    import urllib.request
    import tarfile

    url = "https://www.openslr.org/resources/103/Hindi.tar.gz"
    tar_path = output_dir / "Hindi.tar.gz"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"[prepare] Downloading IndicTTS Hindi from OpenSLR (~1.2GB)...")
        urllib.request.urlretrieve(url, tar_path)

    print("[prepare] Extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(output_dir)

    return _scan_directory(output_dir / "Hindi")


# ──────────────────────────────────────────────────────────────────────────────
# Custom audio folder
# ──────────────────────────────────────────────────────────────────────────────

def process_custom(audio_dir: str, output_dir: Path) -> List[dict]:
    """
    Process a folder of WAV files with matching text files.

    Expected structure:
        audio_dir/
            001.wav
            001.txt     (or metadata.csv)
            002.wav
            002.txt

    Or with a metadata.csv:
        audio_dir/
            wavs/
                001.wav
            metadata.csv    (filename|text or filename,text)
    """
    audio_path = Path(audio_dir)

    # Try metadata.csv first
    meta_path = audio_path / "metadata.csv"
    if meta_path.exists():
        return _parse_metadata_csv(audio_path, meta_path)

    # Scan for wav+txt pairs
    return _scan_directory(audio_path)


def _scan_directory(path: Path) -> List[dict]:
    records = []
    for wav in sorted(path.rglob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").strip()
            if text:
                records.append({"audio": str(wav), "text": text})
    print(f"[prepare] Found {len(records)} wav+txt pairs in {path}")
    return records


def _parse_metadata_csv(audio_path: Path, meta_path: Path) -> List[dict]:
    import csv
    records = []
    with open(meta_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            fname, text = row[0].strip(), row[1].strip()
            wav = audio_path / "wavs" / f"{fname}.wav"
            if not wav.exists():
                wav = audio_path / f"{fname}.wav"
            if wav.exists() and text:
                records.append({"audio": str(wav), "text": text})
    print(f"[prepare] Parsed {len(records)} records from metadata.csv")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Train/val split + JSONL export
# ──────────────────────────────────────────────────────────────────────────────

def split_and_save(
    records:    List[dict],
    output_dir: Path,
    val_ratio:  float = 0.05,
    seed:       int   = 42,
) -> Tuple[Path, Path]:
    random.seed(seed)
    random.shuffle(records)

    n_val   = max(1, int(len(records) * val_ratio))
    val     = records[:n_val]
    train   = records[n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[prepare] ✅ Dataset ready:")
    print(f"   Train: {len(train)} samples → {train_path}")
    print(f"   Val:   {len(val)} samples   → {val_path}")

    # Stats
    total_chars = sum(len(r["text"]) for r in records)
    avg_chars   = total_chars / len(records) if records else 0
    print(f"   Avg text length: {avg_chars:.0f} chars")

    return train_path, val_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      choices=["hf", "indictts", "custom"], default="hf")
    parser.add_argument("--output",      default="./data/hindi")
    parser.add_argument("--audio-dir",   default=None,  help="For --source custom")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--val-ratio",   type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.source == "hf":
        records = download_hf(output_dir, args.max_samples)
    elif args.source == "indictts":
        records = download_indictts(output_dir)
    elif args.source == "custom":
        if not args.audio_dir:
            print("ERROR: --audio-dir required for --source custom")
            return
        records = process_custom(args.audio_dir, output_dir)

    if not records:
        print("[prepare] No records found. Exiting.")
        return

    split_and_save(records, output_dir, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
