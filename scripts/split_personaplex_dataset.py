#!/usr/bin/env python3
"""
split_personaplex_dataset.py — Split a prepared PersonaPlex dataset into train/val dirs.

Copies manifest subsets plus the referenced token and wav files so training can
use a real held-out validation directory.

Usage:
    python scripts/split_personaplex_dataset.py \
      --src /path/to/hindi_data_real_paired \
      --train-dst /path/to/hindi_data_real_paired_train \
      --val-dst /path/to/hindi_data_real_paired_val
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def _copy_sample(sample: dict, src_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "tokens").mkdir(parents=True, exist_ok=True)
    (dst_dir / "wavs").mkdir(parents=True, exist_ok=True)

    sample_id = sample["id"]
    src_token = src_dir / "tokens" / f"{sample_id}.npz"
    dst_token = dst_dir / "tokens" / f"{sample_id}.npz"
    if src_token.exists():
        shutil.copy2(src_token, dst_token)

    for key in ("audio_file", "user_audio_file", "assistant_audio_file"):
        rel = sample.get(key)
        if not rel:
            continue
        src_file = src_dir / rel
        dst_file = dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.exists():
            shutil.copy2(src_file, dst_file)


def _write_split(samples: list[dict], src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    for sample in samples:
        _copy_sample(sample, src_dir, dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Split a prepared PersonaPlex dataset into train/val subsets")
    parser.add_argument("--src", required=True, help="Prepared PersonaPlex dataset directory")
    parser.add_argument("--train-dst", required=True, help="Output directory for training split")
    parser.add_argument("--val-dst", required=True, help="Output directory for validation split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples to place in val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(args.src)
    manifest_path = src_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    rng = random.Random(args.seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]

    _write_split(train_samples, src_dir, Path(args.train_dst))
    _write_split(val_samples, src_dir, Path(args.val_dst))

    print(f"Source samples : {len(samples)}")
    print(f"Train samples  : {len(train_samples)} -> {args.train_dst}")
    print(f"Val samples    : {len(val_samples)} -> {args.val_dst}")


if __name__ == "__main__":
    main()
