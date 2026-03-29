#!/usr/bin/env python3
"""
prepare_multilingual.py — Combine per-language preprocessed JSONL files into
a single train/val JSONL for multilingual training.

Expects each language to already have train_codes.jsonl and val_codes.jsonl
with a lang_code field (stamped by add_lang_code.py or download scripts).

Usage:
    python scripts/prepare_multilingual.py \\
        --langs hi kn mr ta te en \\
        --data_root ./data \\
        --output ./data/multilingual

Outputs:
    data/multilingual/train_codes.jsonl
    data/multilingual/val_codes.jsonl
    data/multilingual/stats.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs",     nargs="+", required=True,
                        help="Language codes to combine (e.g. hi kn mr ta te en)")
    parser.add_argument("--data_root", default="./data",
                        help="Root directory containing per-language folders")
    parser.add_argument("--output",    default="./data/multilingual",
                        help="Output directory")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    stats = defaultdict(lambda: {"train": 0, "val": 0})

    for split in ("train", "val"):
        all_records = []

        for lang in args.langs:
            # Try both naming conventions
            candidates = [
                root / lang / f"{split}_codes.jsonl",
                root / lang / f"{split}.jsonl",
            ]

            src = next((p for p in candidates if p.exists()), None)
            if src is None:
                print(f"[prepare] WARNING: no {split} data for '{lang}' — skipping")
                print(f"[prepare]   Looked in: {[str(c) for c in candidates]}")
                continue

            count = 0
            src_dir = src.parent.resolve()
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Resolve paths to absolute so the merged JSONL works from any location
                    for field in ("audio", "codec_path", "ref_audio"):
                        if field in rec and rec[field] and not Path(rec[field]).is_absolute():
                            rec[field] = str((src_dir / rec[field]).resolve())
                    # Ensure lang_code is set
                    rec.setdefault("lang_code", lang)
                    all_records.append(rec)
                    count += 1

            stats[lang][split] = count
            print(f"[prepare] {lang} {split}: {count} samples from {src}")

        # Shuffle combined records
        random.Random(args.seed).shuffle(all_records)

        out_path = out_dir / f"{split}_codes.jsonl"
        with open(out_path, "w") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[prepare] Combined {split}: {len(all_records)} total → {out_path}")

    # Save stats
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2)
    print(f"\n[prepare] Stats saved to {stats_path}")
    print("\nLanguage breakdown:")
    for lang, s in sorted(stats.items()):
        print(f"  {lang}: {s['train']} train / {s['val']} val")

    print(f"\nNext step: train with")
    print(f"  python scripts/train.py --config configs/qwen3_tts_multilingual.yaml")


if __name__ == "__main__":
    main()
