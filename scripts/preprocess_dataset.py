#!/usr/bin/env python3
"""
preprocess_dataset.py — Pre-tokenize audio files to codec IDs.

Mirrors the official Qwen3-TTS prepare_data.py but runs on Apple Silicon via MLX.

WHY: Running the speech tokenizer during training is slow and causes OOM because
the MLX computation graph accumulates across thousands of encodes. Pre-tokenizing
once saves codec IDs to .npy files, so training loads integers from disk instead.

Usage:
    # Pre-tokenize the Hindi dataset (run once before training)
    python scripts/preprocess_dataset.py --input data/hindi/train.jsonl
    python scripts/preprocess_dataset.py --input data/hindi/val.jsonl

    # Or process both at once
    python scripts/preprocess_dataset.py \\
        --input data/hindi/train.jsonl data/hindi/val.jsonl

Output:
    Each audio/hi_00001.wav → audio/hi_00001.codec.npy  (int32 [T_codec])
    data/hindi/train.jsonl  → data/hindi/train_codes.jsonl  (adds codec_path field)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def encode_dataset(
    jsonl_path: str,
    model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    sample_rate: int = 24000,
    overwrite: bool = False,
):
    import mlx.core as mx

    jsonl_path = Path(jsonl_path)
    out_path = jsonl_path.parent / (jsonl_path.stem + "_codes.jsonl")

    # Load speech tokenizer via model
    print(f"[preprocess] Loading speech tokenizer from: {model_id}")
    from mlx_audio.tts.utils import load_model as mlx_load
    model = mlx_load(model_id)
    speech_tokenizer = model.speech_tokenizer

    # Parse JSONL
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[preprocess] {len(records)} records to process from {jsonl_path.name}")
    base_dir = jsonl_path.parent

    encoded   = 0
    skipped   = 0
    errors    = 0
    out_lines = []

    for i, rec in enumerate(records):
        audio_rel = rec.get("audio", "")
        audio_abs = base_dir / audio_rel if not Path(audio_rel).is_absolute() else Path(audio_rel)

        # Where to save codec IDs
        codec_path = audio_abs.with_suffix(".codec.npy")
        codec_rel  = str(codec_path.relative_to(base_dir))

        if codec_path.exists() and not overwrite:
            # Already encoded — just update the record
            rec["codec_path"] = codec_rel
            out_lines.append(rec)
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(records)}] (skip existing)")
            continue

        if not audio_abs.exists():
            print(f"  [SKIP] Audio not found: {audio_abs}")
            errors += 1
            out_lines.append(rec)  # keep record, just without codec_path
            continue

        try:
            # Load + resample using project's audio_utils (handles stereo→mono + resample)
            from data.audio_utils import load_audio
            audio, _ = load_audio(str(audio_abs), target_sr=sample_rate)

            # Encode — speech_tokenizer.encode() expects [batch, 1, samples]
            audio_mx = mx.array(audio)[None, None, :]      # [1, 1, T]
            codes    = speech_tokenizer.encode(audio_mx)   # [1, n_q, T_codec]
            codec_ids = codes[0, 0]
            mx.eval(codec_ids)                              # flush MLX graph NOW
            codec_np = np.array(codec_ids, dtype=np.int32)

            # Save
            np.save(str(codec_path), codec_np)
            rec["codec_path"] = codec_rel
            encoded += 1

        except Exception as e:
            print(f"  [ERROR] {audio_abs.name}: {e}")
            errors += 1

        out_lines.append(rec)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}]  encoded={encoded}  skipped={skipped}  errors={errors}")

    # Write output JSONL
    with open(out_path, "w") as f:
        for rec in out_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[preprocess] Done:")
    print(f"  Encoded:  {encoded}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Errors:   {errors}")
    print(f"  Output:   {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize audio to codec IDs")
    parser.add_argument("--input",    nargs="+", required=True, help="JSONL file(s) to process")
    parser.add_argument("--model-id", default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode even if .npy exists")
    args = parser.parse_args()

    for jsonl in args.input:
        print(f"\n{'='*60}")
        print(f"  Processing: {jsonl}")
        print(f"{'='*60}")
        out = encode_dataset(jsonl, model_id=args.model_id, overwrite=args.overwrite)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
