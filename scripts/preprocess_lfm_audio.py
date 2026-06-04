#!/usr/bin/env python3
"""
preprocess_lfm_audio.py — Pre-tokenize audio for LFM 2.5 Audio training.

Runs the Mimi neural codec over every audio file in the JSONL dataset and
saves the 8-codebook codes to a <stem>.codec.npy file alongside the audio.
Training then loads these instead of running Mimi live, which:
  - Eliminates the largest source of OOM during training
  - Makes data loading ~10× faster

Optionally also saves mel spectrograms (<stem>.mel.npy) for ASR training.

Usage:
    python scripts/preprocess_lfm_audio.py \\
        --input  data/train.jsonl \\
        --model-id mlx-community/LFM2.5-Audio-1.5B-8bit

    # Also save mel features (needed for ASR fine-tuning)
    python scripts/preprocess_lfm_audio.py \\
        --input  data/train.jsonl \\
        --model-id mlx-community/LFM2.5-Audio-1.5B-8bit \\
        --save-mel

    # Write updated JSONL (adds codec_path field)
    python scripts/preprocess_lfm_audio.py \\
        --input  data/train.jsonl \\
        --output data/train_codes.jsonl \\
        --model-id mlx-community/LFM2.5-Audio-1.5B-8bit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Pre-tokenize audio for LFM 2.5 training")
    p.add_argument("--input",    required=True, help="Input JSONL path")
    p.add_argument("--output",   default=None,  help="Output JSONL path (default: <input>_codes.jsonl)")
    p.add_argument("--model-id", default="mlx-community/LFM2.5-Audio-1.5B-8bit",
                   help="HuggingFace model ID for LFM2AudioProcessor")
    p.add_argument("--sample-rate", type=int, default=24000,
                   help="Target sample rate for Mimi codec (default: 24000)")
    p.add_argument("--save-mel", action="store_true",
                   help="Also save mel spectrograms for ASR training")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .codec.npy files")
    p.add_argument("--max-duration", type=float, default=30.0,
                   help="Skip audio longer than this many seconds")
    return p.parse_args()


def load_lfm_processor(model_id: str):
    """Load LFM2AudioProcessor from HuggingFace."""
    try:
        from mlx_audio.sts.models.lfm_audio import LFM2AudioProcessor
        print(f"[preprocess] Loading LFM2AudioProcessor: {model_id}")
        processor = LFM2AudioProcessor.from_pretrained(model_id)
        return processor
    except Exception as e:
        print(f"[preprocess] ERROR: could not load LFM2AudioProcessor: {e}")
        print("[preprocess] Install mlx-audio: pip install mlx-audio")
        sys.exit(1)


def encode_audio_codes(processor, audio, sr: int, target_sr: int = 24000):
    """
    Audio waveform → Mimi codes [T, 8].

    Resamples to target_sr if needed before passing to Mimi.
    """
    import numpy as np
    import mlx.core as mx

    audio_f32 = audio.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio_f32 = resample_poly(audio_f32, target_sr // g, sr // g).astype(np.float32)

    # tokenize_audio expects [B, 1, T] at 24kHz
    audio_mx = mx.array(audio_f32)[None, None, :]   # [1, 1, T]
    codes    = processor.tokenize_audio(audio_mx)    # [1, 8, T]
    mx.eval(codes)
    # Transpose to [T, 8]
    return np.array(codes[0]).T.astype(np.int32)


def encode_mel(processor, audio, sr: int, target_sr: int = 16000):
    """
    Audio waveform → mel features [T_mel, 128] for ConformerEncoder.

    Resamples to target_sr (16kHz) if needed.
    """
    import numpy as np
    import mlx.core as mx

    audio_f32 = audio.astype(np.float32)

    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio_f32 = resample_poly(audio_f32, target_sr // g, sr // g).astype(np.float32)

    audio_mx = mx.array(audio_f32)
    mel = processor.preprocess_audio(audio_mx)  # [T_mel, 128]
    mx.eval(mel)
    return np.array(mel, dtype=np.float32)  # convert mlx → numpy


def main():
    args = parse_args()

    import numpy as np
    import soundfile as sf

    processor = load_lfm_processor(args.model_id)

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_codes.jsonl"
    )

    base    = input_path.parent  # resolve relative audio paths against JSONL dir
    lines   = input_path.read_text().strip().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]

    print(f"[preprocess] {len(records)} samples → {output_path}")

    skipped   = 0
    processed = 0
    out_lines = []

    for idx, rec in enumerate(records):
        audio_path = rec.get("audio", "")
        if not audio_path:
            print(f"  [{idx+1}/{len(records)}] SKIP: no audio path")
            skipped += 1
            out_lines.append(json.dumps(rec))
            continue
        # Resolve relative to JSONL parent so both absolute and relative paths work
        audio_path = str(base / audio_path) if not Path(audio_path).is_absolute() else audio_path
        if not Path(audio_path).exists():
            print(f"  [{idx+1}/{len(records)}] SKIP: audio not found: {audio_path}")
            skipped += 1
            out_lines.append(json.dumps(rec))
            continue

        codec_npy = Path(audio_path).with_suffix(".codec.npy")
        mel_npy   = Path(audio_path).with_suffix(".mel.npy")

        need_codec = args.overwrite or not codec_npy.exists()
        need_mel   = args.save_mel and (args.overwrite or not mel_npy.exists())

        if not need_codec and not need_mel:
            print(f"  [{idx+1}/{len(records)}] SKIP (cached): {Path(audio_path).name}")
            rec["codec_path"] = str(codec_npy)
            out_lines.append(json.dumps(rec))
            processed += 1
            continue

        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            dur = len(audio) / sr
            if dur > args.max_duration:
                print(f"  [{idx+1}/{len(records)}] SKIP (too long {dur:.1f}s): {Path(audio_path).name}")
                skipped += 1
                out_lines.append(json.dumps(rec))
                continue

            if need_codec:
                codes = encode_audio_codes(processor, audio, sr, target_sr=args.sample_rate)
                np.save(str(codec_npy), codes)

            if need_mel:
                mel = encode_mel(processor, audio, sr, target_sr=16000)
                np.save(str(mel_npy), mel)  # encode_mel already returns np.ndarray

            rec["codec_path"] = str(codec_npy)
            if need_mel:
                rec["mel_path"] = str(mel_npy)

            t = len(np.load(str(codec_npy))) / 12.5 if codec_npy.exists() else 0
            print(f"  [{idx+1}/{len(records)}] OK  {Path(audio_path).name:40s}  "
                  f"{dur:.1f}s → {int(t * 12.5)} frames, {int(t * 12.5) * 8} codes")
            processed += 1

        except Exception as e:
            print(f"  [{idx+1}/{len(records)}] ERROR {audio_path}: {e}")
            skipped += 1

        out_lines.append(json.dumps(rec))

    output_path.write_text("\n".join(out_lines) + "\n")
    print(f"\n[preprocess] Done: {processed} OK, {skipped} skipped → {output_path}")


if __name__ == "__main__":
    main()
