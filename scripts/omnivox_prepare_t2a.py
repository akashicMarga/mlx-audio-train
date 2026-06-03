"""
Prepare Hindi T2A (text-to-audio) training data for OmniVox.

Reads hindi/train.jsonl, encodes each audio file with the Mimi codec
(8 codebooks × T_frames @ 12.5fps, 24kHz), and writes a Parquet file
that omnivox_t2a_train.py can directly consume.

Data format produced:
  conversations  (large_utf8):  JSON chat turns — system + user(text) + assistant(text)
  answer_audios  (large_binary): flat interleaved Mimi codes stored as int32 bytes
                                 layout: [cb0_f0, cb1_f0, ..., cb7_f0, cb0_f1, ...]
                                 shape equivalent: (n_frames × 8,)

Why T2A first:
  The Talker is randomly initialised. Without T2A training it cannot generate
  any coherent speech. T2A teaches Qwen+Talker to produce Hindi Mimi codes
  from Hindi text — once that works, A2A (speech input via Whisper) is layered on top.

Usage:
    python scripts/omnivox_prepare_t2a.py \\
        --input  /Users/akashsingh/Documents/exps/hindi/train.jsonl \\
        --audio-dir /Users/akashsingh/Documents/exps/hindi \\
        --output /Users/akashsingh/Documents/exps/omnivox_t2a/train.parquet

    python scripts/omnivox_prepare_t2a.py \\
        --input  /Users/akashsingh/Documents/exps/hindi/val.jsonl \\
        --audio-dir /Users/akashsingh/Documents/exps/hindi \\
        --output /Users/akashsingh/Documents/exps/omnivox_t2a/val.parquet
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SYSTEM_PROMPTS = [
    "आप एक सहायक हिंदी वॉयस असिस्टेंट हैं।",
    "You are a helpful Hindi voice assistant.",
    "आप एक ज्ञानवान AI हैं जो हिंदी में बोलते हैं।",
]


# ---------------------------------------------------------------------------
# Mimi encoder (lazy-loaded, singleton)
# ---------------------------------------------------------------------------

_mimi = None

def get_mimi():
    global _mimi
    if _mimi is None:
        print("Loading Mimi codec (kyutai/moshiko-mlx-bf16) ...", end=" ", flush=True)
        from mlx_audio.codec import Mimi
        _mimi = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")
        print(f"OK  ({_mimi.sample_rate}Hz, {_mimi.frame_rate}fps, 8 codebooks)")
    return _mimi


def encode_audio_to_mimi(wav_path: str) -> np.ndarray | None:
    """
    Load a WAV file and encode to Mimi codes.

    Returns: np.ndarray shape (8, T_frames) int32
             or None if the file cannot be read / encoded.
    """
    import mlx.core as mx
    import soundfile as sf
    import librosa

    try:
        wav, sr = sf.read(wav_path)
    except Exception as e:
        print(f"\n  [WARN] Cannot read {wav_path}: {e}", flush=True)
        return None

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    mimi   = get_mimi()
    target = mimi.sample_rate  # 24000

    if sr != target:
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=target)

    wav  = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak < 0.001:
        return None   # silent / empty clip
    wav = wav / peak * 0.9

    wav_mx = mx.array(wav)[None, None, :]         # (1, 1, samples)
    codes  = mimi.encode(wav_mx)                  # (1, n_codebooks, n_frames)
    mx.eval(codes)

    codes_np = np.array(codes[0, :8, :].tolist(), dtype=np.int32)  # (8, T_frames)
    return codes_np


def codes_to_flat_bytes(codes: np.ndarray) -> bytes:
    """
    (8, T) int32 array → flat interleaved bytes stored as large_binary.
    Layout: [cb0_f0, cb1_f0, ..., cb7_f0,  cb0_f1, ...]
    """
    # Interleave: transpose to (T, 8) then flatten
    flat = codes.T.flatten().astype(np.int32)
    return flat.tobytes()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare Hindi T2A parquet for OmniVox")
    p.add_argument("--input",     required=True, help="JSONL file: {audio, text} per line")
    p.add_argument("--audio-dir", required=True, help="Root dir containing audio/ subfolder")
    p.add_argument("--output",    required=True, help="Output parquet path")
    p.add_argument("--max",       type=int, default=0, help="Max samples (0 = all)")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # Load JSONL
    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.max > 0:
        rows = rows[:args.max]
    print(f"Loaded {len(rows)} samples from {args.input}", flush=True)

    # Encode
    records     = []
    n_skipped   = 0
    t_start     = time.time()

    for i, row in enumerate(rows):
        audio_rel = row["audio"]          # e.g. "audio/hi_01750.wav"
        text      = row["text"].strip()
        wav_path  = os.path.join(args.audio_dir, audio_rel)

        codes = encode_audio_to_mimi(wav_path)   # (8, T) or None
        if codes is None or codes.shape[1] < 4:
            n_skipped += 1
            continue

        system = random.choice(SYSTEM_PROMPTS)
        convs  = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": text},
            {"role": "assistant", "content": text},   # T2A: same text spoken back
        ]

        records.append({
            "conversations": json.dumps(convs, ensure_ascii=False),
            "answer_audios": codes_to_flat_bytes(codes),
            "n_frames":      codes.shape[1],
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(rows)}]  encoded={len(records)}  "
                  f"skipped={n_skipped}  {elapsed:.0f}s", flush=True)

    print(f"\nEncoded {len(records)} samples  ({n_skipped} skipped)", flush=True)

    if not records:
        print("ERROR: no samples produced — check audio paths", flush=True)
        sys.exit(1)

    # Duration stats
    frames = [r["n_frames"] for r in records]
    fps    = 12.5
    durs   = [f / fps for f in frames]
    print(f"Mimi frames: min={min(frames)}  max={max(frames)}  mean={np.mean(frames):.0f}", flush=True)
    print(f"Duration:    min={min(durs):.1f}s  max={max(durs):.1f}s  mean={np.mean(durs):.1f}s", flush=True)

    # Write parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table = pa.table({
        "conversations": pa.array([r["conversations"]  for r in records], type=pa.large_utf8()),
        "answer_audios": pa.array([r["answer_audios"]  for r in records], type=pa.large_binary()),
    })
    pq.write_table(table, args.output, compression="snappy", row_group_size=100)
    print(f"\nWrote {len(records)} rows → {args.output}", flush=True)

    # Sanity check
    pf  = pq.ParquetFile(args.output)
    row = pf.read(columns=["conversations", "answer_audios"]).to_pydict()
    c0  = json.loads(row["conversations"][0])
    a0  = np.frombuffer(row["answer_audios"][0], dtype=np.int32)
    print(f"\nSanity check [0]:", flush=True)
    print(f"  system:    {c0[0]['content']}", flush=True)
    print(f"  text:      {c0[1]['content'][:60]}", flush=True)
    print(f"  codes len: {len(a0)}  (= {len(a0)//8} frames × 8 codebooks)", flush=True)
    print(f"  code range: {a0.min()} – {a0.max()}", flush=True)


if __name__ == "__main__":
    main()
