"""
Prepare T2A training parquet from FLEURS Hindi + existing Hindi clips.

Sources:
  1. FLEURS Hindi parquet  (audio embedded as bytes, 16kHz)
  2. Existing hindi/train.jsonl  (WAV files on disk, any sample rate)

Output: same schema as omnivox_prepare_t2a.py — ready for omnivox_t2a_train.py.

Usage:
    # Full combined train set
    python scripts/omnivox_prepare_fleurs.py \\
        --fleurs-train /Users/akashsingh/Documents/exps/fleurs_hindi/parquet-data/hi_in/train-00000-of-00001.parquet \\
        --jsonl        /Users/akashsingh/Documents/exps/hindi/train.jsonl \\
        --audio-dir    /Users/akashsingh/Documents/exps/hindi \\
        --output       /Users/akashsingh/Documents/exps/omnivox_t2a_synth/train.parquet

    # Val set (FLEURS val only — keep hindi val separate or combined)
    python scripts/omnivox_prepare_fleurs.py \\
        --fleurs-train /Users/akashsingh/Documents/exps/fleurs_hindi/parquet-data/hi_in/validation-00000-of-00001.parquet \\
        --output       /Users/akashsingh/Documents/exps/omnivox_t2a_synth/val.parquet
"""

import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPTS = [
    "आप एक सहायक इंडिक वॉयस असिस्टेंट हैं।",
    "You are a helpful Indic voice assistant.",
    "आप एक ज्ञानवान AI हैं जो देवनागरी भाषाओं में बोलते हैं।",
]

MIN_FRAMES = 8     # ~0.64s  — skip silent/bad clips
MAX_FRAMES = 375   # ~30s    — skip very long clips


# ─────────────────────────────────────────────────────────
# Mimi encoder (singleton)
# ─────────────────────────────────────────────────────────

_mimi = None

def get_mimi():
    global _mimi
    if _mimi is None:
        print("Loading Mimi codec (kyutai/moshiko-mlx-bf16) ...", end=" ", flush=True)
        from mlx_audio.codec import Mimi
        _mimi = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")
        print(f"OK  ({_mimi.sample_rate}Hz, {_mimi.frame_rate}fps, 8 codebooks)", flush=True)
    return _mimi


def encode_wav_array(wav: np.ndarray, sr: int) -> "np.ndarray | None":
    """Encode a float32 mono wav array to (8, T) Mimi codes."""
    import mlx.core as mx
    import librosa

    mimi = get_mimi()
    if sr != mimi.sample_rate:
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=mimi.sample_rate)

    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak < 0.001:
        return None
    wav = wav / peak * 0.9

    wav_mx = mx.array(wav)[None, None, :]
    codes = mimi.encode(wav_mx)   # (1, 8, T)
    mx.eval(codes)
    return np.array(codes[0, :8, :].tolist(), dtype=np.int32)


def codes_to_flat_bytes(codes: np.ndarray) -> bytes:
    return codes.T.flatten().astype(np.int32).tobytes()


def make_record(text: str, codes: np.ndarray) -> dict:
    system = random.choice(SYSTEM_PROMPTS)
    convs = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": text},
        {"role": "assistant", "content": text},
    ]
    return {
        "conversations": json.dumps(convs, ensure_ascii=False),
        "answer_audios": codes_to_flat_bytes(codes),
        "n_frames":      codes.shape[1],
    }


# ─────────────────────────────────────────────────────────
# FLEURS source
# ─────────────────────────────────────────────────────────

def encode_fleurs_parquet(parquet_path: str) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf

    table = pq.read_table(parquet_path)
    n = len(table)
    print(f"\nFLEURS parquet: {parquet_path}  ({n} rows)", flush=True)

    records = []
    n_skip  = 0
    t0      = time.time()

    for i in range(n):
        audio_cell = table.column("audio")[i].as_py()
        text       = table.column("raw_transcription")[i].as_py()
        if not text or not audio_cell:
            n_skip += 1
            continue

        text = text.strip()
        if not text:
            n_skip += 1
            continue

        audio_bytes = audio_cell.get("bytes", b"")
        if not audio_bytes:
            n_skip += 1
            continue

        try:
            wav, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            print(f"\n  [WARN] Row {i}: {e}", flush=True)
            n_skip += 1
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        codes = encode_wav_array(wav, sr)
        if codes is None or not (MIN_FRAMES <= codes.shape[1] <= MAX_FRAMES):
            n_skip += 1
            continue

        records.append(make_record(text, codes))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}]  ok={len(records)}  skip={n_skip}  {elapsed:.0f}s",
                  flush=True)

    print(f"FLEURS done: {len(records)} records  ({n_skip} skipped)", flush=True)
    return records


# ─────────────────────────────────────────────────────────
# Existing JSONL + WAV source
# ─────────────────────────────────────────────────────────

def encode_jsonl(jsonl_path: str, audio_dir: str) -> list[dict]:
    import soundfile as sf

    with open(jsonl_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"\nJSONL: {jsonl_path}  ({len(rows)} rows)", flush=True)

    records = []
    n_skip  = 0
    t0      = time.time()

    for i, row in enumerate(rows):
        text = row.get("text", "").strip()
        if not text:
            n_skip += 1
            continue

        wav_path = os.path.join(audio_dir, row["audio"])
        try:
            wav, sr = sf.read(wav_path)
        except Exception as e:
            print(f"\n  [WARN] Row {i}: {e}", flush=True)
            n_skip += 1
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        codes = encode_wav_array(wav, sr)
        if codes is None or not (MIN_FRAMES <= codes.shape[1] <= MAX_FRAMES):
            n_skip += 1
            continue

        records.append(make_record(text, codes))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(rows)}]  ok={len(records)}  skip={n_skip}  {elapsed:.0f}s",
                  flush=True)

    print(f"JSONL done: {len(records)} records  ({n_skip} skipped)", flush=True)
    return records


# ─────────────────────────────────────────────────────────
# IISc SYSPIN source  (WAV dir + JSON transcripts)
# ─────────────────────────────────────────────────────────

def encode_syspin(wav_dir: str, transcript_json: str, max_clips: int = 0) -> list[dict]:
    import soundfile as sf

    with open(transcript_json) as f:
        meta = json.load(f)
    transcripts = meta.get("Transcripts", {})
    items = list(transcripts.items())
    if max_clips > 0:
        items = items[:max_clips]
    print(f"\nSYSPIN: {wav_dir}  ({len(items)} clips)", flush=True)

    records = []
    n_skip  = 0
    t0      = time.time()

    for i, (clip_id, entry) in enumerate(items):
        text = entry.get("Transcript", "").strip()
        if not text:
            n_skip += 1
            continue

        wav_path = os.path.join(wav_dir, clip_id + ".wav")
        if not os.path.exists(wav_path):
            n_skip += 1
            continue

        try:
            wav, sr = sf.read(wav_path)
        except Exception as e:
            print(f"\n  [WARN] {clip_id}: {e}", flush=True)
            n_skip += 1
            continue

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        codes = encode_wav_array(wav, sr)
        if codes is None or not (MIN_FRAMES <= codes.shape[1] <= MAX_FRAMES):
            n_skip += 1
            continue

        records.append(make_record(text, codes))

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(items) - i - 1) / rate
            print(f"  [{i+1}/{len(items)}]  ok={len(records)}  skip={n_skip}  "
                  f"{elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"SYSPIN done: {len(records)} records  ({n_skip} skipped)", flush=True)
    return records


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Prepare T2A parquet from FLEURS / JSONL / SYSPIN")
    p.add_argument("--fleurs-train", default=None,
                   help="FLEURS parquet with embedded audio bytes (train or val)")
    p.add_argument("--jsonl",      default=None,
                   help="Existing JSONL file with {audio, text} entries")
    p.add_argument("--audio-dir",  default="/Users/akashsingh/Documents/exps/hindi",
                   help="Root dir for JSONL audio paths")
    p.add_argument("--syspin-wav", default=None,
                   help="IISc SYSPIN wav/ directory")
    p.add_argument("--syspin-json", default=None,
                   help="IISc SYSPIN transcripts JSON file")
    p.add_argument("--syspin-max", type=int, default=0,
                   help="Max SYSPIN clips to use (0=all)")
    p.add_argument("--output",     required=True, help="Output parquet path")
    p.add_argument("--max",        type=int, default=0, help="Max total samples (0=all)")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_records: list[dict] = []

    if args.fleurs_train:
        all_records.extend(encode_fleurs_parquet(args.fleurs_train))

    if args.jsonl:
        all_records.extend(encode_jsonl(args.jsonl, args.audio_dir))

    if args.syspin_wav and args.syspin_json:
        all_records.extend(encode_syspin(args.syspin_wav, args.syspin_json,
                                         max_clips=args.syspin_max))

    if not all_records:
        print("ERROR: no sources provided / no records produced", flush=True)
        sys.exit(1)

    random.shuffle(all_records)
    if args.max > 0:
        all_records = all_records[:args.max]

    frames = [r["n_frames"] for r in all_records]
    durs   = [f / 12.5 for f in frames]
    total_h = sum(durs) / 3600
    print(f"\nTotal: {len(all_records)} records", flush=True)
    print(f"Duration: min={min(durs):.1f}s  max={max(durs):.1f}s  "
          f"mean={np.mean(durs):.1f}s  total={total_h:.2f}h", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table = pa.table({
        "conversations": pa.array([r["conversations"] for r in all_records], type=pa.large_utf8()),
        "answer_audios": pa.array([r["answer_audios"] for r in all_records], type=pa.large_binary()),
    })
    pq.write_table(table, args.output, compression="snappy", row_group_size=100)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote → {args.output}  ({size_mb:.0f} MB)", flush=True)

    # Sanity check
    pf  = pq.ParquetFile(args.output)
    row = pf.read(columns=["conversations", "answer_audios"]).to_pydict()
    c0  = json.loads(row["conversations"][0])
    a0  = np.frombuffer(row["answer_audios"][0], dtype=np.int32)
    print(f"Sanity [0]: text='{c0[1]['content'][:60]}'  "
          f"frames={len(a0)//8}  range=[{a0.min()},{a0.max()}]", flush=True)


if __name__ == "__main__":
    main()
