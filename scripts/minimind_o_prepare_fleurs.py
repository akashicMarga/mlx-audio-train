"""
Download and prepare Google FLEURS Hindi data for MiniMind-O training.

Source: google/fleurs  (hi_in config)
  Train: 2120 clips, mean 11.3s — 82% are under 12s (perfect for alignment)
  Test:  ~400 clips (used as val)

Why FLEURS over IndicSUPERB:
  IndicSUPERB mean duration = 21s (news paragraphs, overflow 512-token context)
  FLEURS mean duration      = 11s (single sentences, fit cleanly in context)
  Better gradient signal per step → faster alignment convergence.

Strategy:
  Stream the train.tar.gz (1.2 GB) via HTTP — no need to store it fully.
  Extract only clips with duration < max_dur_s.
  Write directly to Parquet (flat large_binary schema).

Usage:
    # Default: clips ≤12s, up to 2000 train + 200 val
    python scripts/minimind_o_prepare_fleurs.py --output data/fleurs_hindi

    # More data
    python scripts/minimind_o_prepare_fleurs.py \\
        --output    data/fleurs_hindi \\
        --max-dur   15 \\
        --max-train 2000 \\
        --max-val   300
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ID = "google/fleurs"
SPLIT_FILES = {
    "train": ("data/hi_in/audio/train.tar.gz", "data/hi_in/train.tsv"),
    "test":  ("data/hi_in/audio/test.tar.gz",  "data/hi_in/test.tsv"),
}

SYSTEM_PROMPTS = [
    "You are a helpful Hindi voice assistant.",
    "You are a knowledgeable AI that understands Hindi speech.",
    "You are a friendly assistant that listens and responds accurately.",
]


# ---------------------------------------------------------------------------
# TSV parsing — columns: id, filename, raw_transcription, normalized, phonemes, duration_samples, gender
# ---------------------------------------------------------------------------

def parse_tsv(tsv_path: str, max_dur_s: float) -> dict[str, tuple[str, float]]:
    """
    Returns {filename_stem: (transcription, duration_s)} for clips under max_dur_s.
    Uses the normalized transcription (cleaner for training).
    """
    clips = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            try:
                fname    = parts[1]                    # e.g. "123456789.wav"
                raw_text = parts[2].strip()
                norm_text = parts[3].strip() if len(parts) > 3 else raw_text
                dur_s    = int(parts[5]) / 16000.0
            except (ValueError, IndexError):
                continue
            if dur_s <= max_dur_s and norm_text:
                clips[fname] = (norm_text, dur_s)
    return clips


# ---------------------------------------------------------------------------
# Stream tar.gz and extract matching clips
# ---------------------------------------------------------------------------

def stream_extract(
    tar_repo_path: str,
    clip_meta: dict[str, tuple[str, float]],
    max_samples: int,
    seed: int = 42,
) -> list[dict]:
    """
    Download tar.gz via hf_hub_download (cached), then stream-extract matching clips.
    Returns list of row dicts.
    """
    from huggingface_hub import hf_hub_download

    print(f"  Downloading {tar_repo_path} (cached after first run) ...")
    local_tar = hf_hub_download(DATASET_ID, tar_repo_path, repo_type="dataset")
    print(f"  Extracting from {local_tar} ...")

    rows = []
    with tarfile.open(local_tar, mode="r:gz") as tf:
        for member in tf:
            if len(rows) >= max_samples:
                break
            fname = os.path.basename(member.name)
            if fname not in clip_meta:
                continue
            text, dur_s = clip_meta[fname]
            f = tf.extractfile(member)
            if f is None:
                continue
            wav_bytes = f.read()
            system = random.choice(SYSTEM_PROMPTS)
            convs  = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": ""},
                {"role": "assistant", "content": text},
            ]
            rows.append({
                "conversations":  json.dumps(convs, ensure_ascii=False),
                "question_audio": wav_bytes,
                "duration":       dur_s,
            })
            if len(rows) % 100 == 0:
                print(f"    {len(rows)} clips extracted ...", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        print(f"  [WARN] No rows for {path}")
        return

    table = pa.table({
        "conversations":  pa.array([r["conversations"]  for r in rows], type=pa.large_utf8()),
        "question_audio": pa.array([r["question_audio"] for r in rows], type=pa.large_binary()),
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table, path, compression="snappy", row_group_size=100)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare FLEURS Hindi for MiniMind-O (streaming)")
    p.add_argument("--output",    required=True, help="Output directory")
    p.add_argument("--max-dur",   type=float, default=12.0, help="Max clip duration in seconds")
    p.add_argument("--max-train", type=int,   default=2000)
    p.add_argument("--max-val",   type=int,   default=300)
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    from huggingface_hub import hf_hub_download

    for split, max_n, out_name in [
        ("train", args.max_train, "train.parquet"),
        ("test",  args.max_val,  "val.parquet"),
    ]:
        tar_path, tsv_repo_path = SPLIT_FILES[split]

        # Download TSV (tiny)
        print(f"\n[{split}] Downloading TSV ...")
        tsv_local = hf_hub_download(DATASET_ID, tsv_repo_path, repo_type="dataset")

        clip_meta = parse_tsv(tsv_local, args.max_dur)
        print(f"  {len(clip_meta)} clips ≤{args.max_dur}s in TSV")

        # Stream tar.gz, extract matching clips
        print(f"[{split}] Streaming audio tar.gz ...")
        rows = stream_extract(tar_path, clip_meta, max_n, args.seed)
        print(f"[{split}] Extracted {len(rows)} clips")

        if not rows:
            print(f"  [WARN] No rows for {split}")
            continue

        # Show duration stats
        import numpy as np
        durs = [r["duration"] for r in rows]
        print(f"  Duration: min={min(durs):.1f}s  max={max(durs):.1f}s  mean={np.mean(durs):.1f}s")

        write_parquet(rows, os.path.join(args.output, out_name))

    # Sample check
    import pyarrow.parquet as pq
    pf   = pq.ParquetFile(os.path.join(args.output, "train.parquet"), pre_buffer=False)
    tbl  = pf.read(columns=["conversations"])
    row0 = json.loads(tbl.to_pydict()["conversations"][0])
    print()
    print("Sample check (train[0]):")
    print(f"  system:    {row0[0]['content']}")
    print(f"  assistant: {row0[2]['content'][:80]}")

    print()
    print("Next — Phase 1a training on FLEURS:")
    print(f"  python scripts/minimind_o_phase1a_train.py \\")
    print(f"      --config configs/minimind_o/whisper_small_phase1a.yaml \\")
    print(f"      --data   {os.path.join(args.output, 'train.parquet')} \\")
    print(f"      --val-data {os.path.join(args.output, 'val.parquet')}")


if __name__ == "__main__":
    main()
