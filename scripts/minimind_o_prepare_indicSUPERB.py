"""
Download and prepare IndicSUPERB Hindi data for MiniMind-O Phase 1a/1b training.

Source: collabora/indic-superb (HuggingFace, 24.9k train / 872 test)
  Each shard: ~265 rows, columns: audio{bytes, path}, transcription, duration

Bypasses the `datasets` library entirely — downloads raw Parquet shards via
huggingface_hub and reads them with pyarrow. No PyTorch dependency.

Output Parquet columns (MiniMind-O format):
  conversations   — JSON: [system, user(""), assistant(Hindi transcription)]
  question_audios — [[raw WAV bytes]] of the spoken audio
  answer_audios, image_bytes, ref_audios, spk_emb — empty lists

Training objective:
  User speaks X (audio) → Thinker predicts X (Hindi text)
  This is ASR-style alignment — the correct Phase 1a objective.

Usage:
    # Default: 2000 train + 200 val (~8 train shards + 1 test shard)
    python scripts/minimind_o_prepare_indicSUPERB.py --output data/indicSUPERB_hindi

    # More data
    python scripts/minimind_o_prepare_indicSUPERB.py \\
        --output    data/indicSUPERB_hindi \\
        --max-train 5000 \\
        --max-val   500
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_ID  = "collabora/indic-superb"
ROWS_PER_SHARD = 265   # empirical from shard 0

SYSTEM_PROMPTS = [
    "You are a helpful Hindi voice assistant.",
    "You are a knowledgeable AI that understands Hindi speech.",
    "You are a friendly assistant that listens and responds accurately.",
]


# ---------------------------------------------------------------------------
# Fetch shard names from HF
# ---------------------------------------------------------------------------

def list_shards(repo_id: str) -> tuple[list[str], list[str]]:
    """Return (train_shards, test_shards) sorted."""
    from huggingface_hub import HfApi
    api   = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    train = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    test  = sorted(f for f in files if f.startswith("data/test-")  and f.endswith(".parquet"))
    return train, test


# ---------------------------------------------------------------------------
# Convert one shard to training rows
# ---------------------------------------------------------------------------

def shard_to_rows(parquet_path: str, max_rows: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    tbl  = pq.read_table(parquet_path)
    data = tbl.to_pydict()
    n    = len(data["transcription"])
    if max_rows is not None:
        n = min(n, max_rows)

    rows = []
    for i in range(n):
        transcription = str(data["transcription"][i]).strip()
        if not transcription or len(transcription) < 3:
            continue

        audio_struct = data["audio"][i]
        if isinstance(audio_struct, dict):
            wav_bytes = audio_struct.get("bytes") or audio_struct.get("path")
        else:
            wav_bytes = audio_struct

        if not wav_bytes or not isinstance(wav_bytes, (bytes, bytearray)):
            continue

        system = random.choice(SYSTEM_PROMPTS)
        convs  = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": ""},
            {"role": "assistant", "content": transcription},
        ]

        rows.append({
            "conversations":  json.dumps(convs, ensure_ascii=False),
            "question_audio": bytes(wav_bytes),   # flat large_binary avoids chunked-list issue
        })
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
    # row_group_size=100 keeps binary chunks small enough to read back cleanly
    pq.write_table(table, path, compression="snappy", row_group_size=100)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare IndicSUPERB Hindi data (no datasets library)")
    p.add_argument("--output",    required=True, help="Output dir, e.g. data/indicSUPERB_hindi")
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val",   type=int, default=200)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--dataset",   default=DATASET_ID)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    print(f"Listing shards in {args.dataset} ...")
    train_shards, test_shards = list_shards(args.dataset)
    print(f"  {len(train_shards)} train shards, {len(test_shards)} test shards")

    # ── Download train shards until we have enough rows ────────────────────────
    train_rows = []
    n_shards_needed = (args.max_train // ROWS_PER_SHARD) + 2

    print(f"\nCollecting up to {args.max_train} train samples ...")
    for shard_file in train_shards[:n_shards_needed]:
        remaining = args.max_train - len(train_rows)
        if remaining <= 0:
            break
        print(f"  Downloading {shard_file} ...", end=" ", flush=True)
        local = hf_hub_download(args.dataset, shard_file, repo_type="dataset")
        rows  = shard_to_rows(local, max_rows=remaining)
        train_rows.extend(rows)
        print(f"{len(rows)} rows  (total {len(train_rows)})")

    print(f"Train: {len(train_rows)} samples collected")

    # ── Download test shards for val ───────────────────────────────────────────
    val_rows = []
    print(f"\nCollecting up to {args.max_val} val samples ...")
    for shard_file in test_shards:
        remaining = args.max_val - len(val_rows)
        if remaining <= 0:
            break
        print(f"  Downloading {shard_file} ...", end=" ", flush=True)
        local = hf_hub_download(args.dataset, shard_file, repo_type="dataset")
        rows  = shard_to_rows(local, max_rows=remaining)
        val_rows.extend(rows)
        print(f"{len(rows)} rows  (total {len(val_rows)})")

    print(f"Val: {len(val_rows)} samples collected")

    # ── Write output ───────────────────────────────────────────────────────────
    print()
    write_parquet(train_rows, os.path.join(args.output, "train.parquet"))
    write_parquet(val_rows,   os.path.join(args.output, "val.parquet"))

    # ── Sample check ───────────────────────────────────────────────────────────
    if train_rows:
        s    = train_rows[0]
        conv = json.loads(s["conversations"])
        print()
        print("Sample check:")
        print(f"  system:         {conv[0]['content']}")
        print(f"  assistant:      {conv[2]['content'][:80]}")
        print(f"  audio bytes:    {len(s['question_audio']):,}")

    print()
    print("Next — Phase 1a training:")
    print(f"  python scripts/minimind_o_phase1a_train.py \\")
    print(f"      --config configs/minimind_o/whisper_small_phase1a.yaml \\")
    print(f"      --data   {os.path.join(args.output, 'train.parquet')}")
    print()
    print("Then Phase 1b:")
    print(f"  python scripts/minimind_o_phase1b_train.py \\")
    print(f"      --config configs/minimind_o/whisper_small_phase1b.yaml \\")
    print(f"      --data   {os.path.join(args.output, 'train.parquet')}")


if __name__ == "__main__":
    main()
