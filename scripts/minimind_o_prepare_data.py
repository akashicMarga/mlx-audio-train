"""
MiniMind-O data preparation — converts audio + text pairs into the Parquet
format that OmniDataset (models/minimind_o/dataset.py) expects for training.

Input JSONL (one sample per line):
  {"text": "user question", "audio": "response.wav"}
  {"text": "hello", "audio": "hi_response.wav", "input_audio": "hi_input.wav"}
  {"conversation": [{"role":"user","content":"hi"}, {"role":"assistant","content":"hello"}], "audio": "hello.wav"}

Output: train.parquet + val.parquet with columns:
  conversations   — JSON chat turns
  answer_audios   — flat list of Mimi code ints (8 codebooks interleaved per frame)
  question_audios — optional: raw audio bytes for the user turn (speech input)

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    # Basic text→speech data (most common)
    python scripts/minimind_o_prepare_data.py \\
        --input  data/my_lang/samples.jsonl \\
        --output data/my_lang/train.parquet

    # With speech input (question_audios)
    python scripts/minimind_o_prepare_data.py \\
        --input  data/my_lang/samples.jsonl \\
        --output data/my_lang/train.parquet \\
        --with-input-audio

    # Split into train/val
    python scripts/minimind_o_prepare_data.py \\
        --input  data/my_lang/samples.jsonl \\
        --output data/my_lang/ \\
        --val-split 0.05

JSONL format options
────────────────────
Simple (text question, audio answer):
  {"text": "What is the capital of France?", "audio": "paris.wav"}

With spoken input:
  {"text": "user text", "audio": "response.wav", "input_audio": "question.wav"}

Multi-turn (use conversation key):
  {"conversation": [
      {"role": "user",      "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ],
    "audio": "hi_there.wav"
  }

System prompt:
  {"text": "...", "audio": "...", "system": "You are a helpful Hindi assistant."}
"""

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Mimi encoder (mlx-audio)
# ---------------------------------------------------------------------------

_mimi = None

def get_mimi():
    global _mimi
    if _mimi is None:
        print("Loading Mimi codec ...", end=" ", flush=True)
        from mlx_audio.codec import Mimi
        _mimi = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")
        print(f"OK  ({_mimi.sample_rate}Hz, {_mimi.frame_rate}fps, 8 codebooks)")
    return _mimi


def wav_to_mimi_codes(wav_path: str) -> Optional[list[int]]:
    """
    Load a WAV file and encode it to a flat list of Mimi code integers.

    Layout: [cb0_f0, cb1_f0, ..., cb7_f0, cb0_f1, cb1_f1, ..., cb7_fN]
    i.e. 8 codes per frame, frames in time order.
    This is the interleaved format OmniDataset expects in answer_audios.
    """
    import mlx.core as mx
    import soundfile as sf
    import librosa

    try:
        wav, sr = sf.read(wav_path)
    except Exception as e:
        print(f"\n  [WARN] Cannot read {wav_path}: {e}")
        return None

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    mimi = get_mimi()
    target_sr = mimi.sample_rate  # 24000

    if sr != target_sr:
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=target_sr)

    wav = wav.astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0.01:
        wav = wav / peak * 0.9

    wav_mx = mx.array(wav)[None, None, :]    # (1, 1, samples)
    codes  = mimi.encode(wav_mx)             # (1, n_codebooks, n_frames) — mlx-audio returns array directly
    mx.eval(codes)

    # Use first 8 codebooks (acoustic stream used by MiniMind-O Talker)
    codes_np = np.array(codes[0, :8, :].tolist(), dtype=np.int32)  # (8, n_frames)
    n_frames  = codes_np.shape[1]

    # Interleave: [cb0_f0, cb1_f0, ..., cb7_f0, cb0_f1, ...]
    flat = codes_np.T.flatten().tolist()   # (n_frames * 8,) ints
    return flat


def audio_to_bytes(wav_path: str) -> Optional[bytes]:
    """Load a WAV file and return raw bytes for question_audios column."""
    try:
        with open(wav_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"\n  [WARN] Cannot read {wav_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPTS = [
    "You are a helpful voice assistant.",
    "You are a knowledgeable AI. Answer the user's questions accurately.",
    "You are a friendly assistant. Reply concisely.",
]


def build_sample(
    item: dict,
    base_dir: str,
    with_input_audio: bool = False,
) -> Optional[dict]:
    """
    Convert one JSONL record into an OmniDataset row.

    Returns None if the audio file is missing or encoding fails.
    """

    # ── Build conversation turns ────────────────────────────────────────────
    if "conversation" in item:
        # Pre-formatted multi-turn
        conversations = item["conversation"]
    else:
        text   = item.get("text", "").strip()
        system = item.get("system", "")
        conversations = []
        if system:
            conversations.append({"role": "system", "content": system})
        elif random.random() < 0.2:
            conversations.append({"role": "system", "content": random.choice(DEFAULT_SYSTEM_PROMPTS)})
        conversations.append({"role": "user",      "content": text})
        conversations.append({"role": "assistant", "content": item.get("response", "")})

    # ── Encode answer audio (required) ─────────────────────────────────────
    audio_rel = item.get("audio", "")
    audio_path = os.path.join(base_dir, audio_rel) if not os.path.isabs(audio_rel) else audio_rel

    if not os.path.exists(audio_path):
        print(f"\n  [WARN] Audio not found: {audio_path}")
        return None

    codes = wav_to_mimi_codes(audio_path)
    if codes is None:
        return None

    # ── Encode input audio (optional) ──────────────────────────────────────
    question_audio_bytes = None
    if with_input_audio and item.get("input_audio"):
        qa_rel  = item["input_audio"]
        qa_path = os.path.join(base_dir, qa_rel) if not os.path.isabs(qa_rel) else qa_rel
        if os.path.exists(qa_path):
            question_audio_bytes = audio_to_bytes(qa_path)

    return {
        "conversations":   json.dumps(conversations, ensure_ascii=False),
        "answer_audios":   codes,                          # list[int]
        "question_audios": [question_audio_bytes] if question_audio_bytes else [],
        "image_bytes":     [],
        "ref_audios":      [],
        "spk_emb":         [],
    }


# ---------------------------------------------------------------------------
# Write Parquet
# ---------------------------------------------------------------------------

def write_parquet(rows: list[dict], output_path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        print(f"  [WARN] No rows to write to {output_path}")
        return

    table = pa.table({
        "conversations":   pa.array([r["conversations"]   for r in rows], type=pa.large_utf8()),
        "answer_audios":   pa.array([r["answer_audios"]   for r in rows]),
        "question_audios": pa.array([r["question_audios"] for r in rows]),
        "image_bytes":     pa.array([r["image_bytes"]     for r in rows]),
        "ref_audios":      pa.array([r["ref_audios"]      for r in rows]),
        "spk_emb":         pa.array([r["spk_emb"]         for r in rows]),
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    print(f"  Wrote {len(rows)} rows → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare MiniMind-O training data (JSONL → Parquet)")
    p.add_argument("--input",  required=True, help="Input JSONL file")
    p.add_argument("--output", required=True,
                   help="Output path: file.parquet (single) or dir/ (split into train/val)")
    p.add_argument("--val-split",       type=float, default=0.0,
                   help="Fraction of data for validation set (0 = no split, default 0)")
    p.add_argument("--with-input-audio", action="store_true",
                   help="Also encode input_audio fields into question_audios")
    p.add_argument("--max-samples",     type=int, default=0,
                   help="Cap number of samples processed (0 = all)")
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    base_dir = os.path.dirname(os.path.abspath(args.input))

    # Load JSONL
    with open(args.input) as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.max_samples > 0:
        items = items[:args.max_samples]

    print(f"Processing {len(items)} samples from {args.input} ...")

    rows = []
    skipped = 0
    for i, item in enumerate(items):
        print(f"\r  [{i+1}/{len(items)}]  ok={len(rows)}  skip={skipped}", end="", flush=True)
        row = build_sample(item, base_dir, args.with_input_audio)
        if row:
            rows.append(row)
        else:
            skipped += 1

    print(f"\nDone: {len(rows)} ok, {skipped} skipped")

    if not rows:
        print("No valid samples — check audio paths in your JSONL.")
        sys.exit(1)

    # ── Write output ────────────────────────────────────────────────────────
    output = args.output

    if args.val_split > 0:
        # Split train / val
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_split))
        val_rows   = rows[:n_val]
        train_rows = rows[n_val:]
        out_dir = output if output.endswith("/") else output + "/"
        write_parquet(train_rows, os.path.join(out_dir, "train.parquet"))
        write_parquet(val_rows,   os.path.join(out_dir, "val.parquet"))
    else:
        write_parquet(rows, output if output.endswith(".parquet") else output + ".parquet")

    # ── Print summary ───────────────────────────────────────────────────────
    print()
    sample = rows[0]
    conv   = json.loads(sample["conversations"])
    n_code = len(sample["answer_audios"])
    dur_s  = n_code / 8 / 12.5  # 8 codes/frame × 12.5fps
    print("Sample check:")
    print(f"  Turns:          {len(conv)}")
    print(f"  First turn:     {conv[0]['role']}: {conv[0]['content'][:60]}")
    print(f"  Mimi codes:     {n_code}  ({dur_s:.1f}s audio)")
    print(f"  question_audio: {'yes' if sample['question_audios'] else 'no'}")
    print()
    print("Next step — train:")
    print(f"  python scripts/minimind_o_lora_train.py --data_path {output}")


if __name__ == "__main__":
    main()
