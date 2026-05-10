"""
scripts/prepare_bhojpuri_dataset.py

Converts the IISc SYSPIN Bhojpuri Female corpus into JSONL for Parler TTS training.

Supports two modes:
  --mode language   (Plan B) Varied per-clip descriptions — teaches Bhojpuri phoneme patterns.
  --mode speaker    (Plan A) Fixed description for one stable voice identity.

Usage:
    python scripts/prepare_bhojpuri_dataset.py \\
        --corpus /path/to/IISc_SYSPINProject_Bhojpuri_Female_Spk001_HC \\
        --output /path/to/bhojpuri \\
        --mode language \\
        --max-hours 10

Output:
    <output>/train.jsonl
    <output>/val.jsonl        (EVAL domain clips, or 5% random if none)

JSONL schema:
    {"audio": "<abs_path>.wav", "text": "...", "description": "..."}
"""

import argparse
import json
import os
import random
import wave
from pathlib import Path


# ── Description generation ────────────────────────────────────────────────────

# Fixed speaker description for Plan A (speaker adaptation).
# Derived from SYSPIN metadata: female, 35yo, 12yr experience, studio @ 48kHz/40dBSNR.
FIXED_SPEAKER_DESC = (
    "A female speaker delivers Bhojpuri speech at a moderate pace. "
    "The recording is of very high quality, with the speaker's voice "
    "sounding clear and very close up. The speaker has a calm and expressive tone."
)

# Pace buckets for Plan B based on Devanagari chars per second.
# Devanagari has ~3-4 chars/syllable; studio rate ≈ 9-14 chars/sec.
_PACE_LABELS = [
    (9.5,  "slowly"),
    (12.5, "at a moderate pace"),
    (float("inf"), "quickly"),
]

_QUALITY_PHRASES = [
    "The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "The audio quality is excellent with a very clean and clear sound.",
    "The recording is studio quality with very low background noise.",
]


def _pace_label(chars_per_sec: float) -> str:
    for threshold, label in _PACE_LABELS:
        if chars_per_sec < threshold:
            return label
    return "at a moderate pace"


def _make_description(text: str, duration_s: float, clip_idx: int) -> str:
    """Generate a heuristic per-clip description for language adaptation (Plan B)."""
    cps = len(text) / max(duration_s, 0.1)
    pace = _pace_label(cps)
    quality = _QUALITY_PHRASES[clip_idx % len(_QUALITY_PHRASES)]
    return f"A female speaker delivers speech {pace}. {quality}"


# ── Audio duration helper ─────────────────────────────────────────────────────

def _wav_duration(path: str) -> float:
    try:
        with wave.open(path) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        default="/Users/akashsingh/Documents/exps/IISc_SYSPIN_Data/IISc_SYSPINProject_Bhojpuri_Female_Spk001_HC",
        help="Root dir of the SYSPIN Bhojpuri corpus",
    )
    ap.add_argument(
        "--output",
        default="/Users/akashsingh/Documents/exps/bhojpuri",
        help="Output dir for train.jsonl / val.jsonl",
    )
    ap.add_argument(
        "--mode",
        choices=["language", "speaker"],
        default="language",
        help="language = Plan B (varied descriptions), speaker = Plan A (fixed description)",
    )
    ap.add_argument(
        "--min-dur", type=float, default=2.0, help="Min clip duration in seconds"
    )
    ap.add_argument(
        "--max-dur", type=float, default=12.0, help="Max clip duration in seconds"
    )
    ap.add_argument(
        "--max-hours", type=float, default=10.0, help="Max total training hours to include"
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = ap.parse_args()

    random.seed(args.seed)
    corpus = Path(args.corpus)
    wav_dir = corpus / "wav"
    transcript_json = next(corpus.glob("*_Transcripts.json"))

    print(f"Loading transcripts from {transcript_json}")
    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)
    transcripts = data["Transcripts"]

    print(f"Total entries: {len(transcripts)}")

    # ── Scan all clips ────────────────────────────────────────────────────────
    records = []  # (clip_id, text, domain, duration)
    print("Scanning wav durations …")
    for i, (clip_id, entry) in enumerate(transcripts.items()):
        wav_path = wav_dir / (clip_id + ".wav")
        if not wav_path.exists():
            continue
        dur = _wav_duration(str(wav_path))
        if dur < args.min_dur or dur > args.max_dur:
            continue
        records.append((clip_id, entry["Transcript"], entry["Domain"], dur, str(wav_path)))
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i+1}/{len(transcripts)} …")

    print(f"Clips in {args.min_dur}–{args.max_dur}s range: {len(records)}")

    # ── Split: EVAL domain → val, rest → train ────────────────────────────────
    val_records   = [r for r in records if r[2] == "EVALUATION"]
    train_records = [r for r in records if r[2] != "EVALUATION"]

    if not val_records:
        random.shuffle(train_records)
        n_val = max(1, int(0.05 * len(train_records)))
        val_records   = train_records[:n_val]
        train_records = train_records[n_val:]

    # ── Cap training hours ────────────────────────────────────────────────────
    random.shuffle(train_records)
    max_sec = args.max_hours * 3600
    kept, total_sec = [], 0.0
    for r in train_records:
        if total_sec + r[3] > max_sec:
            break
        kept.append(r)
        total_sec += r[3]
    train_records = kept

    print(f"Train clips: {len(train_records)} ({total_sec/3600:.2f}h)")
    print(f"Val   clips: {len(val_records)}")

    # ── Write JSONL ───────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    def _build_sample(clip_id, text, domain, dur, wav_path, idx):
        if args.mode == "speaker":
            desc = FIXED_SPEAKER_DESC
        else:
            desc = _make_description(text, dur, idx)
        return {"audio": wav_path, "text": text, "description": desc}

    train_path = os.path.join(args.output, "train.jsonl")
    val_path   = os.path.join(args.output, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(train_records):
            sample = _build_sample(*r, i)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(val_records):
            sample = _build_sample(*r, i)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSample train entries:")
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            s = json.loads(line)
            print(f"  text:  {s['text'][:60]}...")
            print(f"  desc:  {s['description']}")
            print()


if __name__ == "__main__":
    main()
