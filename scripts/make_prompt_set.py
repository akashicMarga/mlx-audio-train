#!/usr/bin/env python
"""
make_prompt_set.py — build a fixed held-out Hindi prompt set for benchmarks.

Source: SPRINGLab/IndicTTS-Hindi (Hindi monolingual Indic TTS recordings; text
column only — audio is never loaded). The adapters were trained on
IndicVoices-R, a different corpus, so these sentences are unseen by them.

Selection (deterministic for a given --seed):
  - dedupe after whitespace normalisation
  - 30–120 characters (≈4–15 s at ~8 chars/s, under the 240-frame cap)
  - ≥ 95% of letters are Devanagari (no Latin / digit-heavy lines, so Whisper CER
    measures speech, not transliteration)
  - random sample of --n across all shards

Writes JSONL rows: {"text", "lang_code": "hi", "source", "shard", "row", "gender"}.

Usage:
    python scripts/make_prompt_set.py --n 100 --seed 0 --out bench/prompts/hi_indictts_100.jsonl
"""

import argparse
import glob
import json
import os
import random
import re
import unicodedata

DEFAULT_GLOB = ("~/.cache/huggingface/hub/datasets--SPRINGLab--IndicTTS-Hindi/"
                "snapshots/*/data/*.parquet")


def devanagari_ratio(s: str) -> float:
    letters = [c for c in s if unicodedata.category(c)[0] in ("L", "M", "N")]
    if not letters:
        return 0.0
    return sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=120)
    ap.add_argument("--min-devanagari", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.expanduser(args.glob)))
    if not files:
        raise SystemExit(f"no parquet shards match {args.glob}")

    seen, pool = set(), []
    for shard in files:
        t = pq.read_table(shard, columns=["text", "gender"])
        for row, (text, gender) in enumerate(zip(t.column("text").to_pylist(), t.column("gender").to_pylist())):
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            if devanagari_ratio(text) < args.min_devanagari or text in seen:
                continue
            seen.add(text)
            pool.append({"text": text, "lang_code": "hi", "source": "SPRINGLab/IndicTTS-Hindi",
                         "shard": os.path.basename(shard), "row": row, "gender": gender})

    random.Random(args.seed).shuffle(pool)
    picked = pool[: args.n]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = sorted(len(r["text"]) for r in picked)
    print(f"{len(pool)} eligible of {len(files)} shards → wrote {len(picked)} to {args.out}; "
          f"chars min/median/max {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}; "
          f"female {sum(r['gender'] == 0 for r in picked)}, male {sum(r['gender'] == 1 for r in picked)}")


if __name__ == "__main__":
    main()
