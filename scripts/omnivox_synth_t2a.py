"""
OmniVox Synthetic T2A Data Pipeline
====================================
Hindi text  →  Qwen3-TTS audio  →  Mimi codes  →  T2A parquet

Sources used (auto-combined):
  1. Existing JSONL  (e.g. hindi/train.jsonl)       — real-data transcripts
  2. FLEURS Hindi parquets                           — ~2120 + 239 texts
  3. Any extra JSONL passed via --extra-jsonl

Output is the same parquet schema as omnivox_prepare_t2a.py so it can be
fed directly to omnivox_t2a_train.py.

Usage:
    # Quick test with 50 samples
    python scripts/omnivox_synth_t2a.py \\
        --output /Users/akashsingh/Documents/exps/omnivox_t2a_synth/test.parquet \\
        --max 50

    # Full run — all sources, ~4k clips
    python scripts/omnivox_synth_t2a.py \\
        --output /Users/akashsingh/Documents/exps/omnivox_t2a_synth/train.parquet

    # Val split (FLEURS val only, keep OmniVox val.parquet separate)
    python scripts/omnivox_synth_t2a.py \\
        --output /Users/akashsingh/Documents/exps/omnivox_t2a_synth/val.parquet \\
        --sources fleurs_val --max 200
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────

FLEURS_TRAIN_PARQUET = "/Users/akashsingh/Documents/exps/fleurs_hindi/parquet-data/hi_in/train-00000-of-00001.parquet"
FLEURS_VAL_PARQUET   = "/Users/akashsingh/Documents/exps/fleurs_hindi/parquet-data/hi_in/validation-00000-of-00001.parquet"
HINDI_TRAIN_JSONL    = "/Users/akashsingh/Documents/exps/hindi/train.jsonl"
HINDI_VAL_JSONL      = "/Users/akashsingh/Documents/exps/hindi/val.jsonl"

QWEN3_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

MIN_CHARS = 20    # skip very short sentences
MAX_CHARS = 300   # skip very long sentences (TTS quality degrades)

SYSTEM_PROMPTS = [
    "आप एक सहायक हिंदी वॉयस असिस्टेंट हैं।",
    "You are a helpful Hindi voice assistant.",
    "आप एक ज्ञानवान AI हैं जो हिंदी में बोलते हैं।",
]


# ─────────────────────────────────────────────────────────
# Text collection
# ─────────────────────────────────────────────────────────

def load_texts_jsonl(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", "").strip()
            if MIN_CHARS <= len(t) <= MAX_CHARS:
                texts.append(t)
    return texts


def load_texts_fleurs(path: str) -> list[str]:
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["raw_transcription"])
    texts = []
    for i in range(len(table)):
        t = table.column("raw_transcription")[i].as_py()
        if t and MIN_CHARS <= len(t) <= MAX_CHARS:
            texts.append(t.strip())
    return texts


def collect_texts(sources: list[str], extra_jsonls: list[str]) -> list[str]:
    all_texts: list[str] = []

    if "hindi_train" in sources and os.path.exists(HINDI_TRAIN_JSONL):
        t = load_texts_jsonl(HINDI_TRAIN_JSONL)
        print(f"  hindi_train JSONL:  {len(t)} texts")
        all_texts.extend(t)

    if "hindi_val" in sources and os.path.exists(HINDI_VAL_JSONL):
        t = load_texts_jsonl(HINDI_VAL_JSONL)
        print(f"  hindi_val JSONL:    {len(t)} texts")
        all_texts.extend(t)

    if "fleurs_train" in sources and os.path.exists(FLEURS_TRAIN_PARQUET):
        t = load_texts_fleurs(FLEURS_TRAIN_PARQUET)
        print(f"  fleurs_train:       {len(t)} texts")
        all_texts.extend(t)

    if "fleurs_val" in sources and os.path.exists(FLEURS_VAL_PARQUET):
        t = load_texts_fleurs(FLEURS_VAL_PARQUET)
        print(f"  fleurs_val:         {len(t)} texts")
        all_texts.extend(t)

    for extra in extra_jsonls:
        if os.path.exists(extra):
            t = load_texts_jsonl(extra)
            print(f"  extra {extra}: {len(t)} texts")
            all_texts.extend(t)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in all_texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"Total unique texts after dedup: {len(unique)}")
    return unique


# ─────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────

_tts_model = None

def get_tts_model(model_path: str):
    global _tts_model
    if _tts_model is None:
        from mlx_audio.tts.generate import load_model
        from huggingface_hub import snapshot_download
        from pathlib import Path
        # Resolve HF model ID to local cache path if needed
        local_path = model_path
        if not os.path.isdir(model_path):
            print(f"Resolving {model_path} from HF cache ...", flush=True)
            local_path = snapshot_download(model_path, local_files_only=True)
        print(f"Loading Qwen3-TTS: {local_path} ...", flush=True)
        t0 = time.time()
        _tts_model = load_model(Path(local_path))
        print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    return _tts_model


def synthesize_text(text: str, model, out_dir: str, prefix: str,
                    temperature: float = 0.7) -> "str | None":
    """
    Synthesize Hindi text to WAV using Qwen3-TTS.

    generate_audio saves to  {out_dir}/{prefix}_000.wav  (or _001, _002 for
    multi-segment text).  We request join_audio=True so there is always a
    single file: {out_dir}/{prefix}_joined.wav (or {prefix}_000.wav when only
    one segment is produced).

    Returns the WAV path on success, None on failure.
    """
    from mlx_audio.tts.generate import generate_audio
    try:
        generate_audio(
            text,
            model=model,
            lang_code="auto",
            output_path=out_dir,
            file_prefix=prefix,
            audio_format="wav",
            join_audio=True,
            temperature=temperature,
            verbose=False,
            play=False,
        )
    except Exception as e:
        print(f"\n  [WARN] TTS failed for '{text[:40]}': {e}", flush=True)
        return None

    # generate_audio writes {prefix}_joined.wav if join_audio=True,
    # otherwise {prefix}_000.wav for a single segment.
    for candidate in (f"{prefix}_joined.wav", f"{prefix}_000.wav"):
        p = os.path.join(out_dir, candidate)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    # Fallback: find any file starting with prefix
    for fname in os.listdir(out_dir):
        if fname.startswith(prefix) and fname.endswith(".wav"):
            p = os.path.join(out_dir, fname)
            if os.path.getsize(p) > 0:
                return p
    return None


# ─────────────────────────────────────────────────────────
# Mimi encoder
# ─────────────────────────────────────────────────────────

_mimi = None

def get_mimi():
    global _mimi
    if _mimi is None:
        print("Loading Mimi codec ...", end=" ", flush=True)
        from mlx_audio.codec import Mimi
        _mimi = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")
        print(f"OK ({_mimi.sample_rate}Hz, {_mimi.frame_rate}fps)", flush=True)
    return _mimi


def encode_wav_to_mimi(wav_path: str) -> "np.ndarray | None":
    """Encode WAV to (8, T) Mimi codes. Returns None on failure."""
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

    codes_np = np.array(codes[0, :8, :].tolist(), dtype=np.int32)
    return codes_np


def codes_to_flat_bytes(codes: np.ndarray) -> bytes:
    """(8, T) int32  →  flat interleaved bytes (same layout as prepare_t2a)."""
    return codes.T.flatten().astype(np.int32).tobytes()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Synthesise Hindi T2A parquet via Qwen3-TTS")
    p.add_argument("--output",   required=True, help="Output parquet path")
    p.add_argument("--sources",  default="hindi_train,fleurs_train,fleurs_val",
                   help="Comma-separated: hindi_train hindi_val fleurs_train fleurs_val")
    p.add_argument("--extra-jsonl", nargs="*", default=[],
                   help="Extra JSONL files with {text} fields")
    p.add_argument("--max",    type=int, default=0, help="Max samples (0 = all)")
    p.add_argument("--tts-model", default=QWEN3_TTS_MODEL)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--tmpdir", default=None, help="Temp dir for WAV files (default: system)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sources = [s.strip() for s in args.sources.split(",")]
    print(f"Sources: {sources}")
    texts = collect_texts(sources, args.extra_jsonl or [])

    if args.max > 0:
        texts = texts[:args.max]
    random.shuffle(texts)
    print(f"\nSynthesising {len(texts)} texts ...\n")

    tts_model = get_tts_model(args.tts_model)
    # Mimi loaded lazily on first encode

    records = []
    n_tts_fail  = 0
    n_mimi_fail = 0
    t_start     = time.time()

    with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
        for i, text in enumerate(texts):
            prefix  = f"synth_{i:05d}"

            # TTS synthesis
            wav_path = synthesize_text(text, tts_model, tmpdir, prefix,
                                       temperature=args.temperature)
            if wav_path is None:
                n_tts_fail += 1
                continue

            # Mimi encode
            codes = encode_wav_to_mimi(wav_path)
            os.remove(wav_path)   # free disk immediately

            if codes is None or codes.shape[1] < 8:
                n_mimi_fail += 1
                continue

            # Duration guard: 0.5s–20s  (6–250 Mimi frames @ 12.5fps)
            if not (6 <= codes.shape[1] <= 250):
                n_mimi_fail += 1
                continue

            system = random.choice(SYSTEM_PROMPTS)
            convs = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": text},
                {"role": "assistant", "content": text},
            ]
            records.append({
                "conversations": json.dumps(convs, ensure_ascii=False),
                "answer_audios": codes_to_flat_bytes(codes),
                "n_frames":      codes.shape[1],
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate    = (i + 1) / elapsed
                eta     = (len(texts) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(texts)}]  ok={len(records)}  "
                      f"tts_fail={n_tts_fail}  mimi_fail={n_mimi_fail}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"\nFinished: {len(records)} records  "
          f"(tts_fail={n_tts_fail} mimi_fail={n_mimi_fail})", flush=True)

    if not records:
        print("ERROR: no records produced — check TTS model path and text sources")
        sys.exit(1)

    # Stats
    frames = [r["n_frames"] for r in records]
    durs   = [f / 12.5 for f in frames]
    total_h = sum(durs) / 3600
    print(f"Duration: min={min(durs):.1f}s  max={max(durs):.1f}s  "
          f"mean={np.mean(durs):.1f}s  total={total_h:.2f}h", flush=True)

    # Write parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table = pa.table({
        "conversations": pa.array([r["conversations"] for r in records], type=pa.large_utf8()),
        "answer_audios": pa.array([r["answer_audios"] for r in records], type=pa.large_binary()),
    })
    pq.write_table(table, args.output, compression="snappy", row_group_size=100)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {len(records)} rows → {args.output}  ({size_mb:.0f} MB)", flush=True)

    # Sanity check
    pf  = pq.ParquetFile(args.output)
    row = pf.read(columns=["conversations", "answer_audios"]).to_pydict()
    c0  = json.loads(row["conversations"][0])
    a0  = np.frombuffer(row["answer_audios"][0], dtype=np.int32)
    print(f"Sanity [0]: text='{c0[1]['content'][:60]}'  "
          f"frames={len(a0)//8}  range=[{a0.min()},{a0.max()}]", flush=True)


if __name__ == "__main__":
    main()
