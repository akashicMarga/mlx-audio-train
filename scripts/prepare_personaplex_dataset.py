#!/usr/bin/env python3
"""
prepare_personaplex_dataset.py — Convert synthetic-dialog-gen output to PersonaPlex training format.

Source format (synthetic-dialog-gen):
    output/hi/
        manifest.json           # {id, audio_file, text, duration, scenario, domain, num_turns}
        metadata/{id}.json      # {turns: [{speaker, text, duration}]}
        wavs/{id}.wav           # mixed 24kHz mono conversation
        work/{id}/turn_000.wav  # per-turn audio (speaker A or B)

Output format (PersonaPlexDataset):
    dst_dir/
        manifest.json           # paired conversational samples
        wavs/{id}_user.wav
        wavs/{id}_assistant.wav
        tokens/{id}.npz         # {user_audio_tokens, assistant_audio_tokens, text_tokens}
                                #   (after --tokenize; otherwise only manifest + wavs)

Two sample modes:
    --mode turn   (default) — one sample per user->assistant exchange.
                              Uses consecutive A then B turns from work/{id}/turn_*.wav.
    --mode dialog           — one sample per dialog.
                              Uses mixed wavs/{id}.wav.

Usage:
    # Prepare manifest + copy WAVs (no MIMI needed)
    python scripts/prepare_personaplex_dataset.py \\
        --src /path/to/synthetic-dialog-gen/output/hi \\
        --dst ./hindi_data

    # Also tokenize with MIMI
    python scripts/prepare_personaplex_dataset.py \\
        --src /path/to/synthetic-dialog-gen/output/hi \\
        --dst ./hindi_data \\
        --tokenize --mimi-weight /path/to/tokenizer-e351c8d8.safetensors

    # Dialog-level mode
    python scripts/prepare_personaplex_dataset.py \\
        --src /path/to/synthetic-dialog-gen/output/hi \\
        --dst ./hindi_data --mode dialog

    # Start training after preparation
    python scripts/train.py --config configs/personaplex_hindi.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import data/ modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Turn-level preparation
# ---------------------------------------------------------------------------

def prepare_turns(
    src_dir: Path,
    dst_dir: Path,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
) -> list[dict]:
    """Create one sample per user->assistant exchange using per-turn WAV files.

    Returns list of manifest entries.
    """
    manifest_path = src_dir / "manifest.json"
    src_manifest = json.loads(manifest_path.read_text())

    dst_wavs = dst_dir / "wavs"
    dst_wavs.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped_no_work = 0
    skipped_duration = 0

    for entry in src_manifest:
        dialog_id = entry["id"]
        work_dir  = src_dir / "work" / dialog_id
        meta_path = src_dir / "metadata" / f"{dialog_id}.json"

        if not work_dir.exists() or not meta_path.exists():
            skipped_no_work += 1
            continue

        meta  = json.loads(meta_path.read_text())
        turns = meta["turns"]

        for turn_idx in range(len(turns) - 1):
            user_turn = turns[turn_idx]
            asst_turn = turns[turn_idx + 1]

            if user_turn.get("speaker") != "A" or asst_turn.get("speaker") != "B":
                continue

            user_wav = work_dir / f"turn_{turn_idx:03d}.wav"
            asst_wav = work_dir / f"turn_{turn_idx + 1:03d}.wav"
            if not user_wav.exists() or not asst_wav.exists():
                continue

            user_duration = user_turn["duration"]
            asst_duration = asst_turn["duration"]
            if (
                user_duration < min_duration or user_duration > max_duration
                or asst_duration < min_duration or asst_duration > max_duration
            ):
                skipped_duration += 1
                continue

            sample_id = f"{dialog_id}_{turn_idx:02d}_{turn_idx + 1:02d}"
            dst_user_wav = dst_wavs / f"{sample_id}_user.wav"
            dst_asst_wav = dst_wavs / f"{sample_id}_assistant.wav"
            shutil.copy2(user_wav, dst_user_wav)
            shutil.copy2(asst_wav, dst_asst_wav)

            num_frames = int(max(user_duration, asst_duration) * 12.5)

            samples.append({
                "id":                 sample_id,
                "text":               asst_turn["text"],
                "prompt_text":        user_turn["text"],
                "response_text":      asst_turn["text"],
                "user_duration":      round(user_duration, 3),
                "assistant_duration": round(asst_duration, 3),
                "duration":           round(user_duration + asst_duration, 3),
                "num_frames":         num_frames,
                "user_audio_file":    f"wavs/{sample_id}_user.wav",
                "assistant_audio_file": f"wavs/{sample_id}_assistant.wav",
                "speaker":            "paired",
                "dialog_id":          dialog_id,
                "user_turn_idx":      turn_idx,
                "assistant_turn_idx": turn_idx + 1,
                "scenario":           entry.get("scenario", ""),
                "domain":             entry.get("domain", ""),
            })

    print(f"  Exchanges collected: {len(samples)}")
    print(f"  Skipped (no work) : {skipped_no_work} dialogs")
    print(f"  Skipped (duration): {skipped_duration} exchanges")
    return samples


# ---------------------------------------------------------------------------
# Dialog-level preparation
# ---------------------------------------------------------------------------

def prepare_dialogs(src_dir: Path, dst_dir: Path) -> list[dict]:
    """Create one sample per dialog using the mixed WAV.

    Returns list of manifest entries.
    """
    manifest_path = src_dir / "manifest.json"
    src_manifest  = json.loads(manifest_path.read_text())

    dst_wavs = dst_dir / "wavs"
    dst_wavs.mkdir(parents=True, exist_ok=True)

    samples = []

    for entry in src_manifest:
        dialog_id = entry["id"]
        src_wav   = src_dir / entry["audio_file"]
        meta_path = src_dir / "metadata" / f"{dialog_id}.json"

        if not src_wav.exists():
            print(f"  Warning: missing {src_wav}, skipping")
            continue

        dst_wav = dst_wavs / f"{dialog_id}.wav"
        shutil.copy2(src_wav, dst_wav)

        if meta_path.exists():
            meta      = json.loads(meta_path.read_text())
            user_text = " ".join(t["text"] for t in meta["turns"] if t["speaker"] == "A")
            asst_text = " ".join(t["text"] for t in meta["turns"] if t["speaker"] == "B")
        else:
            user_text = entry["text"]
            asst_text = entry["text"]

        num_frames = int(entry["duration"] * 12.5)

        samples.append({
            "id":             dialog_id,
            "text":           entry["text"],
            "user_text":      user_text,
            "assistant_text": asst_text,
            "duration":       round(entry["duration"], 3),
            "num_frames":     num_frames,
            "audio_file":     f"wavs/{dialog_id}.wav",
            "speaker":        "mixed",
            "scenario":       entry.get("scenario", ""),
            "domain":         entry.get("domain", ""),
            "num_turns":      entry.get("num_turns", 0),
        })

    print(f"  Dialogs collected: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# MIMI tokenization
# ---------------------------------------------------------------------------

def tokenize_samples(
    samples:       list[dict],
    dst_dir:       Path,
    mimi_weight:   str,
    text_tokenizer: str | None = None,
    num_codebooks: int = 8,
    sample_rate:   int = 24000,
) -> None:
    """Tokenize WAV files with MIMI and save tokens/{id}.npz.

    For 'user' speaker turns   → stored as user_audio_tokens  (rows 9-16 in model)
    For 'assistant' speaker    → stored as assistant_audio_tokens (rows 1-8)
    For 'mixed' dialog samples → stored as audio_tokens (used for both streams)
    """
    try:
        import rustymimi
        import sentencepiece
        import sphn
    except ImportError:
        raise ImportError(
            "rustymimi, sentencepiece and sphn are required for tokenization.\n"
            "Install with: pip install rustymimi sentencepiece sphn"
        )

    tokens_dir = dst_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    sp = sentencepiece.SentencePieceProcessor(model_file=text_tokenizer) if text_tokenizer else None

    chunk_size = 1920  # MIMI chunk size in samples (24000 / 12.5 = 1920)

    done   = 0
    errors = 0

    for sample in samples:
        out_path = tokens_dir / f"{sample['id']}.npz"
        if out_path.exists():
            done += 1
            continue

        def _encode_audio(wav_key: str) -> np.ndarray:
            wav_path = dst_dir / sample[wav_key]
            pcm, _ = sphn.read(str(wav_path), sample_rate=sample_rate)
            if pcm.ndim == 2 and pcm.shape[0] > 1:
                pcm = pcm[0:1]
            elif pcm.ndim == 1:
                pcm = pcm[np.newaxis, :]

            tokenizer = rustymimi.Tokenizer(mimi_weight, num_codebooks=num_codebooks)
            total = pcm.shape[-1]
            steps = (total + chunk_size - 1) // chunk_size
            all_codes = []
            for idx in range(steps):
                start = idx * chunk_size
                end = min((idx + 1) * chunk_size, total)
                chunk = pcm[:, start:end]
                if chunk.shape[-1] < chunk_size:
                    chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[-1])), mode="constant")
                codes = tokenizer.encode_step(chunk[None, :])
                all_codes.append(codes)
            return np.concatenate(all_codes, axis=-1)[0]

        try:
            if sample.get("speaker") == "paired":
                user_audio_tokens = _encode_audio("user_audio_file")
                assistant_audio_tokens = _encode_audio("assistant_audio_file")
                response_text = sample.get("response_text") or sample.get("text", "")
                prompt_text = sample.get("prompt_text", "")
                text_tokens = np.array(sp.encode(response_text), dtype=np.int32) if sp else np.array([], dtype=np.int32)
                prompt_text_tokens = np.array(sp.encode(prompt_text), dtype=np.int32) if sp and prompt_text else np.array([], dtype=np.int32)
                np.savez(
                    out_path,
                    user_audio_tokens=user_audio_tokens,
                    assistant_audio_tokens=assistant_audio_tokens,
                    text_tokens=text_tokens,
                    prompt_text_tokens=prompt_text_tokens,
                )
            else:
                wav_path = dst_dir / sample["audio_file"]
                try:
                    pcm, _ = sphn.read(str(wav_path), sample_rate=sample_rate)
                except Exception as e:
                    print(f"  Error reading {wav_path}: {e}")
                    errors += 1
                    continue

                if pcm.ndim == 2 and pcm.shape[0] > 1:
                    pcm = pcm[0:1]
                elif pcm.ndim == 1:
                    pcm = pcm[np.newaxis, :]

                tokenizer = rustymimi.Tokenizer(mimi_weight, num_codebooks=num_codebooks)
                total = pcm.shape[-1]
                steps = (total + chunk_size - 1) // chunk_size
                all_codes = []
                for idx in range(steps):
                    start = idx * chunk_size
                    end = min((idx + 1) * chunk_size, total)
                    chunk = pcm[:, start:end]
                    if chunk.shape[-1] < chunk_size:
                        chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[-1])), mode="constant")
                    codes = tokenizer.encode_step(chunk[None, :])
                    all_codes.append(codes)

                audio_tokens = np.concatenate(all_codes, axis=-1)[0]
                response_text = sample.get("response_text") or sample.get("assistant_text") or sample.get("text", "")
                prompt_text = sample.get("prompt_text") or sample.get("user_text", "")
                text_tokens = np.array(sp.encode(response_text), dtype=np.int32) if sp else np.array([], dtype=np.int32)
                prompt_text_tokens = np.array(sp.encode(prompt_text), dtype=np.int32) if sp and prompt_text else np.array([], dtype=np.int32)

                speaker = sample.get("speaker", "mixed")
                if speaker == "user":
                    np.savez(out_path, user_audio_tokens=audio_tokens,
                             text_tokens=text_tokens, prompt_text_tokens=prompt_text_tokens)
                elif speaker == "assistant":
                    np.savez(out_path, assistant_audio_tokens=audio_tokens,
                             text_tokens=text_tokens, prompt_text_tokens=prompt_text_tokens)
                else:
                    np.savez(out_path, audio_tokens=audio_tokens,
                             text_tokens=text_tokens, prompt_text_tokens=prompt_text_tokens)
        except Exception as e:
            print(f"  Error tokenizing {sample['id']}: {e}")
            errors += 1
            continue

        done += 1
        if done % 50 == 0:
            print(f"  Tokenized {done}/{len(samples)} ...")

    print(f"  Tokenization complete: {done} done, {errors} errors")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(samples: list[dict], mode: str) -> None:
    durations = [s["duration"] for s in samples]
    total_h   = sum(durations) / 3600
    speakers: dict[str, int] = {}
    for s in samples:
        sp = s.get("speaker", "mixed")
        speakers[sp] = speakers.get(sp, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Mode     : {mode}")
    print(f"  Samples  : {len(samples)}")
    print(f"  Total    : {total_h:.2f}h audio")
    print(f"  Duration : min={min(durations):.1f}s, "
          f"max={max(durations):.1f}s, "
          f"mean={sum(durations)/len(durations):.1f}s")
    print(f"  Speakers : {speakers}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic dialog data for PersonaPlex training"
    )
    parser.add_argument("--src", type=str, required=True,
                        help="Source directory (synthetic-dialog-gen output/hi)")
    parser.add_argument("--dst", type=str, default="./hindi_data",
                        help="Destination directory for PersonaPlex training data")
    parser.add_argument("--mode", choices=["turn", "dialog"], default="turn",
                        help="turn: one sample per turn (default); dialog: one sample per dialog")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="Minimum turn duration in seconds (turn mode only)")
    parser.add_argument("--max-duration", type=float, default=20.0,
                        help="Maximum turn duration in seconds (turn mode only)")
    parser.add_argument("--tokenize", action="store_true",
                        help="Run MIMI tokenization (requires --mimi-weight)")
    parser.add_argument("--mimi-weight", type=str, default=None,
                        help="Path to MIMI tokenizer weights (.safetensors)")
    parser.add_argument("--text-tokenizer", type=str, default=None,
                        help="Path to SentencePiece tokenizer model for response text")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--sample-rate",   type=int, default=24000)
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    print(f"Source : {src_dir}")
    print(f"Dest   : {dst_dir}")
    print(f"Mode   : {args.mode}")

    # Collect samples
    if args.mode == "turn":
        samples = prepare_turns(src_dir, dst_dir, args.min_duration, args.max_duration)
    else:
        samples = prepare_dialogs(src_dir, dst_dir)

    if not samples:
        print("No samples collected. Check source directory.")
        return

    print_stats(samples, args.mode)

    # Write manifest
    manifest_path = dst_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Manifest written: {manifest_path} ({len(samples)} samples)")

    # Tokenize
    if args.tokenize:
        if not args.mimi_weight:
            print("\nWarning: --tokenize requires --mimi-weight. Skipping tokenization.")
            print("To tokenize later, run with --tokenize --mimi-weight <path>")
        else:
            print(f"\nTokenizing with MIMI ({args.mimi_weight}) ...")
            tokenize_samples(
                samples, dst_dir, args.mimi_weight,
                text_tokenizer=args.text_tokenizer,
                num_codebooks=args.num_codebooks,
                sample_rate=args.sample_rate,
            )
    else:
        print("\nSkipping tokenization (no --tokenize flag).")
        print("To tokenize later:")
        print(f"  python scripts/prepare_personaplex_dataset.py \\")
        print(f"    --src {args.src} --dst {args.dst} --mode {args.mode} \\")
        print(f"    --tokenize --mimi-weight /path/to/mimi.safetensors")

    print(f"\nDone. Training data at: {dst_dir}")
    print(f"Start training with:")
    print(f"  python scripts/train.py --config configs/personaplex_hindi.yaml")


if __name__ == "__main__":
    main()
