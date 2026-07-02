#!/usr/bin/env python
"""
ttsds2_eval.py — generate held-out Hindi audio from each adapter and dump it to
per-adapter wav directories for TTSDS2 distributional scoring.

This is the *generation* half (runs in the mlx-audio env). Scoring is a separate
step (scripts/ttsds2_score.py) that runs in an isolated `ttsds` env, because
TTSDS2 pins numpy<2 / Python 3.10–3.12 and pulls a heavy torch stack that
conflicts with mlx. The two halves talk only through directories of wavs — which
is all TTSDS2's distributional score needs (content need not match the
reference).

It reuses grpo_heldout_eval.py wholesale for adapter discovery, model build, the
held-out prompt set, and the interleaved (deployment-path) rollout — so the audio
scored here is generated exactly like the CER held-out eval, just written to disk
instead of CER-scored.

Layout written under --out-root:
    <out-root>/<adapter-label>/*.wav      # one wav per (sentence, seed)
    <out-root>/reference_real/*.wav       # optional real-speech subset (--ref-audio-dir)

Then score (isolated env):
    uv run --python 3.12 --with "numpy<2" --with ttsds --with onnxruntime \
        python scripts/ttsds2_score.py \
            --reference <out-root>/reference_real \
            --synth SFT-baseline=<out-root>/SFT-baseline \
            --synth <cell>=<out-root>/<cell> \
            --multilingual --out <out-root>/ttsds2_results.csv

Usage (generation):
    python scripts/ttsds2_eval.py \
        --config ~/grpo_ablation/base_hindi_grpo.local.yaml \
        --ablation-root ~/grpo_ablation/run1 \
        --heldout-jsonl ~/Documents/exps/hindi/val_codes.abs.jsonl \
        --out-root ~/ttsds2_out \
        --num-sentences 100 --seeds 2 \
        --ref-audio-dir ~/Documents/exps/hindi/audio --ref-n 200
"""

import argparse
import importlib.util
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_heldout_eval_module():
    """Import scripts/grpo_heldout_eval.py by path to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(
        "grpo_heldout_eval", REPO_ROOT / "scripts" / "grpo_heldout_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_reference_dir(src_dir: str, dst_dir: str, n: int, seed: int = 0):
    """Copy a random n-wav subset of real speech into dst_dir (the TTSDS2
    reference). A subset (not all 4000) keeps scoring quick while staying a fair
    distributional sample."""
    src = Path(os.path.expanduser(src_dir))
    wavs = sorted(src.glob("*.wav"))
    if not wavs:
        sys.exit(f"[ttsds2] no wavs in --ref-audio-dir {src}")
    random.Random(seed).shuffle(wavs)
    wavs = wavs[:n]
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for w in wavs:
        shutil.copy2(w, dst / w.name)
    print(f"[ttsds2] reference: copied {len(wavs)} real wavs -> {dst}")


def generate_adapter_wavs(model, train_mod, prompts, out_dir: Path, *, lang_code,
                          seeds, max_new_tokens, temperature, top_p, top_k, sample_rate):
    """Generate audio for every (sentence, seed) via the interleaved deployment
    path (same as grpo_heldout_eval) and write one wav each into out_dir."""
    import mlx.core as mx
    from train.grpo.rollout import sample_rollouts_interleaved, decode_codes_to_audio

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for si in range(seeds):
        for pi, prompt in enumerate(prompts):
            mx.random.seed(1000 * si + pi)
            out = sample_rollouts_interleaved(
                model, prompt["text"], lang_code=prompt.get("lang_code", lang_code),
                group_size=1, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, compute_ref=False, ref_params=None,
                ref_audio=prompt.get("ref_audio_wav"),   # Pipeline 2: clone the voice
            )
            audios = decode_codes_to_audio(model, out["full_codes"], out["codec_mask"])
            wav = np.asarray(audios[0], dtype=np.float32)
            sf.write(str(out_dir / f"s{si:02d}_p{pi:04d}.wav"), wav, sample_rate)
            n_written += 1
    print(f"[ttsds2] wrote {n_written} wavs -> {out_dir}")
    return n_written


def main():
    ap = argparse.ArgumentParser(description="Generate held-out audio for TTSDS2 scoring")
    ap.add_argument("--config", required=True, help="Model/GRPO YAML (base model + init_adapters)")
    ap.add_argument("--ablation-root", required=True, help="Sweep out-root (L-*/checkpoint-final)")
    ap.add_argument("--heldout-jsonl", required=True, help="Held-out sentences (text used; not trained on)")
    ap.add_argument("--out-root", required=True, help="Where per-adapter wav dirs are written")
    ap.add_argument("--num-sentences", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ref-audio-dir", default=None,
                    help="Dir of REAL speech wavs; a random subset is copied to <out-root>/reference_real")
    ap.add_argument("--ref-n", type=int, default=200, help="How many real wavs to sample for the reference")
    ap.add_argument("--dry-run", action="store_true", help="List adapters + held-out count, load nothing")
    ap.add_argument("--only", default=None,
                    help="Substring filter: generate only adapters whose label contains this "
                         "(e.g. 'pg-token__sft-0' for one cell; 'SFT' also matches the baseline)")
    args = ap.parse_args()

    he = _load_heldout_eval_module()
    train_mod = he._load_train_module()
    cfg = train_mod.load_config(args.config)
    ablation_root = Path(os.path.expanduser(args.ablation_root)).resolve()
    out_root = Path(os.path.expanduser(args.out_root)).resolve()
    adapters = he.discover_adapters(cfg, ablation_root)
    if args.only:
        adapters = [(l, p) for l, p in adapters if args.only in l]

    print(f"Adapters to generate ({len(adapters)}):")
    for label, path in adapters:
        print(f"  {label:<38} {path}")
    if not adapters:
        print("No adapters found — is the sweep done?"); sys.exit(1)

    if args.dry_run:
        n = sum(1 for _ in open(args.heldout_jsonl)) if os.path.exists(args.heldout_jsonl) else 0
        print(f"[dry-run] held-out file has {n} lines; would generate "
              f"{min(n, args.num_sentences)} × {args.seeds} seeds per adapter.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    if args.ref_audio_dir:
        sample_reference_dir(args.ref_audio_dir, str(out_root / "reference_real"),
                             args.ref_n)

    # ── Build the model once (same path as grpo_heldout_eval main) ──────────
    from train.lora import load_adapters

    model = train_mod.load_model(cfg)
    train_mod.apply_lora(model, cfg)
    for attr in ("speech_tokenizer", "speaker_encoder"):
        sub = getattr(model, attr, None)
        if sub is not None:
            sub.freeze()
    if getattr(model, "speech_tokenizer", None) is None:
        print("[ttsds2] ERROR: speech_tokenizer required to decode audio."); sys.exit(1)

    prompts = he.load_heldout(train_mod, cfg, model, args.heldout_jsonl, args.num_sentences)
    is_p2 = bool(prompts) and prompts[0].get("ref_mel") is not None
    print(f"[ttsds2] {len(prompts)} held-out sentences × {args.seeds} seeds"
          f"{'  (Pipeline 2: cloning)' if is_p2 else ''}\n")

    g = cfg.get("grpo", {})
    t = cfg.get("trainer", {})
    common = dict(
        lang_code=t.get("lang_code", "auto"), seeds=args.seeds,
        max_new_tokens=g.get("max_new_tokens", 240), temperature=g.get("temperature", 0.9),
        top_p=g.get("top_p", 0.95), top_k=g.get("top_k", 50),
        sample_rate=cfg.get("data", {}).get("target_sr", 24000),
    )

    for label, path in adapters:
        print(f"=== {label} ===")
        load_adapters(model, path)
        generate_adapter_wavs(model, train_mod, prompts, out_root / label, **common)
        # Free MLX's buffer cache between adapters — it grows unbounded across the
        # ~n_sentences×seeds generations and OOM-kills later adapters.
        import gc
        import mlx.core as mx
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass

    print(f"\n[ttsds2] done. Wav dirs under {out_root}")
    print("Next: score in the isolated ttsds env, e.g.")
    synth_flags = " ".join(f"--synth {label}={out_root / label}" for label, _ in adapters)
    ref_flag = f"--reference {out_root / 'reference_real'}" if args.ref_audio_dir else "--reference <REAL_WAV_DIR>"
    print(f"  uv run --python 3.12 --with 'numpy<2' --with ttsds --with onnxruntime \\\n"
          f"    python scripts/ttsds2_score.py {ref_flag} \\\n"
          f"    {synth_flags} --multilingual --out {out_root / 'ttsds2_results.csv'}")


if __name__ == "__main__":
    main()
