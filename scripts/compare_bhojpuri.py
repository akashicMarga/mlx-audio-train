"""
scripts/compare_bhojpuri.py

Side-by-side inference: base model vs LoRA-adapted model.
Saves paired WAV files so you can hear the difference.

Usage:
    python scripts/compare_bhojpuri.py                          # uses defaults
    python scripts/compare_bhojpuri.py --adapter <ckpt_dir>    # custom checkpoint
    python scripts/compare_bhojpuri.py --out-dir /tmp/compare  # custom output dir
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import mlx.core as mx

# ── Test sentences ─────────────────────────────────────────────────────────────

# Two descriptions: Divya (named voice) and the generic one used during training.
DIVYA   = "Divya's voice is slightly expressive and very animated. She speaks at a moderate pace."
GENERIC = "A female speaker delivers speech at a moderate pace. The recording is of very high quality."

TEST_SENTENCES = [
    # ── Divya speaking Bhojpuri ───────────────────────────────────────────────
    {
        "tag": "divya_bhojpuri_short",
        "text": "ई बहुत नीमन बा। हम कल जाइब।",
        "description": DIVYA,
    },
    {
        "tag": "divya_bhojpuri_long",
        "text": "रउरा के राम राम। आज हम एही गाँव में रहीला। का हाल बा रउरा के?",
        "description": DIVYA,
    },
    # ── Generic (training) description speaking Bhojpuri ─────────────────────
    {
        "tag": "generic_bhojpuri_short",
        "text": "ई बहुत नीमन बा। हम कल जाइब।",
        "description": GENERIC,
    },
    {
        "tag": "generic_bhojpuri_long",
        "text": "रउरा के राम राम। आज हम एही गाँव में रहीला। का हाल बा रउरा के?",
        "description": GENERIC,
    },
    # ── Sanity: Hindi should sound the same on both models ───────────────────
    {
        "tag": "divya_hindi_sanity",
        "text": "नमस्ते, आप कैसे हैं? मैं बहुत अच्छा हूँ।",
        "description": DIVYA,
    },
    {
        "tag": "generic_hindi_sanity",
        "text": "नमस्ते, आप कैसे हैं? मैं बहुत अच्छा हूँ।",
        "description": GENERIC,
    },
]

HF_REPO       = "ai4bharat/indic-parler-tts"
DEFAULT_CKPT  = "/Users/akashsingh/Documents/exps/checkpoints/parler-bhojpuri-language/checkpoint-best"
DEFAULT_OUT   = "/Users/akashsingh/Documents/exps/bhojpuri_compare"
SR            = 44100


def generate_one(model, tokenizers, item: dict, label: str) -> np.ndarray:
    from models.indic_parler_tts.generate import generate
    print(f"  [{label}] {item['tag']} -- \"{item['text'][:40]}...\"")
    audio = generate(
        model,
        tokenizers,
        description=item["description"],
        text=item["text"],
        max_audio_length_s=10.0,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    )
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_CKPT,
                    help="Path to checkpoint dir containing adapters.safetensors")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="Directory to write WAV files into")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--sentences", nargs="*",
                    help="Subset of tags to run (default: all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = TEST_SENTENCES
    if args.sentences:
        sentences = [s for s in sentences if s["tag"] in args.sentences]

    adapter_path = Path(args.adapter) / "adapters.safetensors"
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        sys.exit(1)

    # ── Load base model ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Loading base model …")
    print(f"{'='*60}")
    from models.indic_parler_tts.generate import load_model
    model, tokenizers = load_model(args.hf_repo)

    # ── Base inference ─────────────────────────────────────────────────────────
    print(f"\n── Base model (no adapter) ──")
    base_audios = {}
    for item in sentences:
        audio = generate_one(model, tokenizers, item, "base")
        path = out_dir / f"{item['tag']}_base.wav"
        sf.write(str(path), audio, SR)
        print(f"    → {path}")
        base_audios[item["tag"]] = audio

    # ── Apply LoRA + load adapter ──────────────────────────────────────────────
    print(f"\n── Applying LoRA and loading adapter from:\n   {adapter_path} ──")
    from train.lora import apply_lora, load_adapters, LoRAConfig
    import yaml
    cfg_path = Path(args.adapter).parent / "model_config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        lora_raw = cfg.get("lora", {})
    else:
        print("  Warning: model_config.yaml not found, using config defaults")
        lora_raw = {}

    lora_config = LoRAConfig(
        rank           = lora_raw.get("rank",    8),
        alpha          = lora_raw.get("alpha",   16.0),
        dropout        = 0.0,   # no dropout at inference
        target_modules = lora_raw.get("target_modules", None),
        model_type     = "indic_parler_tts",
    )
    n = apply_lora(model, lora_config)
    print(f"  Patched {n} LoRA layers")

    load_adapters(model, str(adapter_path))
    mx.eval(model.parameters())

    # ── Adapted inference ──────────────────────────────────────────────────────
    print(f"\n── Adapted model ──")
    for item in sentences:
        audio = generate_one(model, tokenizers, item, "adapted")
        path = out_dir / f"{item['tag']}_adapted.wav"
        sf.write(str(path), audio, SR)
        print(f"    → {path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Output files in: {out_dir}")
    print(f"{'='*60}")
    for item in sentences:
        tag = item["tag"]
        print(f"  {tag}_base.wav    ←→   {tag}_adapted.wav")
    print()


if __name__ == "__main__":
    main()
