#!/usr/bin/env python3
"""
PyTorch inference script for ai4bharat/indic-parler-tts.
Uses the official parler-tts package directly.

Usage:
    python scripts/infer_parler_pt.py --text "नमस्ते, आप कैसे हैं?" --output out.wav

    # With a specific speaker:
    python scripts/infer_parler_pt.py \
        --text "Hello, how are you doing today?" \
        --description "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise." \
        --output out.wav
"""

import argparse
import time
from pathlib import Path

import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run ai4bharat/indic-parler-tts via PyTorch.")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--description",
        default=(
            "A female speaker delivers a slightly expressive and animated speech "
            "with a moderate speed and pitch. The recording is of very high quality, "
            "with the speaker's voice sounding clear and very close up."
        ),
        help="Style/speaker description prompt",
    )
    parser.add_argument("--output", default="output_pt.wav", help="Output WAV path")
    parser.add_argument("--repo-id", default="ai4bharat/indic-parler-tts", help="HF repo ID")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer-pt] device: {device}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[infer-pt] Loading model from {args.repo_id} ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.repo_id).to(device)
    model.eval()

    prompt_tok = AutoTokenizer.from_pretrained(args.repo_id)
    desc_tok   = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    print(f"[infer-pt] Loaded in {time.time() - t0:.1f}s")

    desc_enc   = desc_tok(args.description, return_tensors="pt").to(device)
    prompt_enc = prompt_tok(args.text, return_tensors="pt").to(device)

    print(f"[infer-pt] Synthesizing: {args.text[:80]}")
    t1 = time.time()
    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_enc.input_ids,
            attention_mask=desc_enc.attention_mask,
            prompt_input_ids=prompt_enc.input_ids,
            prompt_attention_mask=prompt_enc.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze()
    sr    = model.config.sampling_rate
    sf.write(str(out_path), audio, sr)

    audio_s = len(audio) / sr
    print(f"[infer-pt] Generated {audio_s:.2f}s in {time.time() - t1:.1f}s  (sr={sr})")
    print(f"[infer-pt] Saved: {out_path}")


if __name__ == "__main__":
    main()
