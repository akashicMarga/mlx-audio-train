#!/usr/bin/env python3
"""
Inference script for Indic Parler TTS using the local pure-MLX implementation.

Usage:
    python scripts/infer_indic_parler.py \
        --text "नमस्ते, आप कैसे हैं?" \
        --description "A female speaker delivers clear speech at a moderate pace." \
        --output out.wav

Supported languages: en, hi, bn, ta, te, kn, ml, mr, gu, pa, or, as, ur, sa, ne, sd, ks, om
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run ai4bharat/indic-parler-tts with the local MLX backend."
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--description",
        default="A female speaker delivers a slightly expressive and animated speech "
                "with a moderate speed and pitch. The recording is of very high quality, "
                "with the speaker's voice sounding clear and very close up.",
        help="Style description prompt",
    )
    parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max audio seconds")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0=disabled)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability (1.0=disabled)")
    parser.add_argument("--repo-id", default="ai4bharat/indic-parler-tts", help="HF repo ID")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed")
    parser.add_argument(
        "--no-load-audit",
        action="store_true",
        help="Skip HF-to-MLX weight mapping audit before loading",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Warn instead of failing if the weight mapping audit finds missing keys",
    )
    args = parser.parse_args()

    from models.indic_parler_tts import load_model, generate
    import soundfile as sf

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tokenizers = load_model(
        args.repo_id,
        strict_load=not args.allow_partial_load,
        audit_load=not args.no_load_audit,
    )
    print(f"[infer] Loaded in {time.time() - t0:.1f}s")

    print(f"[infer] Synthesizing: {args.text[:80]}")
    t1 = time.time()
    audio = generate(
        model, tokenizers,
        description=args.description,
        text=args.text,
        max_audio_length_s=args.max_duration,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    sf.write(str(out_path), audio, 44100)
    audio_s = len(audio) / 44100
    print(f"[infer] Generated {audio_s:.2f}s in {time.time() - t1:.1f}s")
    print(f"[infer] Saved: {out_path}")


if __name__ == "__main__":
    main()
