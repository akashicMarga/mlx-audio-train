#!/usr/bin/env python3
"""
Run Nemotron-Labs-Audex-2B in MLX: text generation and text-to-speech.

First convert weights (one-time):
    python -m models.audex.convert --out checkpoints/audex_mlx

Text generation:
    python scripts/infer_audex.py --model checkpoints/audex_mlx \
        --text "Explain what RoPE is in one sentence."

Text-to-speech:
    python scripts/infer_audex.py --model checkpoints/audex_mlx \
        --tts "The weather is so good, I want to enjoy the morning in the park." \
        --output out.wav --cfg-scale 1.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="converted MLX checkpoint dir")
    ap.add_argument("--text", default=None, help="prompt for text generation")
    ap.add_argument("--tts", default=None, help="transcription to synthesize")
    ap.add_argument("--output", default="out.wav", help="TTS output wav path")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--cfg-scale", type=float, default=1.0, help="TTS classifier-free guidance (1.0=off)")
    ap.add_argument("--reasoning", action="store_true", help="text: enable <think> reasoning")
    ap.add_argument("--quantize", type=int, default=None, choices=[4, 8], help="quantize the LM to 4 or 8 bits")
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not args.text and not args.tts:
        ap.error("provide --text and/or --tts")

    from models.audex import load_model, text_generate, tts_generate

    model = load_model(args.model, quantize=args.quantize, q_group_size=args.q_group_size)

    if args.text:
        kw = dict(reasoning=args.reasoning)
        if args.max_new_tokens is not None: kw["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None: kw["temperature"] = args.temperature
        if args.top_k is not None: kw["top_k"] = args.top_k
        if args.top_p is not None: kw["top_p"] = args.top_p
        if args.seed is not None: kw["seed"] = args.seed
        print("\n=== TEXT ===")
        print(text_generate(model, args.text, **kw))

    if args.tts:
        import soundfile as sf
        kw = dict(cfg_scale=args.cfg_scale)
        if args.max_new_tokens is not None: kw["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None: kw["temperature"] = args.temperature
        if args.top_k is not None: kw["top_k"] = args.top_k
        if args.top_p is not None: kw["top_p"] = args.top_p
        if args.seed is not None: kw["seed"] = args.seed
        wav = tts_generate(model, args.tts, **kw)
        sf.write(args.output, wav, model.decoder.cfg.sample_rate)
        print(f"\n=== TTS === wrote {args.output} "
              f"({len(wav) / model.decoder.cfg.sample_rate:.2f}s @ {model.decoder.cfg.sample_rate}Hz)")


if __name__ == "__main__":
    main()
