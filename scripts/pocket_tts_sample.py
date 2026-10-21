#!/usr/bin/env python3
"""
Generate speech from any Pocket TTS checkpoint with the MLX FlowLM, at any guidance.

The official `pocket-tts generate` has no guidance option: it suits distilled students
(guidance baked in, cfg 1). Teachers — e.g. a Hindi 24-layer fine-tune — are meant to be
sampled guided (cfg 2), which this script does. Decoding uses the official Mimi
(pip install pocket-tts).

Usage:
    python scripts/pocket_tts_sample.py --config checkpoints/pocket-hindi-teacher-24l/step_0020000/config.yaml \\
        --cfg 2 --voice voice.wav --text "आज मौसम बहुत अच्छा है" --normalize-text hindi --out teacher.wav
    python scripts/pocket_tts_sample.py --config checkpoints/pocket-hindi-4l/step_0050000/config.yaml \\
        --voice voice.wav --text "आज मौसम बहुत अच्छा है" --normalize-text hindi --out student.wav
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.codec import Mimi, load_wav  # noqa: E402
from models.pocket_tts.data import load_tokenizer  # noqa: E402
from models.pocket_tts.text import normalizer  # noqa: E402
from models.pocket_tts.weights import build_flow_lm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="pocket-tts config (local or hf://)")
    p.add_argument("--weights", default="config", help="default: the config's weights_path")
    p.add_argument("--text", required=True)
    p.add_argument("--voice", default=None, help="voice prompt wav (needs a voice-cloning checkpoint)")
    p.add_argument("--voice-sec", type=float, default=5.0)
    p.add_argument("--cfg", type=float, default=1.0, help="1 for distilled students, ~2 for teachers")
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--eos-threshold", type=float, default=-1.0)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--normalize-text", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pocket_sample.wav")
    args = p.parse_args()

    import soundfile as sf

    mx.random.seed(args.seed)
    model, raw = build_flow_lm(args.config, weights=args.weights)
    mimi = Mimi(args.config)
    norm = normalizer(args.normalize_text)
    text = norm(args.text) if norm else args.text
    tokens = np.array(load_tokenizer(model.args.tokenizer_path)(text), dtype=np.int32)

    voice = None
    if args.voice:
        if "speaker_proj_weight" not in model:
            sys.exit("this checkpoint has no voice projection (a -without-voice-cloning release)")
        voice = mx.array(mimi.encode(load_wav(args.voice, duration=args.voice_sec)))

    t0 = time.perf_counter()
    latents = model.generate_latents(tokens, voice, max_frames=int(args.max_sec * 12.5), temp=args.temp,
                                     cfg_coef=args.cfg, eos_threshold=args.eos_threshold)
    gen = time.perf_counter() - t0
    audio = mimi.decode(np.array(latents))
    sf.write(args.out, audio, 24000)
    dur = len(audio) / 24000
    print(f"[sample] {latents.shape[0]} frames, {dur:.2f}s audio in {gen:.2f}s "
          f"(RTF {gen / max(dur, 1e-6):.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
