#!/usr/bin/env python3
"""
Audex speech-to-speech (MLX): speak to it, it replies in speech.

Pipeline (all one model): input speech -> NV-Whisper encoder + LLM (understanding)
-> text reply -> speech codec tokens -> Audex speech decoder -> 16 kHz reply wav.

Add the audio encoder to your checkpoint once (if not already present):
    python -m models.audex.convert --out checkpoints/audex_mlx --only-audio

Speech-to-speech from a wav:
    python scripts/audex_s2s.py --model checkpoints/audex_mlx \
        --input question.wav --output reply.wav

Understand only (speech -> text):
    python scripts/audex_s2s.py --model checkpoints/audex_mlx \
        --input question.wav --understand-only

Record from mic (needs sounddevice), then reply:
    python scripts/audex_s2s.py --model checkpoints/audex_mlx --record 5 --output reply.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def record_mic(seconds: float, sr: int = 16000):
    import sounddevice as sd
    import numpy as np
    print(f"[rec] recording {seconds}s ... speak now")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("[rec] done")
    return audio.reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="converted MLX checkpoint dir")
    ap.add_argument("--input", default=None, help="input speech wav")
    ap.add_argument("--record", type=float, default=None, help="record N seconds from mic instead of --input")
    ap.add_argument("--output", default="reply.wav", help="reply wav path")
    ap.add_argument("--instruction", default="", help="optional extra text context for the model")
    ap.add_argument("--understand-only", action="store_true", help="just print the speech->text result")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--tts-cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.input and args.record is None:
        ap.error("provide --input <wav> or --record <seconds>")

    from models.audex import load_model, audio_generate, s2s_generate

    model = load_model(args.model)
    if model.audio_tower is None:
        sys.exit("This checkpoint has no audio encoder. Run: "
                 "python -m models.audex.convert --out <dir> --only-audio")

    audio = record_mic(args.record) if args.record is not None else args.input

    if args.understand_only:
        text = audio_generate(model, audio, args.instruction,
                              max_new_tokens=args.max_new_tokens, seed=args.seed)
        print("\n=== HEARD / RESPONSE ===")
        print(text)
        return

    transcript, reply, wav = s2s_generate(model, audio, args.instruction,
                                          max_new_tokens=args.max_new_tokens,
                                          tts_cfg_scale=args.tts_cfg_scale,
                                          seed=args.seed, return_transcript=True)
    import soundfile as sf
    sr = model.decoder.cfg.sample_rate
    sf.write(args.output, wav, sr)
    print("\n=== HEARD (transcript) ===")
    print(transcript)
    print("\n=== REPLY (text) ===")
    print(reply)
    print(f"\n=== REPLY (speech) === wrote {args.output} ({len(wav) / sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
