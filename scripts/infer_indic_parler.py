#!/usr/bin/env python3
"""
Inference script for Indic Parler TTS using the local pure-MLX implementation.

Usage:
    python scripts/infer_indic_parler.py \
        --text "नमस्ते, आप कैसे हैं?" \
        --description "A female speaker delivers clear speech at a moderate pace." \
        --output out.wav

    # Streaming with real-time playback (requires: pip install sounddevice):
    python scripts/infer_indic_parler.py \
        --text "नमस्ते, आप कैसे हैं?" \
        --stream

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
        "--stream",
        action="store_true",
        help="Stream audio chunks to speakers as they're generated (requires sounddevice)",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=15,
        help="AR steps per streamed chunk (15 ≈ 270ms TTFA; 50 ≈ 600ms TTFA)",
    )
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
    parser.add_argument(
        "--quantize",
        type=int,
        default=None,
        choices=[4, 8],
        help="Quantize the AR decoder to 4/8-bit MLX (8 recommended: ~3x faster, no audible loss)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Cast model to bfloat16 (best with --quantize 8: ~1.28x realtime, no audible loss)",
    )
    args = parser.parse_args()

    from models.indic_parler_tts import load_model, generate, stream_generate
    import numpy as np
    import soundfile as sf

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tokenizers = load_model(
        args.repo_id,
        strict_load=not args.allow_partial_load,
        audit_load=not args.no_load_audit,
        quantize=args.quantize,
        bf16=args.bf16,
    )
    print(f"[infer] Loaded in {time.time() - t0:.1f}s")

    print(f"[infer] Synthesizing: {args.text[:80]}")

    shared_kwargs = dict(
        description=args.description,
        text=args.text,
        max_audio_length_s=args.max_duration,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    if args.stream:
        try:
            import sounddevice as sd
        except ImportError:
            print("[infer] sounddevice not found — install with: pip install sounddevice")
            sys.exit(1)

        import threading
        import queue as Q

        print(f"[infer] Streaming to speakers (chunk_steps={args.chunk_steps}) ...")
        t1 = time.time()
        chunks = []
        first_chunk_t = None

        # Producer-consumer: generator (main thread, GPU) fills the queue;
        # player (background thread, CPU) drains it continuously.
        # wired_limit/generation_stream must stay on the main thread (Metal GPU context).
        audio_q: Q.Queue = Q.Queue(maxsize=8)

        def _player_worker():
            with sd.OutputStream(samplerate=44100, channels=1, dtype="float32") as out_stream:
                while True:
                    chunk = audio_q.get()
                    if chunk is None:
                        break
                    out_stream.write(chunk)

        player_thread = threading.Thread(target=_player_worker, daemon=True)
        player_thread.start()

        for chunk in stream_generate(
            model, tokenizers, chunk_steps=args.chunk_steps, **shared_kwargs
        ):
            audio_q.put(chunk)  # blocks only if player is >8 chunks behind (never happens)
            chunks.append(chunk)
            if first_chunk_t is None:
                first_chunk_t = time.time() - t1
                print(f"[infer] First audio in {first_chunk_t:.2f}s")

        audio_q.put(None)  # sentinel: tell player we're done
        player_thread.join()
        total_t = time.time() - t1
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio_s = len(audio) / 44100
        print(f"[infer] Streamed {audio_s:.2f}s audio in {total_t:.1f}s")

        sf.write(str(out_path), audio, 44100)
        print(f"[infer] Saved: {out_path}")

    else:
        t1 = time.time()
        audio = generate(model, tokenizers, **shared_kwargs)
        sf.write(str(out_path), audio, 44100)
        audio_s = len(audio) / 44100
        print(f"[infer] Generated {audio_s:.2f}s in {time.time() - t1:.1f}s")
        print(f"[infer] Saved: {out_path}")


if __name__ == "__main__":
    main()
