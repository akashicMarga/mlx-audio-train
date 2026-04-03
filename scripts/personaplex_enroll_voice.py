#!/usr/bin/env python3
"""
personaplex_enroll_voice.py — enroll a PersonaPlex voice prompt inside mlx-audio-train.

This keeps PersonaPlex fine-tuning, inference, and speaker onboarding self-contained
in this repo while still allowing generated datasets to live elsewhere.

Usage:
    python scripts/personaplex_enroll_voice.py \
      --input /absolute/path/to/hindi_speaker.wav \
      --output /Users/akashsingh/Documents/mlx-audio-artifacts/voices/hindi_speaker_01.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import numpy as np
import rustymimi
import sphn

from models.personaplex import Lm
from models.personaplex.persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    load_lm_weights,
    seed_all,
)
from models.personaplex.training import apply_lora_to_transformer
from models.personaplex.utils import reshape_input_tokens
from train.lora import load_personaplex_adapters


def extract_voice_embeddings(
    model: Lm,
    audio_tokens_sequence: list[mx.array],
    num_layers: int,
) -> tuple[list[mx.array], mx.array]:
    d_model = model.cfg.transformer.d_model
    audio_codebooks = model.cfg.audio_codebooks
    text_pad = model.cfg.text_out_vocab_size
    audio_pad = model.cfg.audio_padding_token

    for cache in model.transformer_cache:
        cache.reset()

    layer_accumulators = [mx.zeros((1, 1, d_model)) for _ in range(num_layers)]
    all_cache_tokens: list[mx.array] = []
    num_steps = 0

    for audio_frame in audio_tokens_sequence:
        user_cbs = audio_frame.shape[1]
        assistant_cbs = audio_codebooks - user_cbs

        text_token = mx.full((1, 1, 1), text_pad, dtype=mx.int32)
        assistant_tokens = mx.full((1, assistant_cbs, 1), audio_pad, dtype=mx.int32)
        full_input = mx.concatenate([text_token, assistant_tokens, audio_frame], axis=1)
        all_cache_tokens.append(full_input)

        xs = model.embed_codes(full_input[:, :, 0:1])
        for layer_idx, layer in enumerate(model.transformer.layers):
            xs = layer(xs, cache=model.transformer_cache[layer_idx])
            layer_accumulators[layer_idx] = layer_accumulators[layer_idx] + xs

        xs = model.out_norm(xs)
        mx.eval(xs)
        num_steps += 1

    embeddings = []
    for acc in layer_accumulators:
        avg = acc / num_steps
        mx.eval(avg)
        embeddings.append(avg)

    cache_tokens = mx.concatenate(all_cache_tokens, axis=2)
    return embeddings, cache_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a PersonaPlex voice prompt")
    parser.add_argument("--input", required=True, type=str, help="Reference WAV/MP3 with 10-30s of clean speech")
    parser.add_argument("--output", required=True, type=str, help="Output voice embedding file (.pt or .npz)")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--lm-config", type=str, default=None)
    parser.add_argument("--model-file", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8], default=None)
    parser.add_argument("--lora-weights", type=str, default=None, help="Optional PersonaPlex adapters.npz to load before enrollment")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading PersonaPlex model...")
    lm_config = get_lm_config(args.lm_config, args.hf_repo)
    model_path, _ = get_or_download_model_file(args.hf_repo, args.quantized, args.model_file)
    mimi_path = get_or_download_mimi(args.hf_repo, args.mimi_weight)

    model = Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    load_lm_weights(model, lm_config, model_path, args.quantized)

    if args.lora_weights:
        print(f"Loading LoRA adapters: {args.lora_weights}")
        apply_lora_to_transformer(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        load_personaplex_adapters(model, args.lora_weights)

    print(f"Encoding reference audio: {args.input}")
    tokenizer = rustymimi.Tokenizer(mimi_path, num_codebooks=8)
    pcm, _ = sphn.read(args.input, sample_rate=24000)
    if pcm.ndim == 2 and pcm.shape[0] > 1:
        pcm = pcm[0:1]

    max_samples = int(args.max_seconds * 24000)
    if pcm.shape[-1] > max_samples:
        print(f"Trimming reference audio to {args.max_seconds:.1f}s")
        pcm = pcm[:, :max_samples]

    total_samples = pcm.shape[-1]
    chunk_size = 1920
    steps = (total_samples + chunk_size - 1) // chunk_size
    user_codebooks = lm_config.audio_codebooks - lm_config.audio_tokens_per_stream

    audio_frames: list[mx.array] = []
    for idx in range(steps):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total_samples)
        chunk = pcm[:, start:end]
        if chunk.shape[-1] < chunk_size:
            pad = chunk_size - chunk.shape[-1]
            chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")
        encoded = tokenizer.encode_step(chunk[None, :])
        audio_frames.append(reshape_input_tokens(encoded, user_codebooks))

    print(f"Encoded {len(audio_frames)} frames from {total_samples / 24000:.2f}s of audio")
    embeddings, cache = extract_voice_embeddings(model, audio_frames, lm_config.transformer.num_layers)

    try:
        import torch

        embeddings_t = torch.stack(
            [torch.from_numpy(np.array(emb.astype(mx.float32))) for emb in embeddings]
        )
        cache_t = torch.from_numpy(np.array(cache))
        save_path = output_path.with_suffix(".pt")
        torch.save({"embeddings": embeddings_t, "cache": cache_t}, save_path)
        print(f"Saved voice prompt: {save_path}")
    except ImportError:
        save_path = output_path.with_suffix(".npz")
        np.savez(
            save_path,
            **{f"embedding_{i}": np.array(emb) for i, emb in enumerate(embeddings)},
            cache=np.array(cache),
        )
        print(f"Saved voice prompt: {save_path}")
        print("torch is not installed, so the prompt was written as .npz")


if __name__ == "__main__":
    main()
