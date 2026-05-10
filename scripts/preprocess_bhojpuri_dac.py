"""
scripts/preprocess_bhojpuri_dac.py

Pre-encodes Bhojpuri audio files with the Parler TTS DAC encoder.

Reads each .wav listed in a JSONL file, resamples to 44100 Hz, and writes
a companion .codec.npy file with shape (9, T_frames) — 9 codebook indices
at ~86 Hz (44100 / 512 hop).

Run this ONCE before training to avoid encoding during the training loop.

Usage:
    python scripts/preprocess_bhojpuri_dac.py \\
        --jsonl /path/to/bhojpuri/train.jsonl \\
        --model-path ai4bharat/indic-parler-tts \\
        --batch-size 1

The script writes <audio_path>.codec.npy next to each wav file.
The JSONL itself is not modified; the trainer/processor detects the codec
file by checking for the .codec.npy sidecar.
"""

import argparse
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

_DAC_SR = 44100
_HOP    = 512   # 2 * 4 * 8 * 8 encoder strides


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _load_resample(path: str) -> np.ndarray:
    """Load wav, mix to mono float32, resample to 44100 Hz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if sr != _DAC_SR:
        g = gcd(sr, _DAC_SR)
        audio = resample_poly(audio, _DAC_SR // g, sr // g).astype(np.float32)
    return audio


# ── Minimal MLX DAC Encoder ───────────────────────────────────────────────────
# Implements only the encode path (audio → codes). Weights are loaded directly
# from the HF safetensors without weight-normalization bookkeeping.

def _snake(x: mx.array, alpha: mx.array) -> mx.array:
    """Snake activation; alpha shape (1, 1, C) for MLX channels-last."""
    return x + (1.0 / (alpha + 1e-9)) * mx.sin(alpha * x) ** 2


class _ResidualUnit:
    """Dilated conv residual block in the DAC encoder."""
    def __init__(self, dim: int, dilation: int, weights: dict, prefix: str):
        self.w1 = weights[f"{prefix}.conv1.weight"]  # (out, k, in) MLX
        self.b1 = weights[f"{prefix}.conv1.bias"]
        self.a1 = weights[f"{prefix}.snake1.alpha"]  # (1, 1, C)
        self.w2 = weights[f"{prefix}.conv2.weight"]
        self.b2 = weights[f"{prefix}.conv2.bias"]
        self.a2 = weights[f"{prefix}.snake2.alpha"]
        pad = ((7 - 1) * dilation) // 2
        self.dil = dilation
        self.pad = pad

    def __call__(self, x: mx.array) -> mx.array:
        y = _snake(x, self.a1)
        y = mx.conv1d(y, self.w1, stride=1, padding=self.pad, dilation=self.dil) + self.b1
        y = _snake(y, self.a2)
        y = mx.conv1d(y, self.w2, stride=1, padding=0) + self.b2
        # trim if causal asymmetry
        diff = (x.shape[1] - y.shape[1]) // 2
        if diff > 0:
            x = x[:, diff:-diff, :]
        return x + y


class _EncoderBlock:
    """One stride block: 3 residual units + snake + strided conv."""
    def __init__(self, dim: int, stride: int, weights: dict, prefix: str):
        in_dim = dim // 2
        self.res = [
            _ResidualUnit(in_dim, d, weights, f"{prefix}.res_unit{i+1}")
            for i, d in enumerate([1, 3, 9])
        ]
        self.snake = weights[f"{prefix}.snake1.alpha"]  # (1, 1, in_dim)
        self.w_down = weights[f"{prefix}.conv1.weight"]  # strided conv
        self.b_down = weights[f"{prefix}.conv1.bias"]
        self.stride = stride
        pad = math.ceil(stride / 2)
        self.pad = pad

    def __call__(self, x: mx.array) -> mx.array:
        for ru in self.res:
            x = ru(x)
        x = _snake(x, self.snake)
        x = mx.conv1d(x, self.w_down, stride=self.stride, padding=self.pad) + self.b_down
        return x


class _DACEncoder:
    """
    Full DAC audio encoder: audio [B, T, 1] → latent [B, T', latent_dim].
    Encoder strides for Parler TTS DAC: [2, 4, 8, 8] → hop = 512.
    """
    def __init__(self, weights: dict, prefix: str = "audio_encoder.encoder"):
        strides  = [2, 4, 8, 8]
        d_model  = 64
        latent   = 1024

        # Initial conv: (1 → d_model, k=7, pad=3)
        self.w_in = weights[f"{prefix}.conv1.weight"]  # (64, 7, 1)
        self.b_in = weights[f"{prefix}.conv1.bias"]

        # Encoder blocks
        self.blocks = []
        cur_dim = d_model
        for i, stride in enumerate(strides):
            out_dim = cur_dim * 2
            self.blocks.append(
                _EncoderBlock(out_dim, stride, weights, f"{prefix}.block.{i}")
            )
            cur_dim = out_dim  # 128, 256, 512, 1024

        # Final snake + latent proj
        self.snake_final = weights[f"{prefix}.snake1.alpha"]  # (1, 1, 1024)
        self.w_out = weights[f"{prefix}.conv2.weight"]  # (1024, 3, 1024)
        self.b_out = weights[f"{prefix}.conv2.bias"]

    def __call__(self, audio: mx.array) -> mx.array:
        """
        audio: [B, T] float32 mono
        returns: latent [B, T', 1024]
        """
        x = audio[:, :, None]  # [B, T, 1]
        x = mx.conv1d(x, self.w_in, stride=1, padding=3) + self.b_in
        for block in self.blocks:
            x = block(x)
        x = _snake(x, self.snake_final)
        x = mx.conv1d(x, self.w_out, stride=1, padding=1) + self.b_out
        return x  # [B, T', 1024]


def _normalize_l2(x: mx.array, dim: int = -1, eps: float = 1e-12) -> mx.array:
    norm = mx.maximum(mx.sqrt((x * x).sum(axis=dim, keepdims=True)), eps)
    return x / norm


class _DACQuantizer:
    """
    Residual vector quantizer — encode path only (latent → codes).
    Returns codes [9, T'] (int32 indices for each codebook).
    """
    def __init__(self, weights: dict, n_codebooks: int = 9,
                 prefix: str = "audio_encoder.quantizer"):
        self.n_codebooks = n_codebooks
        self.in_proj  = []  # (8, 1024) weights per codebook
        self.codebooks = []  # (1024, 8) per codebook

        for i in range(n_codebooks):
            q_pfx = f"{prefix}.quantizers.{i}"
            # in_proj: loaded as (8, 1, 1024) in MLX (transposed from PyTorch (8, 1024, 1))
            # → squeeze kernel dim to get (8, 1024) for matmul
            w_in = weights[f"{q_pfx}.in_proj.weight"][:, 0, :]   # (8, 1024)
            b_in = weights[f"{q_pfx}.in_proj.bias"]               # (8,)
            self.in_proj.append((w_in, b_in))
            # codebook: (1024, 8)
            self.codebooks.append(weights[f"{q_pfx}.codebook.weight"])

    def __call__(self, z: mx.array) -> mx.array:
        """
        z: [B, T', 1024] encoder latent
        Returns codes: [B, 9, T'] int32
        """
        B, T, D = z.shape
        residual = z
        all_codes = []

        for i in range(self.n_codebooks):
            w_in, b_in = self.in_proj[i]
            # Project latent to codebook dim (8)
            z_e = residual.reshape(B * T, D) @ w_in.T + b_in  # [B*T, 8]
            z_e = z_e.reshape(B, T, 8)

            cb = self.codebooks[i]  # [1024, 8]

            # Normalized cosine lookup
            z_flat = z_e.reshape(B * T, 8)
            z_norm = _normalize_l2(z_flat)
            cb_norm = _normalize_l2(cb)
            sims = z_norm @ cb_norm.T  # [B*T, 1024]
            indices = sims.argmax(axis=-1).reshape(B, T)  # [B, T]
            all_codes.append(indices)

            # Compute quantized vector and subtract from residual
            # out_proj: (1024, 8) codebook → latent space via out_proj
            # For residual update we need the dequantized value
            # z_q_e[b,t] = cb[indices[b,t]]  (un-normalized for residual)
            idx_flat = indices.reshape(B * T)
            z_q_e = cb[idx_flat].reshape(B, T, 8)  # [B, T, 8]

            # out_proj back to latent dim
            # We need out_proj weights to get back to D=1024 for residual
            # But for pure encoding we only need the codes — subtract reconstructed latent
            # Using out_proj weights stored in the model
            # z_q = out_proj(z_q_e) → [B, T, 1024]
            # For now we store out_proj separately
            z_q = z_q_e @ self._out_proj_w[i].T + self._out_proj_b[i]  # [B, T, 1024]
            residual = residual - z_q

        return mx.stack(all_codes, axis=1)  # [B, 9, T]

    def set_out_proj(self, weights, prefix="audio_encoder.quantizer"):
        """Load out_proj weights needed for residual computation."""
        self._out_proj_w = []
        self._out_proj_b = []
        for i in range(self.n_codebooks):
            q_pfx = f"{prefix}.quantizers.{i}"
            # Loaded as (1024, 1, 8) in MLX (transposed from PyTorch (1024, 8, 1))
            # → squeeze kernel dim to get (1024, 8) for matmul
            w = weights[f"{q_pfx}.out_proj.weight"][:, 0, :]  # (1024, 8)
            b = weights[f"{q_pfx}.out_proj.bias"]             # (1024,)
            self._out_proj_w.append(mx.array(w))
            self._out_proj_b.append(mx.array(b))


# ── Weight loading ────────────────────────────────────────────────────────────

def _load_hf_weights(repo_or_path: str) -> dict:
    """Load HF safetensors and transpose conv weights to MLX format."""
    import safetensors.numpy as st
    from huggingface_hub import hf_hub_download

    if os.path.isfile(repo_or_path):
        sft_path = repo_or_path
    else:
        sft_path = hf_hub_download(repo_or_path, "model.safetensors")

    raw = st.load_file(sft_path)

    mlx_weights = {}
    for k, v in raw.items():
        if not k.startswith("audio_encoder.encoder") and not k.startswith("audio_encoder.quantizer"):
            continue
        arr = v.astype(np.float32)
        if k.endswith(".weight") and arr.ndim == 3:
            # PyTorch conv: (out, in, k) → MLX: (out, k, in)
            arr = arr.transpose(0, 2, 1)
        elif k.endswith(".alpha") and arr.ndim == 3:
            # PyTorch snake: (1, C, 1) → MLX: (1, 1, C)
            arr = arr.transpose(0, 2, 1)
        mlx_weights[k] = mx.array(arr)

    return mlx_weights


def _build_dac_encoder(weights: dict):
    enc = _DACEncoder(weights)
    quant = _DACQuantizer(weights)
    quant.set_out_proj(weights)
    return enc, quant


# ── Encoding pipeline ─────────────────────────────────────────────────────────

def _encode_audio(enc: _DACEncoder, quant: _DACQuantizer, audio: np.ndarray) -> np.ndarray:
    """
    audio: float32 mono array at 44100 Hz
    Returns: (9, T_frames) int32 codec codes
    """
    # Pad to multiple of hop length
    length = len(audio)
    padded_len = math.ceil(length / _HOP) * _HOP
    if padded_len > length:
        audio = np.pad(audio, (0, padded_len - length))

    x = mx.array(audio[None, :])  # [1, T]
    z = enc(x)                    # [1, T', 1024]
    codes = quant(z)              # [1, 9, T']
    mx.eval(codes)
    codes_np = np.array(codes[0]).astype(np.int16)  # [9, T'] — int16 saves disk
    return codes_np


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        nargs="+",
        default=[
            "/Users/akashsingh/Documents/exps/bhojpuri/train.jsonl",
            "/Users/akashsingh/Documents/exps/bhojpuri/val.jsonl",
        ],
        help="JSONL files to process (train and/or val)",
    )
    ap.add_argument(
        "--model-path",
        default="ai4bharat/indic-parler-tts",
        help="HF repo ID or path to safetensors file",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Re-encode even if .codec.npy exists"
    )
    args = ap.parse_args()

    print("Loading DAC encoder weights …")
    weights = _load_hf_weights(args.model_path)
    enc, quant = _build_dac_encoder(weights)
    print("DAC encoder ready.")

    for jsonl_path in args.jsonl:
        if not os.path.exists(jsonl_path):
            print(f"Skipping missing file: {jsonl_path}")
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]

        print(f"\nProcessing {jsonl_path} ({len(entries)} clips) …")
        done, skipped, errors = 0, 0, 0

        for i, entry in enumerate(entries):
            wav_path = entry["audio"]
            codec_path = wav_path + ".codec.npy"

            if os.path.exists(codec_path) and not args.overwrite:
                skipped += 1
                continue

            try:
                audio = _load_resample(wav_path)
                codes = _encode_audio(enc, quant, audio)
                np.save(codec_path, codes)
                done += 1
            except Exception as e:
                print(f"  ERROR {wav_path}: {e}")
                errors += 1

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(entries)} done={done} skipped={skipped} err={errors}")

        print(f"  Done: {done}, Skipped: {skipped}, Errors: {errors}")

    print("\nPre-encoding complete.")


if __name__ == "__main__":
    main()
