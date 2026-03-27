"""
audio_utils.py — resampling, normalization, augmentation helpers.
Works entirely with numpy/soundfile so it can be used in dataset workers
before tensors are created.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Load any audio file (wav/mp3/flac/ogg/m4a) and resample to target_sr.
    Returns (waveform float32 mono, sample_rate).
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    # Mix down to mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    audio = audio[:, 0]  # (T,)

    # Resample if needed
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)

    return audio, target_sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, audio, sr)


# ──────────────────────────────────────────────────────────────────────────────
# Resampling (scipy fallback, no torch required)
# ──────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        # Fallback: linear interpolation (low quality but dependency-free)
        duration = len(audio) / orig_sr
        new_len = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_loudness(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """Peak-normalize then apply rough LUFS shift."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    target_rms = 10 ** (target_db / 20)
    return (audio * (target_rms / rms)).clip(-1.0, 1.0)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_ms: int = 200,
) -> np.ndarray:
    """Trim leading / trailing silence."""
    threshold_lin = 10 ** (threshold_db / 20)
    min_samples = int(sr * min_silence_ms / 1000)
    energy = np.abs(audio)

    above = np.where(energy > threshold_lin)[0]
    if len(above) == 0:
        return audio

    start = max(0, above[0] - min_samples)
    end = min(len(audio), above[-1] + min_samples)
    return audio[start:end]


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_audio(
    audio: np.ndarray,
    sr: int,
    min_dur: float = 0.5,
    max_dur: float = 30.0,
) -> Tuple[bool, str]:
    dur = len(audio) / sr
    if dur < min_dur:
        return False, f"Too short: {dur:.2f}s < {min_dur}s"
    if dur > max_dur:
        return False, f"Too long: {dur:.2f}s > {max_dur}s"
    if np.max(np.abs(audio)) < 1e-4:
        return False, "Audio is silent"
    return True, "ok"


def audio_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate
