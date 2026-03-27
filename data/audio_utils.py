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


# ──────────────────────────────────────────────────────────────────────────────
# Mel spectrogram  (matches official Qwen3-TTS speaker encoder input)
# ──────────────────────────────────────────────────────────────────────────────

def mel_spectrogram(
    audio:      np.ndarray,
    sr:         int   = 24000,
    n_fft:      int   = 1024,
    n_mels:     int   = 128,
    hop_length: int   = 256,
    win_length: int   = 1024,
    fmin:       float = 0.0,
    fmax:       float = 12000.0,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram compatible with the Qwen3-TTS speaker encoder.

    Default parameters match the official sft_12hz.py / dataset.py:
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000

    Args:
        audio:      float32 mono waveform, already at `sr`
        sr:         sample rate of `audio` (must be 24000 for Qwen3-TTS)

    Returns:
        float32 array of shape [T_frames, n_mels]
    """
    import scipy.signal as ss

    # STFT — scipy stft uses half-open interval so boundary/padded don't
    # matter as long as parameters match the reference implementation.
    _, _, Zxx = ss.stft(
        audio.astype(np.float32),
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(Zxx) ** 2  # [n_fft//2+1, T]

    filters = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)  # [n_mels, n_fft//2+1]
    mel = filters @ power                                       # [n_mels, T]

    log_mel = np.log(np.maximum(mel, 1e-5))
    return log_mel.T.astype(np.float32)   # [T, n_mels]


def _mel_filterbank(
    sr:     int,
    n_fft:  int,
    n_mels: int,
    fmin:   float,
    fmax:   float,
) -> np.ndarray:
    """Build HTK triangular mel filterbank [n_mels, n_fft//2+1]."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    freqs   = np.linspace(0, sr / 2, n_freqs)   # [n_freqs]

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)

    # Vectorised triangular ramps — shape [n_mels, n_freqs]
    l = hz_pts[:-2, None]   # left  edge
    c = hz_pts[1:-1, None]  # centre
    r = hz_pts[2:, None]    # right edge
    f = freqs[None, :]      # bin frequencies

    ramp_up   = (f - l) / np.where(c == l, 1.0, c - l)
    ramp_down = (r - f) / np.where(r == c, 1.0, r - c)
    return np.maximum(0.0, np.minimum(ramp_up, ramp_down)).astype(np.float32)
