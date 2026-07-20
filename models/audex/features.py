"""
Audio -> NV-Whisper mel features for Audex understanding.

Matches the WhisperFeatureExtractor path Qwen2Audio uses: 16 kHz mono, 128-mel
log-spectrogram (n_fft 400, hop 160), fixed 30 s clips padded to 3000 frames.
Reuses mlx_whisper's mel implementation (identical constants + normalization).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

SAMPLE_RATE = 16000
CLIP_SECONDS = 30.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)   # 480000
N_MELS = 128
EMBEDDINGS_PER_CLIP = 750                     # encoder output frames per 30s clip


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load to mono float32 in [-1, 1] at target_sr."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    m = float(np.abs(audio).max()) if audio.size else 0.0
    if m > 1.0:
        audio = audio / m
    return audio.astype(np.float32, copy=False)


def _split_clips(audio: np.ndarray) -> list[np.ndarray]:
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    n = max(1, math.ceil(audio.shape[0] / N_SAMPLES))
    clips = []
    for i in range(n):
        clip = audio[i * N_SAMPLES : (i + 1) * N_SAMPLES]
        if clip.shape[0] < N_SAMPLES:
            clip = np.pad(clip, (0, N_SAMPLES - clip.shape[0]))
        clips.append(clip.astype(np.float32, copy=False))
    return clips


def extract_features(audio: np.ndarray) -> mx.array:
    """Return NV-Whisper features shaped (num_clips, 128, 3000)."""
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim

    feats = []
    for clip in _split_clips(audio):
        mel = log_mel_spectrogram(pad_or_trim(mx.array(clip), N_SAMPLES), n_mels=N_MELS)  # (3000, 128)
        feats.append(mel.T)                                                                # (128, 3000)
    return mx.stack(feats, axis=0)


def num_audio_embeddings(features: mx.array) -> int:
    """Total <so_embedding> placeholders = num_clips * 750."""
    return features.shape[0] * EMBEDDINGS_PER_CLIP
