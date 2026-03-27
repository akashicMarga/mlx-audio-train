"""
processors/qwen3_tts.py — Qwen3-TTS specific processor.

Converts TTSSample → model-ready tensors for training.

Pipeline:
    audio  → Qwen3-TTS-Tokenizer-12Hz → codec_ids  (int32 [T_codec])
    text   → Qwen3 text tokenizer     → text_ids   (int32 [T_text])

Training batch tensors:
    {
        "text_ids":       [B, T_text]       int32
        "codec_ids":      [B, T_codec]      int32
        "text_lengths":   [B]               int32
        "codec_lengths":  [B]               int32
        "ref_codec_ids":  [B, T_ref]        int32  (optional, for CustomVoice)
    }
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlx.core as mx


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Qwen3TTSProcessorConfig:
    model_id:         str   = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
    tokenizer_id:     str   = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    sample_rate:      int   = 24000
    max_text_len:     int   = 256
    max_codec_len:    int   = 1500   # ~125s at 12Hz
    pad_codec_id:     int   = 0
    pad_text_id:      int   = 0
    speaker_name:     str   = "speaker_0"
    # Pass the already-loaded model's speech_tokenizer directly (avoids re-loading)
    speech_tokenizer: Any   = None


# ──────────────────────────────────────────────────────────────────────────────
# Processor
# ──────────────────────────────────────────────────────────────────────────────

class Qwen3TTSProcessor:
    """
    Handles tokenization for Qwen3-TTS training.

    Audio tokenizer: Qwen3-TTS-Tokenizer-12Hz (converts 24kHz audio → 12Hz codec tokens)
    Text tokenizer:  Qwen3 tokenizer (same as used in inference)

    This mirrors the official prepare_data.py but in pure Python/MLX
    so it can run on Apple Silicon without CUDA.
    """

    def __init__(self, config: Qwen3TTSProcessorConfig):
        self.config = config
        self._speech_tokenizer = None
        self._text_tokenizer   = None
        self._loaded           = False

    def _load(self):
        if self._loaded:
            return

        # Load speech tokenizer (audio → codec tokens)
        # Prefer the pre-loaded tokenizer passed in config (from model.speech_tokenizer)
        if self.config.speech_tokenizer is not None:
            print("[processor] Using pre-loaded speech tokenizer from model")
            self._speech_tokenizer = self.config.speech_tokenizer
        else:
            try:
                from mlx_audio.tts.utils import load_model as mlx_load
                print(f"[processor] Loading speech tokenizer via model: {self.config.model_id}")
                _model = mlx_load(self.config.model_id)
                self._speech_tokenizer = _model.speech_tokenizer
            except Exception as e:
                print(f"[processor] Warning: Could not load speech tokenizer: {e}")
                print("[processor] Will use placeholder codec IDs (for testing only)")
                self._speech_tokenizer = None

        # Load text tokenizer
        try:
            from transformers import AutoTokenizer
            print(f"[processor] Loading text tokenizer from: {self.config.model_id}")
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, trust_remote_code=True
            )
        except Exception as e:
            print(f"[processor] Warning: Could not load text tokenizer: {e}")
            self._text_tokenizer = None

        self._loaded = True

    def encode_audio(self, audio: np.ndarray, audio_path: str = None) -> np.ndarray:
        """
        Convert raw audio waveform → codec token IDs at 12Hz.
        Returns int32 array of shape [T_codec].

        If audio_path is provided and a matching .codec.npy file exists (created by
        preprocess_dataset.py), the pre-computed IDs are loaded from disk instead of
        running the speech tokenizer. This avoids OOM during training.
        """
        # Fast path: load pre-computed codec IDs from disk
        if audio_path:
            codec_npy = Path(audio_path).with_suffix(".codec.npy")
            if codec_npy.exists():
                return np.load(str(codec_npy)).astype(np.int32)

        # Slow path: run speech tokenizer (needed during preprocess_dataset.py or smoke-test)
        self._load()

        if self._speech_tokenizer is None:
            # Placeholder: return dummy tokens for testing pipeline
            dur = len(audio) / self.config.sample_rate
            n_tokens = max(1, int(dur * 12))
            return np.zeros(n_tokens, dtype=np.int32)

        # Ensure mono float32
        if audio.ndim == 2:
            audio = audio.mean(axis=1)   # stereo (T,2) → mono (T,)
        audio = audio.astype(np.float32)
        # encode() expects [batch, 1, samples]
        audio_mx = mx.array(audio)[None, None, :]  # [1, 1, T]
        codes = self._speech_tokenizer.encode(audio_mx)  # [1, n_q, T_codec]
        # Take first quantizer → flat token IDs [T_codec]
        codec_ids = codes[0, 0]
        mx.eval(codec_ids)   # flush MLX graph immediately to avoid memory accumulation
        return np.array(codec_ids, dtype=np.int32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Convert text → token IDs.
        Returns int32 array of shape [T_text].
        """
        self._load()

        if self._text_tokenizer is None:
            # Placeholder
            return np.array([1, 2, 3], dtype=np.int32)

        ids = self._text_tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_text_len,
        )
        return np.array(ids, dtype=np.int32)

    def __call__(self, sample) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a TTSSample into a training-ready dict.
        Returns None if processing fails.
        """
        from ..base_dataset import TTSSample

        if not isinstance(sample, TTSSample):
            return sample  # already processed

        try:
            text_ids  = self.encode_text(sample.text)
            codec_ids = self.encode_audio(sample.audio, audio_path=sample.audio_path)

            # Truncate codec if too long
            if len(codec_ids) > self.config.max_codec_len:
                codec_ids = codec_ids[: self.config.max_codec_len]

            result = {
                "text_ids":      text_ids,
                "codec_ids":     codec_ids,
                "text_length":   len(text_ids),
                "codec_length":  len(codec_ids),
                "text":          sample.text,
                "audio_path":    sample.audio_path,
            }

            # Ref audio for CustomVoice mode
            if sample.ref_audio is not None:
                ref_codec = self.encode_audio(sample.ref_audio)
                result["ref_codec_ids"]    = ref_codec
                result["ref_codec_length"] = len(ref_codec)

            return result

        except Exception as e:
            print(f"[processor] Failed to process {sample.audio_path}: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Collation: list of processed dicts → padded MLX batch
# ──────────────────────────────────────────────────────────────────────────────

def collate_qwen3(
    samples:      List[Dict],
    pad_text_id:  int = 0,
    pad_codec_id: int = 0,
) -> Dict[str, mx.array]:
    """
    Pad a list of processed samples into a single MLX batch dict.

    Returns:
        text_ids:       [B, T_text_max]   int32
        codec_ids:      [B, T_codec_max]  int32
        text_lengths:   [B]               int32
        codec_lengths:  [B]               int32
        text_mask:      [B, T_text_max]   bool
        codec_mask:     [B, T_codec_max]  bool
    """
    samples = [s for s in samples if s is not None]
    if not samples:
        return {}

    B = len(samples)
    T_text  = max(s["text_length"]  for s in samples)
    T_codec = max(s["codec_length"] for s in samples)

    text_ids  = np.full((B, T_text),  pad_text_id,  dtype=np.int32)
    codec_ids = np.full((B, T_codec), pad_codec_id, dtype=np.int32)

    for i, s in enumerate(samples):
        tl = s["text_length"]
        cl = s["codec_length"]
        text_ids[i,  :tl] = s["text_ids"][:tl]
        codec_ids[i, :cl] = s["codec_ids"][:cl]

    text_lengths  = np.array([s["text_length"]  for s in samples], dtype=np.int32)
    codec_lengths = np.array([s["codec_length"] for s in samples], dtype=np.int32)

    # Padding masks (True = valid token)
    text_mask  = (np.arange(T_text)[None,  :] < text_lengths[:,  None])
    codec_mask = (np.arange(T_codec)[None, :] < codec_lengths[:, None])

    return {
        "text_ids":      mx.array(text_ids),
        "codec_ids":     mx.array(codec_ids),
        "text_lengths":  mx.array(text_lengths),
        "codec_lengths": mx.array(codec_lengths),
        "text_mask":     mx.array(text_mask),
        "codec_mask":    mx.array(codec_mask),
    }
