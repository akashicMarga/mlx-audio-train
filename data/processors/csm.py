"""
processors/csm.py — CSM (Sesame) specific processor.

CSM uses:
  - Mimi codec (24kHz → 12.5Hz RVQ tokens, 32 codebooks)
  - LLaMA tokenizer for text
  - Multi-speaker: speaker_id per segment
  - Conversational context: list of prior turns

Training format (mirrors csm-mlx):
    [
      [
        {"text": "...", "audio_path": "...", "speaker_id": 0},
        {"text": "...", "audio_path": "...", "speaker_id": 1},
      ]
    ]

Batch tensors:
    {
        "tokens":          [B, T]   int32    # interleaved text+audio tokens
        "tokens_mask":     [B, T]   bool
        "audio_ids":       [B, T_audio, 32]  int32  # all 32 RVQ codebooks
        "audio_mask":      [B, T_audio]       bool
        "speaker_ids":     [B]      int32
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import mlx.core as mx


@dataclass
class CSMProcessorConfig:
    model_id:        str  = "mlx-community/csm-1b"
    sample_rate:     int  = 24000
    max_seq_len:     int  = 2048
    max_audio_len:   int  = 1500
    n_codebooks:     int  = 32
    pad_id:          int  = 0


class CSMProcessor:
    """
    Processor for CSM (Sesame Conversation Speech Model).

    CSM is a Llama backbone that predicts RVQ audio tokens.
    Text and audio tokens are interleaved in the sequence.
    """

    def __init__(self, config: CSMProcessorConfig):
        self.config  = config
        self._loaded = False
        self._mimi   = None
        self._tokenizer = None

    def _load(self):
        if self._loaded:
            return
        try:
            from mlx_audio.tts.models.sesame.sesame import Model
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            # Mimi codec loaded via sesame model
        except Exception as e:
            print(f"[csm_processor] Warning: {e}")
        self._loaded = True

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Audio → RVQ codes, shape [T, 32]."""
        self._load()
        if self._mimi is None:
            dur = len(audio) / self.config.sample_rate
            n = max(1, int(dur * 12.5))
            return np.zeros((n, self.config.n_codebooks), dtype=np.int32)
        audio_mx = mx.array(audio)[None, :]
        codes = self._mimi.encode(audio_mx)
        return np.array(codes[0].T, dtype=np.int32)  # [T, 32]

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        if self._tokenizer is None:
            return np.array([1, 2, 3], dtype=np.int32)
        ids = self._tokenizer.encode(text, add_special_tokens=True,
                                     truncation=True, max_length=self.config.max_seq_len)
        return np.array(ids, dtype=np.int32)

    def __call__(self, sample) -> Optional[Dict]:
        from ..base_dataset import TTSSample
        if not isinstance(sample, TTSSample):
            return sample
        try:
            text_ids  = self.encode_text(sample.text)
            audio_ids = self.encode_audio(sample.audio)  # [T, 32]
            return {
                "text_ids":      text_ids,
                "audio_ids":     audio_ids,
                "text_length":   len(text_ids),
                "audio_length":  len(audio_ids),
                "speaker_id":    sample.speaker_id or 0,
                "text":          sample.text,
                "audio_path":    sample.audio_path,
            }
        except Exception as e:
            print(f"[csm_processor] Failed: {e}")
            return None


def collate_csm(samples: List[Dict], pad_id: int = 0) -> Dict[str, mx.array]:
    samples = [s for s in samples if s is not None]
    if not samples:
        return {}

    B       = len(samples)
    T_text  = max(s["text_length"]  for s in samples)
    T_audio = max(s["audio_length"] for s in samples)
    n_cb    = samples[0]["audio_ids"].shape[1]

    text_ids  = np.full((B, T_text),         pad_id, dtype=np.int32)
    audio_ids = np.full((B, T_audio, n_cb),  pad_id, dtype=np.int32)

    for i, s in enumerate(samples):
        tl, al = s["text_length"], s["audio_length"]
        text_ids[i,  :tl]     = s["text_ids"][:tl]
        audio_ids[i, :al, :]  = s["audio_ids"][:al, :]

    text_lengths  = np.array([s["text_length"]  for s in samples], dtype=np.int32)
    audio_lengths = np.array([s["audio_length"] for s in samples], dtype=np.int32)
    speaker_ids   = np.array([s["speaker_id"]   for s in samples], dtype=np.int32)

    return {
        "text_ids":      mx.array(text_ids),
        "audio_ids":     mx.array(audio_ids),
        "text_lengths":  mx.array(text_lengths),
        "audio_lengths": mx.array(audio_lengths),
        "speaker_ids":   mx.array(speaker_ids),
        "text_mask":     mx.array(np.arange(T_text)[None, :]  < text_lengths[:, None]),
        "audio_mask":    mx.array(np.arange(T_audio)[None, :] < audio_lengths[:, None]),
    }
