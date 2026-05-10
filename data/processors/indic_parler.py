"""
data/processors/indic_parler.py — Indic Parler TTS processor.

Converts a JSONL sample into training batch tensors for the Parler TTS decoder.

Audio path: pre-encoded .codec.npy sidecar (shape [9, T]) is loaded directly.
            Run scripts/preprocess_bhojpuri_dac.py once to create these files.

Tokenization:
    description → T5 tokenizer (google/flan-t5-large) → desc_ids  [T_desc]
    text        → prompt tokenizer (ai4bharat/indic-parler-tts) → prompt_ids [T_prompt]

Delay pattern (applied here, not in the loss):
    The Parler decoder expects inputs staggered by codebook index.
    For codebook k, the input at frame t is codec[k, max(0, t-k)].
    The total padded sequence length is T + num_codebooks - 1.

Training batch keys:
    desc_ids:       [B, T_desc]  int32      — T5 tokenized description
    desc_mask:      [B, T_desc]  bool       — True = non-padding
    prompt_ids:     [B, T_prompt] int32     — prompt tokenized text
    prompt_mask:    [B, T_prompt] bool
    codes:          [B, 9, T_pad] int32     — delay-pattern codec codes (padded)
    codes_mask:     [B, T_pad]   bool       — True = real frame (not delay-pad)
    codes_lengths:  [B]          int32      — original T before delay-padding
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import mlx.core as mx

_BOS_ID  = 1025   # decoder BOS token (from DecoderConfig)
_EOS_ID  = 1024   # decoder EOS
_PAD_ID  = 0


@dataclass
class IndicParlerProcessorConfig:
    hf_repo:           str  = "ai4bharat/indic-parler-tts"
    desc_tokenizer_id: str  = "google/flan-t5-large"
    sample_rate:       int  = 44100
    num_codebooks:     int  = 9
    max_desc_len:      int  = 128
    max_prompt_len:    int  = 256
    max_frames:        int  = 860    # ~10s at 86Hz
    # For Plan A (speaker adaptation): fixed description (None = use per-sample desc from JSONL)
    fixed_description: Optional[str] = None


class IndicParlerProcessor:
    """
    Tokenizes description + prompt text; loads pre-encoded DAC codec codes.
    Applies delay pattern so the batch is ready for the Parler decoder.
    """

    def __init__(self, config: IndicParlerProcessorConfig):
        self.config = config
        self._desc_tok   = None
        self._prompt_tok = None

    def _load(self):
        if self._desc_tok is not None:
            return
        from transformers import AutoTokenizer
        self._desc_tok   = AutoTokenizer.from_pretrained(self.config.desc_tokenizer_id)
        self._prompt_tok = AutoTokenizer.from_pretrained(self.config.hf_repo)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        sample keys: audio (path), text, description (optional if fixed_description set)
        Returns None if the codec file is missing or the clip is too long.
        """
        self._load()
        cfg = self.config

        # ── Description tokenization ───────────────────────────────────────────
        desc_text = cfg.fixed_description or sample.get("description", "")
        if not desc_text:
            return None

        desc_enc = self._desc_tok(
            desc_text,
            max_length=cfg.max_desc_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        desc_ids  = np.array(desc_enc["input_ids"],      dtype=np.int32)
        desc_mask = np.array(desc_enc["attention_mask"],  dtype=bool)

        # ── Prompt tokenization ────────────────────────────────────────────────
        prompt_text = sample.get("text", "")
        prompt_enc = self._prompt_tok(
            prompt_text,
            max_length=cfg.max_prompt_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_ids  = np.array(prompt_enc["input_ids"],     dtype=np.int32)
        prompt_mask = np.array(prompt_enc["attention_mask"], dtype=bool)

        # ── Load pre-encoded DAC codes ─────────────────────────────────────────
        wav_path    = sample["audio"]
        codec_path  = wav_path + ".codec.npy"
        if not os.path.exists(codec_path):
            return None

        codes = np.load(codec_path).astype(np.int32)  # [9, T]
        if codes.shape[0] != cfg.num_codebooks:
            return None

        T = codes.shape[1]
        if T > cfg.max_frames:
            codes = codes[:, :cfg.max_frames]
            T = cfg.max_frames

        # ── Apply delay pattern ────────────────────────────────────────────────
        # delayed[k, t] = codes[k, t-k] for t >= k, else BOS_ID
        # Padded length: T + num_codebooks - 1
        K  = cfg.num_codebooks
        Tp = T + K - 1
        delayed = np.full((K, Tp), _BOS_ID, dtype=np.int32)
        for k in range(K):
            delayed[k, k : k + T] = codes[k]  # shift codebook k by k

        # Mask: frame t is "real" when ALL codebooks have contributed (t >= K-1)
        # and before the last real frame (t < T + K - 1 when we haven't exhausted real frames)
        codes_mask = np.zeros(Tp, dtype=bool)
        codes_mask[K - 1 : K - 1 + T] = True  # frames that have all codebooks populated

        return {
            "desc_ids":      desc_ids,
            "desc_mask":     desc_mask,
            "prompt_ids":    prompt_ids,
            "prompt_mask":   prompt_mask,
            "codes":         delayed,       # [9, Tp]
            "codes_mask":    codes_mask,    # [Tp]
            "codes_lengths": np.array(T, dtype=np.int32),
        }

    def collate(self, samples: List[Dict[str, np.ndarray]]) -> Dict[str, mx.array]:
        """Pad a list of processed samples into a batch dict of MLX arrays."""
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}

        B = len(samples)

        def _pad_2d(arrs, pad_val=0):
            maxlen = max(a.shape[-1] for a in arrs)
            out = np.full((B, maxlen), pad_val, dtype=arrs[0].dtype)
            for i, a in enumerate(arrs):
                out[i, : a.shape[-1]] = a
            return out

        def _pad_3d(arrs, pad_val=0):
            K, _ = arrs[0].shape
            maxlen = max(a.shape[1] for a in arrs)
            out = np.full((B, K, maxlen), pad_val, dtype=arrs[0].dtype)
            for i, a in enumerate(arrs):
                out[i, :, : a.shape[1]] = a
            return out

        def _pad_bool(arrs):
            maxlen = max(len(a) for a in arrs)
            out = np.zeros((B, maxlen), dtype=bool)
            for i, a in enumerate(arrs):
                out[i, : len(a)] = a
            return out

        desc_ids_pad   = _pad_2d([s["desc_ids"]   for s in samples])
        desc_mask_pad  = _pad_bool([s["desc_mask"]  for s in samples])
        prompt_ids_pad = _pad_2d([s["prompt_ids"] for s in samples])
        prompt_mask_pad = _pad_bool([s["prompt_mask"] for s in samples])
        codes_pad      = _pad_3d([s["codes"]      for s in samples], pad_val=_PAD_ID)
        codes_mask_pad = _pad_bool([s["codes_mask"] for s in samples])
        codes_lengths  = np.array([s["codes_lengths"] for s in samples], dtype=np.int32)

        return {
            "desc_ids":      mx.array(desc_ids_pad),
            "desc_mask":     mx.array(desc_mask_pad),
            "prompt_ids":    mx.array(prompt_ids_pad),
            "prompt_mask":   mx.array(prompt_mask_pad),
            "codes":         mx.array(codes_pad),         # [B, 9, Tp]
            "codes_mask":    mx.array(codes_mask_pad),    # [B, Tp]
            "codes_lengths": mx.array(codes_lengths),     # [B]
        }


class IndicParlerDataset:
    """
    Thin dataset wrapper for BatchIterator.
    Reads a JSONL file and calls IndicParlerProcessor.process() per sample.
    Exposes self.samples (list of raw dicts) so BatchIterator can length-sort.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor: IndicParlerProcessor,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        import json
        import random

        path = Path(jsonl_path)
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "audio" not in obj or "text" not in obj:
                    continue
                if not os.path.isabs(obj["audio"]):
                    obj["audio"] = str(path.parent / obj["audio"])
                records.append(obj)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)
        if max_samples:
            records = records[:max_samples]

        self.samples   = records
        self.processor = processor
        print(f"[dataset] IndicParler: {len(records)} samples from {path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        return self.processor.process(self.samples[idx])
