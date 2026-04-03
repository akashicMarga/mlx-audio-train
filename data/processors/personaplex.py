"""
processors/personaplex.py — PersonaPlex dataset adapter for mlx-audio-train.

PersonaPlex data format (produced by scripts/prepare_personaplex_dataset.py):
    data_dir/
        manifest.json    # [{"id": "0001", "text": "...", "duration": 3.2, "num_frames": 40,
                         #    "audio_file": "wavs/0001.wav", "speaker": "user|assistant|mixed"}]
        tokens/
            0001.npz     # {"user_audio_tokens": (8,T)} or {"assistant_audio_tokens": (8,T)}
                         # or {"audio_tokens": (8,T)} for mixed/dummy
                         # all also contain: "text_tokens": (T_text,) [may be empty]

Token layout in training batch (B, 17, T):
    Row  0:     text tokens
    Rows 1-8:   assistant audio codebook tokens
    Rows 9-16:  user audio codebook tokens

This module is self-contained — no imports from models.personaplex or personaplex_mlx.
Safe for use in prefetch background threads.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sample + Dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaPlexSample:
    """Single PersonaPlex training sample in interleaved token format."""
    input_tokens:  np.ndarray   # (num_streams, T) int32
    target_tokens: np.ndarray   # (num_streams, T) int32
    sample_id:     str
    num_frames:    int          # MIMI frame count — used for sort_by_length


class PersonaPlexDataset:
    """Loads pre-tokenized PersonaPlex data. Compatible with BatchIterator interface.

    Requires:
        data_dir/manifest.json   — sample metadata list
        data_dir/tokens/*.npz    — pre-tokenized MIMI token arrays

    Interface mirrors TTSDataset so BatchIterator works without changes:
        dataset.samples  — list of metadata dicts (from manifest.json)
        dataset[idx]     — returns PersonaPlexSample or None
        len(dataset)     — number of samples
    """

    DEFAULT_AUDIO_DELAYS = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        audio_codebooks: int = 16,
        assistant_codebooks: int = 8,
        audio_pad_token: int = 2048,
        text_pad_token: int = 3,
        text_delay: int = 0,
        audio_delays: Optional[List[int]] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
        split: str = "train",           # "train" | "val" | "all"
        val_fraction: float = 0.05,     # fraction held out for validation
    ):
        self.data_dir           = data_dir
        self.max_seq_len        = max_seq_len
        self.audio_codebooks    = audio_codebooks
        self.assistant_codebooks = assistant_codebooks
        self.audio_pad_token    = audio_pad_token
        self.text_pad_token     = text_pad_token
        self.text_delay         = text_delay
        self.audio_delays       = audio_delays or self.DEFAULT_AUDIO_DELAYS
        self.tokens_dir         = os.path.join(data_dir, "tokens")

        manifest_path = Path(data_dir) / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Always shuffle with fixed seed before splitting so train/val are stable
        rng = random.Random(seed)
        rng.shuffle(manifest)

        # Train / val split
        if split != "all" and val_fraction > 0.0:
            n_val = max(1, int(len(manifest) * val_fraction))
            if split == "val":
                manifest = manifest[:n_val]
            else:  # "train"
                manifest = manifest[n_val:]

        if max_samples:
            manifest = manifest[:max_samples]

        # .samples mirrors TTSDataset.samples for BatchIterator.__len__ and length_key_fn
        self.samples = manifest
        print(f"[personaplex] {len(self.samples)} samples loaded ({split}) from {manifest_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[PersonaPlexSample]:
        entry     = self.samples[idx]
        sample_id = entry["id"]
        npz_path  = os.path.join(self.tokens_dir, f"{sample_id}.npz")

        try:
            data = np.load(npz_path)
        except Exception as e:
            print(f"[personaplex] Failed to load {npz_path}: {e}")
            return None

        # Determine speaker-specific audio tokens
        if "user_audio_tokens" in data and "assistant_audio_tokens" in data:
            user_audio = data["user_audio_tokens"]                          # (8, T)
            asst_audio = data["assistant_audio_tokens"]                     # (8, T)
        elif "user_audio_tokens" in data:
            user_audio = data["user_audio_tokens"]                          # (8, T)
            asst_audio = np.full_like(user_audio, self.audio_pad_token)
        elif "assistant_audio_tokens" in data:
            asst_audio = data["assistant_audio_tokens"]                     # (8, T)
            user_audio = np.full_like(asst_audio, self.audio_pad_token)
        else:
            audio_tokens = data["audio_tokens"]                             # (8, T) mixed/dummy
            user_audio   = audio_tokens
            asst_audio   = audio_tokens

        text_tokens = data.get("text_tokens", data.get("response_text_tokens", np.array([], dtype=np.int32)))
        if not isinstance(text_tokens, np.ndarray):
            text_tokens = np.array(text_tokens, dtype=np.int32)

        return self._build_sample(asst_audio, user_audio, text_tokens, sample_id,
                                  entry.get("num_frames", 0))

    def _build_sample(
        self,
        asst_audio:  np.ndarray,   # (8, T_a)
        user_audio:  np.ndarray,   # (8, T_u)
        text_tokens: np.ndarray,   # (T_text,)
        sample_id:   str,
        num_frames:  int,
    ) -> PersonaPlexSample:
        num_audio_frames = max(asst_audio.shape[1], user_audio.shape[1])
        num_streams      = 1 + self.audio_codebooks   # text row + 16 audio rows
        seq_len          = min(num_audio_frames + max(self.audio_delays) + 1, self.max_seq_len)

        input_seq = np.full((num_streams, seq_len), self.audio_pad_token, dtype=np.int32)
        input_seq[0, :] = self.text_pad_token  # text row default = text pad

        # Fill text tokens into row 0 with optional delay
        text_pos = self.text_delay
        for tok in text_tokens:
            if text_pos >= seq_len:
                break
            input_seq[0, text_pos] = int(tok)
            text_pos += 1

        # Fill assistant audio into rows 1..assistant_codebooks
        for cb in range(self.assistant_codebooks):
            delay = self.audio_delays[cb]
            src   = asst_audio[cb % asst_audio.shape[0]]
            for t in range(min(len(src), seq_len - delay)):
                input_seq[cb + 1, t + delay] = src[t]

        # Fill user audio into rows assistant_codebooks+1..audio_codebooks
        user_offset = self.assistant_codebooks
        for cb in range(self.audio_codebooks - self.assistant_codebooks):
            delay = self.audio_delays[user_offset + cb]
            src   = user_audio[cb % user_audio.shape[0]]
            for t in range(min(len(src), seq_len - delay)):
                input_seq[user_offset + cb + 1, t + delay] = src[t]

        # Target: next-token prediction (shift by 1)
        target_seq = np.full((num_streams, seq_len), self.audio_pad_token, dtype=np.int32)
        target_seq[0, :] = self.text_pad_token
        if seq_len > 1:
            target_seq[:, :-1] = input_seq[:, 1:]

        return PersonaPlexSample(
            input_tokens  = input_seq,
            target_tokens = target_seq,
            sample_id     = sample_id,
            num_frames    = num_frames if num_frames > 0 else num_audio_frames,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_personaplex(samples: List[Optional[PersonaPlexSample]]) -> Dict[str, mx.array]:
    """Pad a list of PersonaPlexSamples to the same sequence length and return mx.arrays.

    Returns:
        {"input_tokens": (B, 17, T), "target_tokens": (B, 17, T)} as mx.arrays
        Empty dict if all samples are None.
    """
    valid = [s for s in samples if s is not None]
    if not valid:
        return {}

    B          = len(valid)
    num_streams = valid[0].input_tokens.shape[0]
    max_len    = max(s.input_tokens.shape[1] for s in valid)

    # Detect pad values from first sample
    audio_pad = int(valid[0].input_tokens[1, 0])  # first audio row initial value
    text_pad  = int(valid[0].input_tokens[0, 0])  # text row initial value

    batch_input  = np.full((B, num_streams, max_len), audio_pad, dtype=np.int32)
    batch_target = np.full((B, num_streams, max_len), audio_pad, dtype=np.int32)
    batch_input[:, 0, :]  = text_pad
    batch_target[:, 0, :] = text_pad

    for i, s in enumerate(valid):
        slen = s.input_tokens.shape[1]
        batch_input[i,  :, :slen] = s.input_tokens
        batch_target[i, :, :slen] = s.target_tokens

    return {
        "input_tokens":  mx.array(batch_input),
        "target_tokens": mx.array(batch_target),
    }
