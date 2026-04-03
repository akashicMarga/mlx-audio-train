"""
base_dataset.py — Universal JSONL audio dataset.

Expected JSONL format (one JSON object per line):
    {"audio": "path/to/utt.wav", "text": "transcript", "ref_audio": "path/to/ref.wav"}

`ref_audio` is optional — needed only for voice-cloning models (CustomVoice, Chatterbox).
`speaker_id` is optional integer — used for multi-speaker models (CSM).

The dataset is model-agnostic: it returns raw audio arrays + text strings.
Each model's Processor (see processors/) converts these into model-specific tensors.
"""

import json
import os
import queue
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .audio_utils import load_audio, validate_audio, normalize_loudness, trim_silence


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTSSample:
    """One training sample before model-specific processing."""
    audio:      np.ndarray       # float32 mono waveform at target_sr
    text:       str              # transcript
    sample_rate: int
    ref_audio:  Optional[np.ndarray] = None   # reference waveform (voice cloning)
    speaker_id: Optional[int]       = None    # integer speaker ID
    audio_path: str                 = ""      # original path (for debugging)
    duration:   float               = 0.0
    lang_code:  str                 = "auto"  # per-sample language code (e.g. "hi", "ta")


@dataclass
class DatasetConfig:
    jsonl_path:     str
    target_sr:      int   = 24000       # resample all audio to this
    min_duration:   float = 0.5         # seconds — skip shorter
    max_duration:   float = 20.0        # seconds — skip longer
    normalize:      bool  = True        # loudness normalize
    trim:           bool  = True        # trim leading/trailing silence
    shuffle:        bool  = True
    seed:           int   = 42
    max_samples:    Optional[int] = None  # cap dataset size (useful for quick tests)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TTSDataset:
    """
    Model-agnostic TTS dataset.

    Usage:
        dataset = TTSDataset(config)
        sample  = dataset[0]              # → TTSSample
        batch   = dataset.collate([0,1])  # → List[TTSSample]

    To get model-ready tensors, wrap with a Processor:
        processor = Qwen3TTSProcessor(model_path)
        tensors   = processor(sample)
    """

    def __init__(self, config: DatasetConfig, processor: Optional[Callable] = None):
        self.config    = config
        self.processor = processor
        self.samples   = self._load_index()

    # ── indexing ────────────────────────────────────────────────────────────

    def _load_index(self) -> List[Dict]:
        """Parse JSONL → list of metadata dicts (no audio loaded yet)."""
        path = Path(self.config.jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        records = []
        skipped = 0
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[dataset] Line {line_no}: JSON parse error — {e}")
                    skipped += 1
                    continue

                if "audio" not in obj or "text" not in obj:
                    print(f"[dataset] Line {line_no}: missing 'audio' or 'text' key, skipping")
                    skipped += 1
                    continue

                # Resolve relative paths against JSONL location
                base = path.parent
                obj["audio"] = str(base / obj["audio"]) if not os.path.isabs(obj["audio"]) else obj["audio"]
                if "ref_audio" in obj and obj["ref_audio"]:
                    obj["ref_audio"] = str(base / obj["ref_audio"]) if not os.path.isabs(obj["ref_audio"]) else obj["ref_audio"]

                records.append(obj)

        if skipped:
            print(f"[dataset] Skipped {skipped} malformed lines")

        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(records)

        if self.config.max_samples:
            records = records[: self.config.max_samples]

        print(f"[dataset] Loaded index: {len(records)} samples from {path.name}")
        return records

    # ── __len__ / __getitem__ ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[TTSSample]:
        meta = self.samples[idx]
        cfg  = self.config

        # Load main audio
        try:
            audio, sr = load_audio(meta["audio"], target_sr=cfg.target_sr)
        except Exception as e:
            print(f"[dataset] Failed to load {meta['audio']}: {e}")
            return None

        # Validate duration
        ok, reason = validate_audio(audio, sr, cfg.min_duration, cfg.max_duration)
        if not ok:
            return None

        # Preprocess
        if cfg.trim:
            audio = trim_silence(audio, sr)
        if cfg.normalize:
            audio = normalize_loudness(audio)

        # Load ref audio if present
        ref_audio = None
        if meta.get("ref_audio"):
            try:
                ref_audio, _ = load_audio(meta["ref_audio"], target_sr=cfg.target_sr)
                if cfg.normalize:
                    ref_audio = normalize_loudness(ref_audio)
            except Exception as e:
                print(f"[dataset] Failed to load ref_audio {meta['ref_audio']}: {e}")

        sample = TTSSample(
            audio       = audio,
            text        = meta["text"].strip(),
            sample_rate = sr,
            ref_audio   = ref_audio,
            speaker_id  = meta.get("speaker_id"),
            audio_path  = meta["audio"],
            duration    = len(audio) / sr,
            lang_code   = meta.get("lang_code", "auto"),
        )

        # Apply processor if provided (returns model-specific tensors)
        if self.processor is not None:
            return self.processor(sample)

        return sample

    def iter_valid(self):
        """Iterator that skips None samples."""
        for i in range(len(self)):
            sample = self[i]
            if sample is not None:
                yield sample

    # ── stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Compute dataset statistics (loads all audio — use on small splits)."""
        durations = []
        skipped   = 0
        for i in range(len(self)):
            s = self[i]
            if s is None:
                skipped += 1
            else:
                durations.append(s.duration if hasattr(s, "duration") else 0)

        if not durations:
            return {"total": 0, "skipped": skipped}

        return {
            "total":        len(durations),
            "skipped":      skipped,
            "total_hours":  sum(durations) / 3600,
            "mean_dur":     np.mean(durations),
            "min_dur":      np.min(durations),
            "max_dur":      np.max(durations),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────────────────────

def collate_samples(samples: List[TTSSample], pad_id: int = 0) -> Dict[str, Any]:
    """
    Pad a list of TTSSamples into numpy arrays for a batch.
    Each processor may override this with its own collation.
    """
    samples = [s for s in samples if s is not None]
    if not samples:
        return {}

    max_audio = max(len(s.audio) for s in samples)
    sr = samples[0].sample_rate

    audio_batch = np.zeros((len(samples), max_audio), dtype=np.float32)
    audio_lens  = np.array([len(s.audio) for s in samples], dtype=np.int32)
    texts       = [s.text for s in samples]
    speaker_ids = [s.speaker_id for s in samples]

    for i, s in enumerate(samples):
        audio_batch[i, : len(s.audio)] = s.audio

    return {
        "audio":       audio_batch,
        "audio_lens":  audio_lens,
        "texts":       texts,
        "speaker_ids": speaker_ids,
        "sample_rate": sr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Simple batch iterator
# ──────────────────────────────────────────────────────────────────────────────

class BatchIterator:
    """
    Yields processed batches from a TTSDataset.

    Performance options
    -------------------
    sort_by_length : bool
        Sort each epoch's samples by codec length before batching so that
        sequences within a batch are similar length.  This cuts padding
        waste and speeds up the forward pass, especially for variable-
        length TTS data.  Shuffles bucket order to preserve randomness.
    prefetch : int
        Number of batches to prepare ahead of time in a background thread.
        Overlaps CPU data-loading / numpy work with GPU compute.
        0 disables prefetching (default).

    Example:
        loader = BatchIterator(dataset, batch_size=4,
                               sort_by_length=True, prefetch=2)
        for batch in loader:
            loss = train_step(model, batch)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        sort_by_length: bool = False,
        prefetch: int = 2,
        length_key_fn: Optional[Callable] = None,
    ):
        self.dataset        = dataset
        self.batch_size     = batch_size
        self.drop_last      = drop_last
        self.collate_fn     = collate_fn or collate_samples
        self.sort_by_length = sort_by_length
        self.prefetch       = prefetch
        # Optional custom length key: fn(meta_dict) -> int.
        # Default: reads codec .npy file length (Qwen3-TTS style).
        # PersonaPlex: lambda meta: meta.get("num_frames", 0) (in-memory, no disk I/O).
        self.length_key_fn  = length_key_fn

    def _iter_batches(self):
        """Core iteration logic (no prefetching)."""
        indices = list(range(len(self.dataset)))

        if self.sort_by_length:
            # Build a lightweight length index from the JSONL metadata.
            # For Qwen3-TTS the codec .npy file length is the relevant
            # sequence length; fall back to audio duration otherwise.
            def _length_key(idx):
                meta = self.dataset.samples[idx]
                if self.length_key_fn is not None:
                    return self.length_key_fn(meta)
                audio_path = meta.get("audio", "")
                codec_npy  = os.path.splitext(audio_path)[0] + ".codec.npy"
                if os.path.exists(codec_npy):
                    # np.load mmap avoids reading the whole file
                    arr = np.load(codec_npy, mmap_mode="r")
                    return arr.shape[0]
                # Fall back to audio file size as a proxy for duration
                try:
                    return os.path.getsize(audio_path)
                except OSError:
                    return 0

            indices.sort(key=_length_key)

            # Shuffle at the bucket level so training order isn't strictly
            # monotone, while still keeping similar lengths together.
            bucket_size = self.batch_size * 16
            buckets = [indices[i:i + bucket_size]
                       for i in range(0, len(indices), bucket_size)]
            random.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]

        batch = []
        for idx in indices:
            sample = self.dataset[idx]
            if sample is None:
                continue
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __iter__(self):
        if self.prefetch <= 0:
            yield from self._iter_batches()
            return

        # Prefetch batches in a background thread to overlap CPU data
        # loading with GPU computation.
        q        = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def _producer():
            try:
                for batch in self._iter_batches():
                    q.put(batch)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        t.join()

    def __len__(self) -> int:
        n = sum(1 for i in range(len(self.dataset)) if self.dataset.samples[i])
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
