"""
Latent dataset for Pocket TTS distillation.

Manifest (written by scripts/pocket_tts_precompute_latents.py), one JSON per line:

    {"latents": "latents/0001.npy",   # [T, 32] Mimi latents at 12.5 Hz (relative to the manifest)
     "text": "full transcript",
     "words": [{"word": "Hello", "start": 0.12, "end": 0.48}, ...],   # optional alignment
     "voice_latents": "latents/0001_ref.npy"}                         # optional, from ref_audio

Voice prompt per sample, matching the official loader where possible:
  * words present  -> cut at a random word boundary inside the first
                      max_voice_prompt_sec: prompt = audio before the cut,
                      target = audio after it (to the last word + 0.2 s),
                      text = the words after the cut. This is the recipe the
                      released teachers were trained on.
  * voice_latents  -> prompt = that clip (another utterance of the speaker).
  * neither        -> prompt = a random window of the same clip (upstream's fallback;
                      the prompt overlaps the target, fine for distillation since the
                      teacher defines the target, but off-distribution for the teacher).

Cuts are made in latent space (frame = 80 ms), not by re-encoding the audio.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import numpy as np

FRAME_RATE = 12.5
MIN_CUT_SEC = 1.0
TRAIL_SEC = 0.2


def load_tokenizer(path: str) -> Callable[[str], list[int]]:
    from models.pocket_tts.weights import resolve_path

    local = resolve_path(path)
    if local.endswith(".json"):
        import tokenizers

        tok = tokenizers.Tokenizer.from_file(local)
        return lambda text: tok.encode(text).ids
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=local)
    return lambda text: sp.encode(text, out_type=int)


class LatentDataset:
    def __init__(self, manifest: str | Path | list[dict], tokenize: Callable[[str], list[int]],
                 max_duration_sec: float = 30.0, max_voice_prompt_sec: float = 5.0,
                 prompt_trim_max_sec: float = 0.5, final_punct_dropout: float = 0.3,
                 seed: int = 0):
        if isinstance(manifest, list):
            self.entries, self.root = manifest, Path(".")
        else:
            manifest = Path(manifest)
            self.root = manifest.parent
            with open(manifest, encoding="utf-8") as f:
                self.entries = [json.loads(line) for line in f if line.strip()]
        self.tokenize = tokenize
        self.max_frames = int(max_duration_sec * FRAME_RATE)
        self.max_voice_frames = int(max_voice_prompt_sec * FRAME_RATE) if max_voice_prompt_sec > 0 else None
        self.prompt_trim_frames = int(round(prompt_trim_max_sec * FRAME_RATE))
        self.final_punct_dropout = final_punct_dropout
        self.rng = random.Random(seed)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if value not in self._cache:
            self._cache[value] = np.load(self.root / value).astype(np.float32)
        return self._cache[value]

    def _tokens(self, text: str) -> np.ndarray:
        if self.final_punct_dropout > 0 and self.rng.random() < self.final_punct_dropout:
            text = re.sub(r"[.।]\s*$", "", text) or text  # also the Devanagari danda
        return np.array(self.tokenize(text), dtype=np.int32)

    def _trim_prompt(self, prompt: np.ndarray) -> np.ndarray:
        if self.max_voice_frames:
            prompt = prompt[: self.max_voice_frames]
        if self.prompt_trim_frames:
            drop = self.rng.randint(0, self.prompt_trim_frames)
            prompt = prompt[: max(len(prompt) - drop, min(len(prompt), int(FRAME_RATE)))]
        return prompt

    def _choose_cut(self, entry: dict, duration: float):
        words = entry["words"]
        cuts = []
        for i in range(1, len(words)):
            prev, cur = words[i - 1], words[i]
            if prev.get("end") is None or cur.get("start") is None:
                continue
            cut = 0.5 * (prev["end"] + cur["start"])
            if cut >= MIN_CUT_SEC and duration - cut >= MIN_CUT_SEC:
                cuts.append((cut, i))
        window = self.max_voice_frames / FRAME_RATE if self.max_voice_frames else float("inf")
        eligible = [(c, i) for c, i in cuts if words[i - 1]["end"] < window]
        cuts = eligible or cuts[:1]
        if not cuts:
            return None
        return self.rng.choice(cuts)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(target latents [T, C], text tokens, voice prompt latents [Tv, C])."""
        entry = self.entries[index]
        lat = self._load(entry["latents"])
        duration = len(lat) / FRAME_RATE
        words = entry.get("words")
        chosen = self._choose_cut(entry, duration) if words else None
        if chosen is not None:
            cut, i = chosen
            cut_f = int(round(cut * FRAME_RATE))
            ends = [w["end"] for w in words if w.get("end") is not None]
            end_f = len(lat)
            if ends and max(ends) > cut:
                end_f = min(end_f, int(np.ceil((max(ends) + TRAIL_SEC) * FRAME_RATE)))
            target = lat[cut_f:end_f][: self.max_frames]
            text = " ".join(w["word"] for w in words[i:])
            return target, self._tokens(text), self._trim_prompt(lat[:cut_f])
        text = entry["text"]
        target = lat[: self.max_frames]
        if entry.get("voice_latents") is not None:
            return target, self._tokens(text), self._trim_prompt(self._load(entry["voice_latents"]))
        win = self.max_voice_frames or int(3 * FRAME_RATE)
        start = self.rng.randint(0, max(0, len(lat) - win))
        return target, self._tokens(text), self._trim_prompt(lat[start:start + win])


def collate(samples: list[tuple]) -> dict:
    """(latents, tokens, voice[, teacher target]) rows -> padded batch."""
    B = len(samples)
    C = samples[0][0].shape[1]
    # +1 frame so every row also gets its EOS position (first padded frame) in the loss.
    T = max(len(s[0]) for s in samples) + 1
    Tv = max(max(len(s[2]) for s in samples), 1)
    latents = np.zeros((B, T, C), dtype=np.float32)
    voice = np.zeros((B, Tv, samples[0][2].shape[1] if len(samples[0][2]) else C), dtype=np.float32)
    mask = np.zeros((B, T), dtype=bool)
    voice_lens = np.zeros(B, dtype=np.int64)
    for b, (lat, _, pr, *_rest) in enumerate(samples):
        latents[b, : len(lat)] = lat
        mask[b, : len(lat)] = True
        voice[b, : len(pr)] = pr
        voice_lens[b] = len(pr)
    out = {
        "latents": mx.array(latents),          # raw; normalized by the caller
        "mask": mx.array(mask),
        "text_tokens": [s[1] for s in samples],
        "voice": mx.array(voice),
        "voice_lens": voice_lens,
        "frames": int(mask.sum()),
    }
    if len(samples[0]) > 3:  # cached teacher targets [T_i + 1, z] (see teacher_cache.py)
        target = np.zeros((B, T, samples[0][3].shape[1]), dtype=np.float32)
        for b, s in enumerate(samples):
            target[b, : len(s[3])] = s[3]
        out["target"] = mx.array(target)
    return out


def batches(ds, batch_size: int, bucket_batches: int = 5,
            shuffle: bool = True, seed: int = 0) -> Iterator[dict]:
    """Endless length-bucketed batches: pools of bucket_batches*batch_size samples are
    sorted by length so rows in a batch have similar lengths (less padding)."""
    rng = random.Random(seed)
    order = list(range(len(ds)))
    while True:
        if shuffle:
            rng.shuffle(order)
        pool = bucket_batches * batch_size
        for start in range(0, len(order) - batch_size + 1, pool):
            samples = [ds.sample(i) for i in order[start:start + pool]]
            samples = [s for s in samples if len(s[0]) >= 2]
            samples.sort(key=lambda s: len(s[0]))
            chunks = [samples[i:i + batch_size] for i in range(0, len(samples) - batch_size + 1, batch_size)]
            if shuffle:
                rng.shuffle(chunks)
            yield from (collate(c) for c in chunks)
        if not shuffle:
            return
