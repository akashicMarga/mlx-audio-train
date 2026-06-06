"""
processors/lfm_audio.py — LFM 2.5 Audio processor for TTS/ASR fine-tuning.

LFM2.5-Audio is a multimodal speech-to-speech model (also does TTS and ASR).
Uses:
  - Mimi neural codec: 24kHz → 8 codebooks × 2049 tokens (~12.5Hz frames)
  - ConformerEncoder: 16kHz mel spectrogram → encoded audio features
  - HuggingFace tokenizer: text → token IDs

For TTS fine-tuning (training_mode="tts"):
  text  → text_ids          (conditioning input)
  audio → audio_codes [T,8] (generation target, teacher-forced)
  audio_features = None     (no input audio needed)

For ASR fine-tuning (training_mode="asr"):
  audio  → audio_features [T_mel, 128] + audio_codes [T,8]
  text   → text_ids (generation target)

Pre-compute audio codes with scripts/preprocess_lfm_audio.py to avoid
live Mimi encoding during training (faster + avoids OOM).

Batch tensors:
  {
    "text_ids":      [B, T_text]              int32
    "audio_codes":   [B, T_audio, 8]          int32
    "text_lengths":  [B]                      int32
    "audio_lengths": [B]                      int32
    "text_mask":     [B, T_text]              bool
    "audio_mask":    [B, T_audio]             bool
    "audio_features": [B, T_mel, 128]         float32  (ASR mode only)
  }
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlx.core as mx


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LFMAudioProcessorConfig:
    model_id:          str  = "mlx-community/LFM2.5-Audio-1.5B-8bit"
    sample_rate:       int  = 24000    # Mimi codec sample rate
    mel_sample_rate:   int  = 16000    # ConformerEncoder mel sample rate
    num_codebooks:     int  = 8        # Mimi codebooks
    audio_vocab_size:  int  = 2049     # per-codebook vocabulary
    max_audio_frames:  int  = 512      # max audio frames after Mimi encoding (~41s)
    max_text_len:      int  = 256
    training_mode:     str  = "tts"    # "tts" | "asr"
    system_prompt:     str  = "You are a text-to-speech assistant."
    # Pre-loaded processor instance to reuse (avoids reloading)
    processor:         Any  = None


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────

class LFMAudioProcessor:
    """
    Handles tokenization for LFM 2.5 Audio training.

    Audio codec: Mimi (24kHz → 8 codebooks × 2049 tokens)
    Audio encoder: ConformerEncoder mel features (16kHz, 128 bins)
    Text tokenizer: HuggingFace tokenizer from the LFM2 model

    Fast path: loads .codec.npy (8-codebook codes [T,8]) pre-computed by
    preprocess_lfm_audio.py — eliminates live Mimi encoding during training.
    """

    def __init__(self, config: LFMAudioProcessorConfig):
        self.config = config
        self._loaded = False
        self._lfm_processor = config.processor
        self._tokenizer = None

    def _load(self):
        if self._loaded:
            return

        if self._lfm_processor is not None:
            print("[lfm_processor] Using pre-loaded LFM2AudioProcessor")
            self._tokenizer = getattr(self._lfm_processor, "tokenizer", None)
            self._loaded = True
            return

        try:
            from mlx_audio.sts.models.lfm_audio import LFM2AudioProcessor
            print(f"[lfm_processor] Loading LFM2AudioProcessor: {self.config.model_id}")
            self._lfm_processor = LFM2AudioProcessor.from_pretrained(self.config.model_id)
            self._tokenizer = getattr(self._lfm_processor, "tokenizer", None)
        except Exception as e:
            print(f"[lfm_processor] Warning: could not load LFM2AudioProcessor: {e}")
            self._lfm_processor = None
            # Fallback: load tokenizer directly
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id, trust_remote_code=True
                )
            except Exception as e2:
                print(f"[lfm_processor] Warning: could not load tokenizer: {e2}")
                self._tokenizer = None

        self._loaded = True

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _calc_audio_frames(mel_frames: int) -> int:
        """ConformerEncoder output length: 3 stride-2 conv layers (kernel=3, padding=1)."""
        def _conv(n): return (n + 2 * 1 - 3) // 2 + 1
        return _conv(_conv(_conv(mel_frames)))

    # ── Audio encoding ────────────────────────────────────────────────────

    def encode_audio_codes(self, audio: np.ndarray, audio_path: str = None) -> np.ndarray:
        """
        Audio waveform (24kHz) → Mimi codec codes [T_audio, 8].

        Fast path: loads pre-computed <audio_stem>.codec.npy if present.
        Slow path: runs Mimi encoder live (slow; use preprocess_lfm_audio.py).

        Returns int32 array of shape [T_audio, 8].
        """
        # Fast path: pre-computed .codec.npy with shape [T, 8]
        if audio_path:
            codec_npy = Path(audio_path).with_suffix(".codec.npy")
            if codec_npy.exists():
                codes = np.load(str(codec_npy))
                if codes.ndim == 2 and codes.shape[1] == self.config.num_codebooks:
                    return codes.astype(np.int32)

        # Slow path: live Mimi encoding
        self._load()
        if self._lfm_processor is None:
            dur = len(audio) / self.config.sample_rate
            n = max(1, int(dur * 12.5))
            return np.zeros((n, self.config.num_codebooks), dtype=np.int32)

        try:
            audio_f32 = audio.astype(np.float32)
            # tokenize_audio expects [B, 1, T] at 24kHz
            audio_mx = mx.array(audio_f32)[None, None, :]  # [1, 1, T]
            codes = self._lfm_processor.tokenize_audio(audio_mx)  # [1, 8, T]
            mx.eval(codes)
            # Transpose to [T, 8]
            return np.array(codes[0]).T.astype(np.int32)
        except Exception as e:
            print(f"[lfm_processor] Mimi encoding failed: {e}")
            dur = len(audio) / self.config.sample_rate
            n = max(1, int(dur * 12.5))
            return np.zeros((n, self.config.num_codebooks), dtype=np.int32)

    def encode_audio_features(self, audio: np.ndarray, audio_path: str = None) -> Optional[np.ndarray]:
        """
        Audio waveform (16kHz) → mel spectrogram [T_mel, 128] for ConformerEncoder.
        Used in ASR mode; None is returned in TTS mode.
        Only called when training_mode="asr".
        """
        if self.config.training_mode != "asr":
            return None

        if audio_path:
            mel_npy = Path(audio_path).with_suffix(".mel.npy")
            if mel_npy.exists():
                return np.load(str(mel_npy)).astype(np.float32)

        self._load()
        if self._lfm_processor is None:
            dur = len(audio) / self.config.mel_sample_rate
            n = max(1, int(dur * 100))  # ~100 frames/s
            return np.zeros((n, 128), dtype=np.float32)

        try:
            audio_f32 = audio.astype(np.float32)
            audio_mx  = mx.array(audio_f32)
            # Pass the actual sample rate so preprocess_audio can resample to 16kHz.
            mel = self._lfm_processor.preprocess_audio(audio_mx, sample_rate=self.config.sample_rate)
            mx.eval(mel)
            return np.array(mel, dtype=np.float32)
        except Exception as e:
            print(f"[lfm_processor] mel extraction failed: {e}")
            return None

    # ── Text encoding ─────────────────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """TTS mode: text → token IDs with chat template. Returns int32 [T_text]."""
        self._load()
        if self._tokenizer is None:
            return np.array([1, 2, 3], dtype=np.int32)
        try:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user",   "content": text},
            ]
            ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return np.array(ids, dtype=np.int32)[:self.config.max_text_len]
        except Exception:
            pass
        try:
            ids = self._tokenizer.encode(text, add_special_tokens=True)
            return np.array(ids, dtype=np.int32)[:self.config.max_text_len]
        except Exception:
            return np.array([1, 2, 3], dtype=np.int32)

    def build_asr_sequence(self, text: str, mel_frames: int):
        """
        Build text_tokens + modalities for ASR training.

        Audio is placed in the user turn (BEFORE assistant tokens) so the
        model can attend to it when predicting the function call.  This
        matches the inference-time layout produced by ChatState.add_audio().

        Returns (text_tokens [N_text], modalities [N_total]) as int32 arrays,
        where N_total = N_text + N_audio_frames.
        """
        self._load()
        tok = self._tokenizer
        if tok is None:
            return np.array([1, 2, 3], dtype=np.int32), None

        TEXT     = 1  # LFMModality.TEXT
        AUDIO_IN = 2  # LFMModality.AUDIO_IN

        text_tokens: list = []
        modalities:  list = []

        def _add_text(s):
            ids = tok.encode(s, add_special_tokens=False)
            text_tokens.extend(ids)
            modalities.extend([TEXT] * len(ids))

        # BOS
        bos_id = getattr(tok, "bos_token_id", 1)
        if bos_id is not None:
            text_tokens.append(bos_id)
            modalities.append(TEXT)

        # System turn
        _add_text(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n")

        # User turn header (no content text — audio fills this space)
        _add_text("<|im_start|>user\n")

        # AUDIO_IN frames (no text tokens for these positions)
        n_audio = self._calc_audio_frames(mel_frames)
        modalities.extend([AUDIO_IN] * n_audio)

        # User turn footer
        _add_text("<|im_end|>\n")

        # Assistant turn with function call
        _add_text(f"<|im_start|>assistant\n{text}<|im_end|>\n")

        return (
            np.array(text_tokens, dtype=np.int32),
            np.array(modalities,  dtype=np.int32),
        )

    # ── Main processing call ──────────────────────────────────────────────

    def __call__(self, sample) -> Optional[Dict]:
        """Process a TTSSample into a training-ready dict."""
        from ..base_dataset import TTSSample
        if not isinstance(sample, TTSSample):
            return sample

        try:
            if self.config.training_mode == "asr":
                # Audio codes are not used in the ASR loss — skip expensive Mimi encoding
                dur = len(sample.audio) / self.config.sample_rate
                n   = max(1, int(dur * 12.5))
                audio_codes = np.zeros((n, self.config.num_codebooks), dtype=np.int32)
            else:
                audio_codes = self.encode_audio_codes(sample.audio, audio_path=sample.audio_path)

            if len(audio_codes) > self.config.max_audio_frames:
                audio_codes = audio_codes[:self.config.max_audio_frames]

            if self.config.training_mode == "asr":
                # Compute mel first so we know N_audio_frames for the modalities array
                mel = self.encode_audio_features(sample.audio, audio_path=sample.audio_path)
                mel_frames = mel.shape[0] if mel is not None else 0
                text_ids, modalities = self.build_asr_sequence(sample.text, mel_frames)
            else:
                text_ids   = self.encode_text(sample.text)
                mel        = None
                modalities = None

            result = {
                "text_ids":      text_ids,
                "audio_codes":   audio_codes,
                "text_length":   len(text_ids),
                "audio_length":  len(audio_codes),
                "text":          sample.text,
                "audio_path":    sample.audio_path,
            }

            if mel is not None:
                result["audio_features"]        = mel
                result["audio_features_length"] = mel.shape[0]

            if modalities is not None:
                result["modalities"] = modalities

            return result

        except Exception as e:
            print(f"[lfm_processor] Failed to process {sample.audio_path}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Collation
# ─────────────────────────────────────────────────────────────────────────────

def collate_lfm_audio(
    samples:       List[Dict],
    pad_text_id:   int = 0,
    pad_audio_id:  int = 0,
    num_codebooks: int = 8,
) -> Dict[str, mx.array]:
    """
    Pad a list of processed LFM samples into a single MLX batch dict.

    Returns:
        text_ids:       [B, T_text_max]           int32
        audio_codes:    [B, T_audio_max, 8]       int32
        text_lengths:   [B]                       int32
        audio_lengths:  [B]                       int32
        text_mask:      [B, T_text_max]           bool
        audio_mask:     [B, T_audio_max]          bool
    """
    samples = [s for s in samples if s is not None]
    if not samples:
        return {}

    B       = len(samples)
    T_text  = max(s["text_length"]  for s in samples)
    T_audio = max(s["audio_length"] for s in samples)

    # Infer actual codebook count from data — Mimi may output more codebooks (e.g. 32)
    # than the model's audio head depth (e.g. 8). Store all; the loss slices as needed.
    actual_codebooks = samples[0]["audio_codes"].shape[1] if samples[0].get("audio_codes") is not None else num_codebooks

    text_ids    = np.full((B, T_text),                    pad_text_id,  dtype=np.int32)
    audio_codes = np.full((B, T_audio, actual_codebooks), pad_audio_id, dtype=np.int32)

    for i, s in enumerate(samples):
        tl = s["text_length"]
        al = s["audio_length"]
        text_ids[i, :tl]       = s["text_ids"][:tl]
        audio_codes[i, :al, :] = s["audio_codes"][:al]

    text_lengths  = np.array([s["text_length"]  for s in samples], dtype=np.int32)
    audio_lengths = np.array([s["audio_length"] for s in samples], dtype=np.int32)

    text_mask  = (np.arange(T_text)[None,  :] < text_lengths[:,  None])
    audio_mask = (np.arange(T_audio)[None, :] < audio_lengths[:, None])

    batch = {
        "text_ids":      mx.array(text_ids),
        "audio_codes":   mx.array(audio_codes),
        "text_lengths":  mx.array(text_lengths),
        "audio_lengths": mx.array(audio_lengths),
        "text_mask":     mx.array(text_mask),
        "audio_mask":    mx.array(audio_mask),
    }

    # ASR mode: include padded mel features
    if any("audio_features" in s for s in samples):
        mels = [s.get("audio_features") for s in samples]
        if all(m is not None for m in mels):
            T_mel     = max(m.shape[0] for m in mels)
            n_mel_bin = mels[0].shape[1]
            mel_arr   = np.zeros((B, T_mel, n_mel_bin), dtype=np.float32)
            mel_lens  = np.array([m.shape[0] for m in mels], dtype=np.int32)
            for i, m in enumerate(mels):
                mel_arr[i, :m.shape[0]] = m
            batch["audio_features"]        = mx.array(mel_arr)
            batch["audio_feature_lengths"] = mx.array(mel_lens)

    # ASR mode: include modalities (TEXT=1 / AUDIO_IN=2 per sequence position)
    if any("modalities" in s for s in samples):
        mods_list = [s["modalities"] for s in samples if s.get("modalities") is not None]
        if len(mods_list) == B:
            T_mod = max(len(m) for m in mods_list)
            # Pad with TEXT (1); padded positions are outside text_mask so ignored in loss
            mods_arr = np.ones((B, T_mod), dtype=np.int32)
            for i, m in enumerate(mods_list):
                mods_arr[i, :len(m)] = m
            batch["modalities"] = mx.array(mods_arr)

    return batch
