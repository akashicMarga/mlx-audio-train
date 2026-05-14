"""
Audio encoder loaders for MiniMind-O.

Config-driven dispatch: set `config.audio_encoder_type` in your YAML and call
`load_audio_encoder(config)` — the right encoder is loaded automatically.
No code changes needed to swap encoders.

Supported encoders
──────────────────
  sensevoice     SenseVoice-Small (MLX-native via mlx_audio, preferred on Apple Silicon)
  wav2vec2-base  facebook/wav2vec2-base-960h   (PyTorch, 768d, English)
  wav2vec2-large facebook/wav2vec2-large-960h  (PyTorch, 1024d, English)
  mms-300m       facebook/mms-300m             (PyTorch, 1024d, 1000+ languages incl. Indic)
  mms-1b         facebook/mms-1b               (PyTorch, 1280d, 1000+ languages)

All encoders return the same interface:
  encoder(input_tensor, lengths) → (hidden_states, None)
    hidden_states: numpy array (batch, T, hidden_size)

All encoder weights are frozen — only the projector (MMAudioProjector) is trained.

Adding a new encoder
────────────────────
1. Register its hidden size in AUDIO_ENCODER_HIDDEN_SIZES (config.py)
2. Register its default path in AUDIO_ENCODER_DEFAULT_PATHS (config.py)
3. Add a `load_<name>()` function here following the same (encoder, processor) pattern
4. Add a branch in `load_audio_encoder()` below
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared processor wrapper interface
# All processors: wav (numpy float32, 16kHz) → model-ready inputs
# ---------------------------------------------------------------------------

class SenseVoiceAudioProcessor:
    """Wraps the SenseVoice frontend to produce fbank features."""

    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(self, wav, sampling_rate: int = 16000, return_tensors: str = "pt",
                 return_attention_mask: bool = True, **kwargs):
        import torch
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        mask = (torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0)
        return SimpleNamespace(input_features=fbank, attention_mask=mask)


class Wav2Vec2AudioProcessor:
    """
    Thin wrapper around HuggingFace Wav2Vec2Processor.
    Normalises 16kHz float32 waveform → model input values.
    Works for wav2vec2-* and mms-* variants.
    """

    def __init__(self, hf_processor):
        self.processor = hf_processor

    def __call__(self, wav, sampling_rate: int = 16000, return_tensors: str = "pt",
                 return_attention_mask: bool = True, **kwargs):
        return self.processor(
            wav, sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            padding=True,
        )


# ---------------------------------------------------------------------------
# Frozen encoder wrappers — uniform call signature
# All return: (hidden_states np.ndarray (B, T, H), None)
# ---------------------------------------------------------------------------

class FrozenSenseVoiceEncoder:
    """Wraps funasr SenseVoice encoder. Input: fbank (B, T, 560) + lengths."""

    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, fbank, lengths):
        import torch
        with torch.no_grad():
            out, _ = self.encoder(fbank, lengths)
        return out.float().cpu().numpy(), None


class FrozenWhisperEncoder:
    """
    MLX-native Whisper encoder wrapper.

    Input:  raw 16kHz float32 waveform (numpy, variable length)
    Output: (1, T_speech, hidden_size) numpy array
              T_speech = round(audio_duration_s * 50)  [50 output frames/sec]

    Whisper always pads to 30s internally. We extract only the frames
    corresponding to actual speech so the Thinker doesn't see padding tokens.
    """

    OUTPUT_FPS = 50   # Whisper encoder outputs 50 frames/sec after 2× conv stride

    def __init__(self, encoder, n_mels: int = 80):
        self.encoder  = encoder
        self.n_mels   = n_mels

    def __call__(self, wav_np: np.ndarray, lengths=None):
        """
        wav_np: (T_samples,) float32 at 16kHz — single utterance
        Returns: (1, T_frames, hidden) numpy array
        """
        import mlx.core as mx
        from mlx_whisper.audio import log_mel_spectrogram, N_FRAMES

        # Compute duration BEFORE padding so we know how many output frames to keep
        duration_s = len(wav_np) / 16000.0
        n_out = max(1, round(duration_s * self.OUTPUT_FPS))

        # Log-mel spectrogram: (T, n_mels)
        mel = np.array(log_mel_spectrogram(wav_np))
        if mel.shape[1] != self.n_mels:
            mel = mel.T   # some versions return (n_mels, T) — normalise to (T, n_mels)

        # Pad to exactly N_FRAMES (3000) on the time axis
        if mel.shape[0] < N_FRAMES:
            mel = np.pad(mel, ((0, N_FRAMES - mel.shape[0]), (0, 0)))
        else:
            mel = mel[:N_FRAMES]

        mel_mx = mx.array(mel)[None]          # (1, 3000, n_mels)
        enc_out = self.encoder(mel_mx)        # (1, 1500, hidden)
        mx.eval(enc_out)

        # Extract only frames for actual speech, discard silence padding
        out_np = np.array(enc_out[0].tolist(), dtype=np.float32)  # (1500, hidden)
        out_np = out_np[:n_out]               # (T_speech, hidden)
        return out_np[None], None             # (1, T_speech, hidden), None


class FrozenWav2Vec2Encoder:
    """
    Wraps HuggingFace Wav2Vec2Model / MMS model.
    Input: input_values (B, T_samples) raw waveform float32.
    Output: last_hidden_state (B, T_frames, hidden_size).
    """

    def __init__(self, model, attention_mask_key: str = "attention_mask"):
        self.model = model
        self.attention_mask_key = attention_mask_key

    def __call__(self, input_values, lengths=None):
        import torch
        kwargs = {"input_values": input_values}
        if lengths is not None:
            # Build attention mask from lengths
            max_len = input_values.shape[1]
            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
            kwargs["attention_mask"] = mask.long()
        with torch.no_grad():
            out = self.model(**kwargs)
        return out.last_hidden_state.float().cpu().numpy(), None


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_sensevoice(path: str) -> Tuple[Optional[object], Optional[SenseVoiceAudioProcessor]]:
    """
    Load SenseVoice-Small encoder (PyTorch via funasr).
    Returns (FrozenSenseVoiceEncoder, SenseVoiceAudioProcessor) or (None, None).
    """
    if not os.path.exists(path) and not path.startswith(("mlx-community/", "iic/")):
        # Treat as HF repo ID — funasr will download it
        pass

    if not os.path.exists(path):
        warnings.warn(f"[SenseVoice] path not found: {path}. Audio input disabled.")
        return None, None

    try:
        import funasr  # noqa: F401
    except ImportError:
        warnings.warn("[SenseVoice] funasr not installed. Audio input disabled.\n"
                      "  Install with: pip install funasr")
        return None, None

    logging.getLogger().setLevel(logging.ERROR)
    try:
        import transformers.utils.logging as hf_log
        hf_log.set_verbosity_error()
    except Exception:
        pass

    from funasr import AutoModel

    with contextlib.redirect_stdout(io.StringIO()):
        m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")

    raw_encoder = m.model.encoder
    frontend     = m.kwargs["frontend"]
    for p in raw_encoder.parameters():
        p.requires_grad = False
    raw_encoder.eval().float()
    frontend.eval()

    return FrozenSenseVoiceEncoder(raw_encoder), SenseVoiceAudioProcessor(frontend)


def load_whisper(path: str, encoder_type: str = "whisper-small") -> Tuple[Optional[object], None]:
    """
    Load an MLX-native Whisper encoder via mlx_whisper.

    Returns (FrozenWhisperEncoder, None) — Whisper does its own mel preprocessing
    so no separate processor is needed. The encoder wrapper handles everything
    internally (mel spectrogram → pad to 30s → encoder forward → trim to speech length).

    Supported types: whisper-tiny/base/small/medium/large-v3
    """
    try:
        import mlx_whisper
    except ImportError:
        warnings.warn("[Whisper] mlx_whisper not installed. Run: pip install mlx-whisper")
        return None, None

    try:
        print(f"  Loading {encoder_type} from {path} ...", end=" ", flush=True)
        m = mlx_whisper.load_models.load_model(path)
        m.encoder.eval()
        enc = FrozenWhisperEncoder(m.encoder, n_mels=m.dims.n_mels)
        print(f"OK  ({m.dims.n_audio_state}d, {m.dims.n_audio_layer} layers)")
        return enc, None   # no processor — encoder handles mel internally
    except Exception as e:
        warnings.warn(f"[Whisper] Failed to load {path}: {e}")
        return None, None


def load_wav2vec2(path: str, encoder_type: str = "wav2vec2-base") -> Tuple[Optional[object], Optional[object]]:
    """
    Load a wav2vec2 or MMS encoder from HuggingFace.
    Freezes all weights — only the projector will be trained.
    Returns (FrozenWav2Vec2Encoder, Wav2Vec2AudioProcessor) or (None, None).

    Works for:
      encoder_type="wav2vec2-base"  → facebook/wav2vec2-base-960h
      encoder_type="wav2vec2-large" → facebook/wav2vec2-large-960h
      encoder_type="mms-300m"       → facebook/mms-300m
      encoder_type="mms-1b"         → facebook/mms-1b
    """
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoProcessor
        import transformers.utils.logging as hf_log
        hf_log.set_verbosity_error()
    except ImportError:
        warnings.warn("[wav2vec2/MMS] transformers not installed or PyTorch stub active.\n"
                      "  Install real PyTorch to use this encoder.")
        return None, None

    try:
        print(f"  Loading {encoder_type} from {path} ...", end=" ", flush=True)

        # MMS uses AutoProcessor (supports language setting)
        is_mms = encoder_type.startswith("mms")
        processor_cls = AutoProcessor if is_mms else Wav2Vec2Processor
        processor = processor_cls.from_pretrained(path)

        model = Wav2Vec2Model.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        model = model.eval().float()

        print("OK")
        return FrozenWav2Vec2Encoder(model), Wav2Vec2AudioProcessor(processor)

    except Exception as e:
        warnings.warn(f"[{encoder_type}] Failed to load from {path}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Dispatch — the single entry point used by MiniMindOmni.__init__
# ---------------------------------------------------------------------------

def load_audio_encoder(config) -> Tuple[Optional[object], Optional[object]]:
    """
    Load the audio encoder specified in config.audio_encoder_type.

    This is the only function MiniMindOmni calls — swapping encoders requires
    only a config change, not code changes.

    Returns:
        (encoder, processor) where both share the interface used by
        MiniMindOmni.encode_audio_inputs().
    """
    enc_type = config.audio_encoder_type
    enc_path = config.audio_encoder_path

    if enc_type == "sensevoice":
        return load_sensevoice(enc_path)

    elif enc_type in ("whisper-tiny", "whisper-base", "whisper-small",
                      "whisper-medium", "whisper-large-v3"):
        return load_whisper(enc_path, enc_type)

    elif enc_type in ("wav2vec2-base", "wav2vec2-large", "mms-300m", "mms-1b"):
        return load_wav2vec2(enc_path, enc_type)

    else:
        warnings.warn(
            f"[AudioEncoder] Unknown encoder type '{enc_type}'. "
            f"Supported: sensevoice, whisper-{{tiny/base/small/medium/large-v3}}, "
            f"wav2vec2-{{base/large}}, mms-{{300m/1b}}. Audio input disabled."
        )
        return None, None
