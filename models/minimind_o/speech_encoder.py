"""
SenseVoice speech encoder wrapper.

Keeps the encoder in PyTorch (via funasr) since there is no MLX port.
Outputs are converted to numpy and then to mx.array for the projector.
Follows the same _ensure_loaded() lazy pattern used by the OmniVox codecs.
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


class SenseVoiceAudioProcessor:
    """Wraps the SenseVoice frontend to produce fbank features."""

    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(
        self,
        wav,
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        return_attention_mask: bool = True,
        **kwargs,
    ):
        import torch

        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))

        mask = (torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0)
        return SimpleNamespace(input_features=fbank, attention_mask=mask)


def load_sensevoice(path: str) -> Tuple[Optional[object], Optional[SenseVoiceAudioProcessor]]:
    """Load SenseVoice-Small encoder and preprocessor.

    Returns (encoder, processor) or (None, None) if path not found / funasr not installed.
    The encoder is a frozen PyTorch module kept on CPU until explicitly moved.
    """
    if not os.path.exists(path):
        warnings.warn(f"[SenseVoice] path not found: {path}. Audio input disabled.")
        return None, None

    try:
        import funasr  # noqa: F401
    except ImportError:
        warnings.warn(
            "[SenseVoice] funasr not installed. Audio input disabled.\n"
            "  Install with: pip install funasr"
        )
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

    encoder = m.model.encoder
    frontend = m.kwargs["frontend"]
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval().float()
    frontend.eval()

    return encoder, SenseVoiceAudioProcessor(frontend)
