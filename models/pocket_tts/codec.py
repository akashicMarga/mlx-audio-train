"""
Mimi codec via the official PyTorch `pocket-tts` package (pip install pocket-tts).

Used offline only: encoding a dataset to latents once, and decoding generated latents
to audio for listening. Training itself runs on precomputed latents in MLX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


def load_wav(path: str | Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Mono float32 at 24 kHz, no loudness normalisation (matches the official loader)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    sr = sf.info(str(path)).samplerate
    frames = -1 if duration is None else int(round(duration * sr))
    wav, sr = sf.read(str(path), start=int(round(start * sr)), frames=frames,
                      dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def best_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class Mimi:
    def __init__(self, config_path: str, device: str | None = None):
        import safetensors.torch
        import torch
        from pocket_tts.models.mimi import build_mimi
        from pocket_tts.utils.config import load_config
        from pocket_tts.utils.utils import download_if_necessary

        self.torch = torch
        self.device = device or best_device()
        config = load_config(config_path)
        mimi = build_mimi(config.mimi)
        weights = getattr(config.mimi, "weights_path", None) or config.weights_path
        state = safetensors.torch.load_file(str(download_if_necessary(str(weights))))
        mimi.load_state_dict({k.removeprefix("mimi."): v for k, v in state.items()
                              if k.startswith("mimi.")}, strict=True)
        self.model = mimi.eval().to(self.device)

    def encode(self, wav: np.ndarray) -> np.ndarray:
        """[T_samples] -> [T_frames, 32] latents at 12.5 Hz."""
        with self.torch.no_grad():
            lat = self.model.encode_to_latent(self.torch.from_numpy(wav)[None, None].to(self.device))
        return lat[0].float().cpu().numpy()

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """[T_frames, 32] -> [T_samples] audio."""
        from pocket_tts.modules.stateful_module import StatefulModule, init_states

        m = self.model
        # Streaming modules find their state by absolute name (pocket-tts stamp_state_names).
        for name, module in m.named_modules():
            if isinstance(module, StatefulModule):
                module._module_absolute_name = name
        ratio = round(m.encoder_frame_rate / m.frame_rate)
        state = init_states(m, 1, (latents.shape[0] + 8) * ratio)
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32))[None].to(self.device)
            return m.decode_from_latent(x, state)[0, 0].float().cpu().numpy()
