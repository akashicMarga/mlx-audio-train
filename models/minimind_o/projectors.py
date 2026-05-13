"""
Multimodal projectors — LN + Linear + GELU + Linear MLPs
that align speech/vision encoder outputs into the Thinker's hidden space.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MMAudioProjector(nn.Module):
    """Projects SenseVoice features (512d) into Thinker hidden size (768d)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(self.ln(x))))


class MMVisionProjector(nn.Module):
    """Projects SigLIP2 patch features (768d) into Thinker hidden size."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(self.ln(x))))
