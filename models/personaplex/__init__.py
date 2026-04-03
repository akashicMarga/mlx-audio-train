"""
PersonaPlex — Full-duplex conversational speech model (NVIDIA PersonaPlex 7B).
Ported to MLX for Apple Silicon training and inference.

This module is self-contained: no dependency on the personaplex-mlx repo.
"""
from .lm import (
    Lm,
    LmConfig,
    config_personaplex_7b_v1,
    config_v0_1,
    config1b_202412,
    config1b_202412_16rvq,
    config_helium_1_preview_2b,
)
from .generate import LmGen
from .mimi import MimiConfig, mimi_202407
