"""
MiniMind-O — Speech-to-speech model ported from minimind-o to MLX.

Thinker (8-layer transformer LM) + Talker (4-layer acoustic head)
with SenseVoice speech input and Mimi 8-codebook audio output.

Upstream: https://github.com/jingyaogong/minimind-o

Usage:
    from models.minimind_o import MiniMindOmni, OmniConfig
    config = OmniConfig()
    model = MiniMindOmni(config)
"""

from models.minimind_o.config import OmniConfig, MiniMindConfig
from models.minimind_o.model import MiniMindOmni

__all__ = ["OmniConfig", "MiniMindConfig", "MiniMindOmni"]
