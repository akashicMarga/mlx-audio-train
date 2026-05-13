"""
Config dataclasses for MiniMind-O — faithful port of minimind-o OmniConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MiniMindConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    dropout: float = 0.0
    vocab_size: int = 6400
    bos_token_id: int = 1
    eos_token_id: int = 2
    flash_attn: bool = True
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    # MoE
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: Optional[int] = None
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size


@dataclass
class OmniConfig(MiniMindConfig):
    # Talker
    num_talker_hidden_layers: int = 4
    talker_hidden_size: int = 768
    # Audio special tokens / vocab
    audio_ids: List[int] = field(default_factory=lambda: [16])
    audio_special_token: str = "<|audio_pad|>"
    audio_hidden_size: int = 512       # SenseVoice output dim
    audio_vocab_size: int = 2112       # 2048 mimi + 64 specials
    audio_pad_token: int = 2049
    audio_stop_token: int = 2050
    audio_spk_token: int = 2051
    spk_emb_size: int = 192            # CAM++ speaker embedding dim
    # Chain-of-thought end marker: </think>\n\n
    think_end_ids: List[int] = field(default_factory=lambda: [26, 234, 234])
    # Vision (SigLIP2)
    image_ids: List[int] = field(default_factory=lambda: [12])
    image_special_token: str = "<|image_pad|>"
    image_hidden_size: int = 768
    image_token_len: int = 64
    # Bridge layer index (which thinker layer's hidden state feeds the Talker)
    bridge_layer: int = -1             # -1 = auto (num_hidden_layers // 2 - 1)

    def __post_init__(self):
        super().__post_init__()
        if self.bridge_layer == -1:
            self.bridge_layer = self.num_hidden_layers // 2 - 1
