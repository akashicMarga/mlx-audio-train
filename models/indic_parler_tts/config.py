from dataclasses import dataclass, field
from typing import List


@dataclass
class T5Config:
    vocab_size: int = 32128
    d_model: int = 1024
    d_kv: int = 64
    d_ff: int = 2816
    num_heads: int = 16
    num_layers: int = 24
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6


@dataclass
class DecoderConfig:
    embed_vocab_size: int = 1089      # actual embed_tokens table size per codebook
    lm_vocab_size: int = 1088         # LM head output vocab per codebook
    prompt_vocab_size: int = 90714    # embed_prompts table size
    hidden_size: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 24
    max_position_embeddings: int = 4096
    num_codebooks: int = 9
    bos_token_id: int = 1025
    eos_token_id: int = 1024


@dataclass
class DACConfig:
    codebook_size: int = 1024
    codebook_dim: int = 8
    hidden_size: int = 1024
    n_codebooks: int = 9
    decoder_hidden_size: int = 1536
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 8, 4, 2])
    sampling_rate: int = 44100


@dataclass
class IndicParlerTTSConfig:
    t5: T5Config = field(default_factory=T5Config)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    dac: DACConfig = field(default_factory=DACConfig)
    repo_id: str = "ai4bharat/indic-parler-tts"
