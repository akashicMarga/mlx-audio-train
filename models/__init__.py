"""
models/ — Model architectures not yet supported by the upstream mlx-audio library.

Each subdirectory is a self-contained model implementation that can be trained
via scripts/train.py by setting model_type in the YAML config.

Current models:
  - personaplex: NVIDIA PersonaPlex 7B full-duplex conversational speech model
  - indic_parler_tts: ai4bharat/indic-parler-tts — pure MLX inference stack
"""
from . import personaplex
# indic_parler_tts imported lazily in train.py to avoid torch dependency at startup
