"""
models/ — Model architectures not yet supported by the upstream mlx-audio library.

Each subdirectory is a self-contained model implementation that can be trained
via scripts/train.py by setting model_type in the YAML config.

Current models:
  - personaplex: NVIDIA PersonaPlex 7B full-duplex conversational speech model
"""
from . import personaplex
