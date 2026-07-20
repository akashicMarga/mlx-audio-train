"""Audex (Nemotron-Labs-Audex-2B) MLX inference: text generation + TTS.

See generate.py for the API and convert.py for building the MLX checkpoint.
"""

from .generate import load_model, text_generate, tts_generate, AudexModel

__all__ = ["load_model", "text_generate", "tts_generate", "AudexModel"]
