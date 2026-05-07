"""
Top-level API for Indic Parler TTS inference.

Usage:
    from models.indic_parler_tts import load_model, generate

    model, tokenizers = load_model()
    audio = generate(
        model, tokenizers,
        description="A female speaker delivers clear speech at a moderate pace.",
        text="नमस्ते, आप कैसे हैं?",
    )
    import soundfile as sf
    sf.write("out.wav", audio, 44100)
"""

import numpy as np

import mlx.core as mx

_SR = 44100


def _trim_silence(
    audio: np.ndarray,
    frame_ms: int = 30,
    energy_threshold: float = 5e-4,
    pad_ms: int = 300,
) -> np.ndarray:
    """Remove trailing silence below energy_threshold, keeping pad_ms of padding."""
    if len(audio) == 0:
        return audio
    frame = int(_SR * frame_ms / 1000)
    pad   = int(_SR * pad_ms  / 1000)
    n_frames = len(audio) // frame
    last_active = 0
    for i in range(n_frames):
        chunk = audio[i * frame : (i + 1) * frame]
        if np.sqrt(np.mean(chunk ** 2)) > energy_threshold:
            last_active = i
    end = min(len(audio), (last_active + 1) * frame + pad)
    return audio[:end]


def load_model(
    repo_id: str = "ai4bharat/indic-parler-tts",
    *,
    strict_load: bool = True,
    audit_load: bool = True,
):
    """
    Downloads and loads the model + both tokenizers.

    Returns:
        model      : IndicParlerTTS (pure MLX, ready for inference)
        tokenizers : dict with keys 'description' and 'prompt'
    """
    from transformers import AutoTokenizer, logging as hf_logging
    from .model import IndicParlerTTS

    print(f"[indic-parler-tts] Loading model from {repo_id} ...")
    model = IndicParlerTTS.from_pretrained(
        repo_id,
        strict=strict_load,
        audit=audit_load,
    )
    model.eval()

    # AutoTokenizer reads the Parler config and emits a harmless model_type
    # warning for the prompt tokenizer; keep script output focused.
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        # Description tokenizer: T5-based (Flan-T5-Large), vocab=32128
        desc_tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
        # Prompt tokenizer: custom multilingual SentencePiece, vocab=90714
        prompt_tok = AutoTokenizer.from_pretrained(repo_id)
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    tokenizers = {"description": desc_tok, "prompt": prompt_tok}
    print("[indic-parler-tts] Ready.")
    return model, tokenizers


def generate(
    model,
    tokenizers: dict,
    description: str,
    text: str,
    max_audio_length_s: float = 10.0,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int | None = None,
) -> np.ndarray:
    """
    Synthesise speech.

    Args:
        model             : loaded IndicParlerTTS
        tokenizers        : dict from load_model()
        description       : style prompt, e.g. "A female speaker delivers clear speech."
        text              : text to synthesise in any of the 18 supported languages
        max_audio_length_s: maximum output duration in seconds
        temperature       : sampling temperature (0 = greedy)
        top_k             : top-k sampling (0 = disabled)
        seed              : optional NumPy RNG seed for reproducible sampling

    Returns:
        float32 numpy array of audio samples at 44100 Hz
    """
    desc_enc = tokenizers["description"](
        description,
        return_tensors="np",
        padding=True,
    )
    prompt_ids = tokenizers["prompt"](
        text,
        return_tensors="np",
        padding=True,
    ).input_ids

    desc_mx = mx.array(desc_enc.input_ids, dtype=mx.int32)
    prompt_mx = mx.array(prompt_ids, dtype=mx.int32)

    # Build encoder attention mask: [1, 1, 1, T_enc], additive (-1e9 = ignore pad)
    enc_attn = desc_enc.attention_mask          # [1, T_enc], 1=valid 0=pad
    encoder_mask = mx.array(
        (1 - enc_attn[:, None, None, :]) * -1e9, dtype=mx.float32
    )

    audio = model.generate(
        desc_mx,
        prompt_mx,
        max_audio_length_s=max_audio_length_s,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        encoder_mask=encoder_mask,
    )
    return _trim_silence(audio)
