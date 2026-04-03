"""
audio_logging.py — TensorBoard audio sample generation during training eval.

During training, after each eval step, generates a short audio sample and logs
it to TensorBoard so you can listen to model quality evolving over training.

PersonaPlex strategy (fast — single forward pass):
    1. Run teacher-forced forward on a stored validation sample
    2. Take argmax of depformer output logits → predicted audio tokens
    3. Decode ground-truth tokens AND predicted tokens with rustymimi
    4. Log both to TensorBoard → compare original vs model prediction

Qwen3-TTS strategy (autoregressive generation):
    1. Run model.generate() on a fixed test text
    2. Log the generated waveform to TensorBoard

Usage:
    fn = make_personaplex_audio_eval_fn(val_dataset, eval_audio_cfg, mimi_weight)
    fn = make_qwen3_tts_audio_eval_fn(model, eval_audio_cfg)

    # In trainer (after each eval):
    if fn and tb_writer:
        fn(model, global_step, tb_writer)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_audio(wav: np.ndarray) -> np.ndarray:
    """Normalize waveform to [-1, 1] float32. Silently returns zeros on empty input."""
    wav = np.asarray(wav, dtype=np.float32).ravel()
    peak = np.abs(wav).max()
    if peak > 1e-8:
        wav = wav / peak
    return wav


def _decode_tokens_rustymimi(
    tokens_np:      np.ndarray,   # (nq, T) int32
    mimi_weight:    str,
    num_codebooks:  int = 8,
    audio_pad:      int = 2048,
    max_frames:     int = 192,    # ~15 seconds at 12.5 Hz
) -> np.ndarray:
    """Decode MIMI tokens to float32 PCM waveform using rustymimi.

    Calls decode_step frame-by-frame, skipping all-padding frames.
    Returns 1D float32 array, shape (samples,).
    """
    try:
        import rustymimi
    except ImportError:
        raise RuntimeError("rustymimi not installed — cannot decode MIMI tokens for audio eval")

    tokenizer = rustymimi.Tokenizer(mimi_weight, num_codebooks=num_codebooks)
    nq, T = tokens_np.shape
    pcm_chunks: list[np.ndarray] = []

    for t in range(min(T, max_frames)):
        frame = tokens_np[:, t]                      # (nq,)
        if (frame >= audio_pad).all():
            continue                                  # skip padding frames
        # Clip to valid range [0, codebook_size-1=2047]
        frame = np.clip(frame, 0, audio_pad - 1)
        # rustymimi.decode_step expects (batch, codebooks, frames)
        frame_input = frame.reshape(1, nq, 1).astype(np.uint32)
        pcm = tokenizer.decode_step(frame_input)     # (1, 1, 1920)
        if pcm is not None:
            pcm_chunks.append(pcm)

    if not pcm_chunks:
        return np.zeros(1920, dtype=np.float32)

    return np.concatenate(pcm_chunks, axis=-1)[0, 0].astype(np.float32)  # (samples,)


def _split_audio_streams(
    input_tokens_np: np.ndarray,   # (17, T)
) -> tuple[np.ndarray, np.ndarray]:
    """Return assistant and user 8-codebook streams from a PersonaPlex sample."""
    asst_tokens = input_tokens_np[1:9, :]    # (8, T)
    user_tokens = input_tokens_np[9:17, :]   # (8, T)
    return asst_tokens, user_tokens


# ─────────────────────────────────────────────────────────────────────────────
# PersonaPlex audio eval
# ─────────────────────────────────────────────────────────────────────────────

def make_personaplex_audio_eval_fn(
    val_dataset,
    eval_audio_cfg: Dict,
    mimi_weight:    str,
    base_model=None,
) -> Optional[Callable]:
    """Returns a callable(model, global_step, tb_writer) for PersonaPlex eval audio.

    At each eval step:
    1. Runs teacher-forced forward on a few stored val samples (fast, single pass)
    2. Takes argmax of depformer logits → predicted audio tokens
    3. Decodes ground-truth and predicted tokens with rustymimi
    4. Logs both to TensorBoard under "eval_audio/sample_N_gt" and "eval_audio/sample_N_pred"

    Args:
        val_dataset:    PersonaPlexDataset instance (used to pre-load a few samples)
        eval_audio_cfg: dict from config["eval_audio"]
        mimi_weight:    path to MIMI tokenizer weights (.safetensors)
    """
    if not eval_audio_cfg.get("enabled", True):
        return None

    max_samples = eval_audio_cfg.get("max_samples", 2)
    audio_pad   = eval_audio_cfg.get("audio_pad_token", 2048)
    sample_rate = eval_audio_cfg.get("sample_rate", 24000)
    max_frames  = eval_audio_cfg.get("max_decode_frames", 128)  # ~10s
    log_user_gt = eval_audio_cfg.get("log_user_gt", True)
    log_assistant_gt = eval_audio_cfg.get("log_assistant_gt", True)
    log_assistant_pred = eval_audio_cfg.get("log_assistant_pred", True)
    log_base_assistant_pred = eval_audio_cfg.get("log_base_assistant_pred", True)
    log_reference_once = eval_audio_cfg.get("log_reference_once", True)

    # Pre-load a few val samples at init time (not at eval time) so we always
    # evaluate the same samples — makes progress easier to hear.
    eval_samples = []
    for i in range(len(val_dataset)):
        s = val_dataset[i]
        if s is not None:
            eval_samples.append(s)
        if len(eval_samples) >= max_samples:
            break

    if not eval_samples:
        print("[audio_logging] No val samples found for PersonaPlex audio eval — skipping")
        return None

    print(f"[audio_logging] PersonaPlex audio eval: {len(eval_samples)} samples, "
          f"mimi_weight={mimi_weight}")

    def _predict_assistant_rows(eval_model, input_np: np.ndarray) -> np.ndarray:
        import mlx.core as mx

        input_batch = mx.array(input_np[None].astype(np.int32))  # (1, 17, T)

        for c in eval_model.transformer_cache:
            c.reset()
        for c in eval_model.depformer_cache:
            c.reset()

        transformer_out, _ = eval_model.forward_codes(input_batch)
        num_slices = eval_model.cfg.depformer.num_slices
        dep_input = input_batch[:, :num_slices, :]
        logits_list = eval_model.depformer(transformer_out, dep_input, eval_model.depformer_cache)
        mx.eval(*logits_list)
        return np.stack(
            [np.array(mx.argmax(logits_list[cb][0], axis=-1)) for cb in range(8)],
            axis=0,
        )

    reference_logged = False

    def audio_eval_fn(model, global_step: int, tb_writer, reference_only: bool = False) -> None:
        import mlx.core as mx
        nonlocal reference_logged

        should_log_reference = reference_only or not reference_logged or not log_reference_once
        should_log_adapted = not reference_only and log_assistant_pred

        for i, sample in enumerate(eval_samples):
            try:
                input_np = sample.input_tokens                    # (17, T)
                asst_rows, user_rows = _split_audio_streams(input_np)

                if should_log_reference and log_user_gt:
                    user_wav = _decode_tokens_rustymimi(
                        user_rows, mimi_weight, num_codebooks=8,
                        audio_pad=audio_pad, max_frames=max_frames,
                    )
                    user_wav = _normalize_audio(user_wav)
                    tb_writer.add_audio(
                        f"eval_audio/sample_{i}_A_gt",
                        user_wav, 0 if log_reference_once else global_step, sample_rate=sample_rate,
                    )

                if should_log_reference and log_assistant_gt:
                    asst_gt_wav = _decode_tokens_rustymimi(
                        asst_rows, mimi_weight, num_codebooks=8,
                        audio_pad=audio_pad, max_frames=max_frames,
                    )
                    asst_gt_wav = _normalize_audio(asst_gt_wav)
                    tb_writer.add_audio(
                        f"eval_audio/sample_{i}_B_gt",
                        asst_gt_wav, 0 if log_reference_once else global_step, sample_rate=sample_rate,
                    )

                if should_log_adapted:
                    pred_rows = _predict_assistant_rows(model, input_np)
                    pred_wav = _decode_tokens_rustymimi(
                        pred_rows, mimi_weight, num_codebooks=8,
                        audio_pad=audio_pad, max_frames=max_frames,
                    )
                    pred_wav = _normalize_audio(pred_wav)
                    tb_writer.add_audio(
                        f"eval_audio/sample_{i}_B_pred",
                        pred_wav, global_step, sample_rate=sample_rate,
                    )

                if should_log_reference and log_base_assistant_pred and base_model is not None:
                    base_pred_rows = _predict_assistant_rows(base_model, input_np)
                    base_pred_wav = _decode_tokens_rustymimi(
                        base_pred_rows, mimi_weight, num_codebooks=8,
                        audio_pad=audio_pad, max_frames=max_frames,
                    )
                    base_pred_wav = _normalize_audio(base_pred_wav)
                    tb_writer.add_audio(
                        f"eval_audio/sample_{i}_B_pred_base",
                        base_pred_wav, 0 if log_reference_once else global_step, sample_rate=sample_rate,
                    )

            except Exception as e:
                print(f"[audio_logging] PersonaPlex sample {i} failed at step {global_step}: {e}")

        if should_log_reference and log_reference_once:
            reference_logged = True

    return audio_eval_fn


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-TTS audio eval
# ─────────────────────────────────────────────────────────────────────────────

def make_qwen3_tts_audio_eval_fn(
    model,
    eval_audio_cfg: Dict,
) -> Optional[Callable]:
    """Returns a callable(model, global_step, tb_writer) for Qwen3-TTS eval audio.

    At each eval step, runs autoregressive generation on each test_text and
    logs the waveform to TensorBoard under "eval_audio/sample_N".

    Args:
        model:          Loaded Qwen3-TTS model instance
        eval_audio_cfg: dict from config["eval_audio"]
    """
    if not eval_audio_cfg.get("enabled", True):
        return None

    test_texts = eval_audio_cfg.get("test_texts", [])
    if not test_texts:
        print("[audio_logging] Qwen3-TTS audio eval: no test_texts in eval_audio config — skipping")
        return None

    lang_code   = eval_audio_cfg.get("lang_code",   "auto")
    voice       = eval_audio_cfg.get("voice",       "Chelsie")
    max_tokens  = eval_audio_cfg.get("max_tokens",  400)
    temperature = eval_audio_cfg.get("temperature", 0.7)
    max_samples = eval_audio_cfg.get("max_samples", 2)
    test_texts  = test_texts[:max_samples]

    print(f"[audio_logging] Qwen3-TTS audio eval: {len(test_texts)} prompts, "
          f"lang_code={lang_code}, voice={voice}")

    def audio_eval_fn(model, global_step: int, tb_writer) -> None:
        for i, text in enumerate(test_texts):
            try:
                audio_chunks: list[np.ndarray] = []
                sr = 24000

                for result in model.generate(
                    text,
                    voice=voice,
                    lang_code=lang_code,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=False,
                ):
                    if result.audio is not None:
                        audio_chunks.append(np.array(result.audio))
                        sr = result.sample_rate

                if not audio_chunks:
                    continue

                audio = np.concatenate(audio_chunks).astype(np.float32)
                audio = _normalize_audio(audio)
                tb_writer.add_audio(
                    f"eval_audio/sample_{i}", audio, global_step, sample_rate=sr
                )

            except Exception as e:
                print(f"[audio_logging] Qwen3-TTS sample {i} failed at step {global_step}: {e}")

    return audio_eval_fn
