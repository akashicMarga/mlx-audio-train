"""
OmniVox T2A inference — Hindi text → Hindi audio (via Mimi codes).

Loads a trained T2A checkpoint (Sarvam-1 backbone + Talker) and synthesises
speech for a Hindi text input. Decodes Mimi codes through the MLX Mimi codec
to a 24kHz WAV file.

The data the Talker was trained on is "echo" T2A:
    user:      <Hindi text>
    assistant: <same Hindi text>   ← Talker learns to speak this back

So during inference we build the full echo prompt and let the Talker generate
Mimi codes diagonally after the assistant turn start.

Usage:
    python scripts/omnivox_t2a_infer.py \\
        --config     configs/omnivox/sarvam_t2a.yaml \\
        --checkpoint /Users/akashsingh/Documents/exps/omnivox_sarvam_t2a/epoch10 \\
        --text       "नमस्ते मैं ठीक हूँ" \\
        --output     out_hindi.wav
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from transformers import AutoTokenizer

from models.omnivox.config import load_omnivox_config
from models.omnivox.model import OmniVox

AUDIO_PAD_CODE  = 2049
AUDIO_STOP_CODE = 2050
NUM_CODEBOOKS   = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_top_p(logits_np: np.ndarray, temperature: float, top_p: float,
                 forbid: tuple = ()) -> int:
    """Top-p sampling on a 1-D numpy logits vector."""
    logits = logits_np.astype(np.float32).copy()
    for tok in forbid:
        logits[tok] = -1e9

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-6)
    # Stable softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    order = np.argsort(-probs)
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    cutoff = int(np.searchsorted(cum, top_p)) + 1
    keep = order[:cutoff]
    p_keep = sorted_p[:cutoff]
    p_keep = p_keep / p_keep.sum()
    return int(np.random.choice(keep, p=p_keep))


# ---------------------------------------------------------------------------
# Autoregressive Talker generation with diagonal placement
# ---------------------------------------------------------------------------

def generate_audio_codes(
    model,
    prompt_ids: np.ndarray,
    asst_start: int,
    max_frames: int = 250,
    temperature: float = 0.7,
    top_p: float = 0.9,
    pad_token_id: int = 0,
    verbose_every: int = 25,
    min_frames_before_stop: int = 20,
) -> np.ndarray:
    """
    Autoregressive generation of (8, T_mimi) Mimi codes.

    The model is called with input_ids (fixed text prompt, padded to T_max)
    and audio_ids (8, T_max) where new codes are written into the diagonal
    positions as they are sampled.

    At step t, we want to predict the t-th frame for each of 8 codebooks.
    Codebook k's t-th frame should land at position asst_start + k + 1 + t.
    The model's autoregressive convention is "logits at position p predict
    the token at position p+1", so we read logits at position asst_start+k+t.
    """
    T_prompt = len(prompt_ids)
    # Total sequence length budget — enough for prompt + max_frames + 7 stagger offset
    T_max = T_prompt + max_frames + NUM_CODEBOOKS + 4

    input_ids_np = np.full(T_max, pad_token_id, dtype=np.int32)
    input_ids_np[:T_prompt] = prompt_ids

    audio_ids_np = np.full((NUM_CODEBOOKS, T_max), AUDIO_PAD_CODE, dtype=np.int32)

    audio_codes = [[] for _ in range(NUM_CODEBOOKS)]
    stop_at   = [None] * NUM_CODEBOOKS   # frame index where each codebook stopped

    for t in range(max_frames):
        input_ids = mx.array(input_ids_np[None], dtype=mx.int32)
        audio_ids = mx.array(audio_ids_np[None], dtype=mx.int32)

        out = model(input_ids, audio_ids=audio_ids, use_cache=False)
        audio_logits = out["audio_logits"]   # list of 8 × (1, T_max, 2112)
        mx.eval(audio_logits[0])             # force eval

        new_codes_this_step = []
        for k in range(NUM_CODEBOOKS):
            if stop_at[k] is not None:
                new_codes_this_step.append(AUDIO_STOP_CODE)
                continue

            # logits at position (asst_start + k + t) predict (asst_start + k + 1 + t)
            pos_pred = asst_start + k + t
            if pos_pred >= T_max - 1:
                stop_at[k] = t
                new_codes_this_step.append(AUDIO_STOP_CODE)
                continue

            logits_np = np.array(audio_logits[k][0, pos_pred, :].tolist())
            # Forbid stop + pad. Stop is additionally forbidden during the early
            # frames — the trained Talker (esp. cbk 4 & 6) overconfidently predicts
            # stop at the start, which would truncate audio to 0 frames.
            forbid = [AUDIO_PAD_CODE]
            if t < min_frames_before_stop:
                forbid.append(AUDIO_STOP_CODE)
            code = sample_top_p(logits_np, temperature, top_p, forbid=tuple(forbid))
            audio_codes[k].append(code)
            new_codes_this_step.append(code)

            if code == AUDIO_STOP_CODE:
                stop_at[k] = t
            else:
                # Write code into its diagonal position for next forward
                next_pos = asst_start + k + 1 + t
                if next_pos < T_max:
                    audio_ids_np[k, next_pos] = code

        if verbose_every and (t % verbose_every == 0):
            lens = [len(c) for c in audio_codes]
            stops = sum(1 for s in stop_at if s is not None)
            log(f"  step {t:3d}  codebook_lens={lens}  stopped={stops}/{NUM_CODEBOOKS}")

        if all(s is not None for s in stop_at):
            log(f"  all codebooks stopped at step {t}")
            break

    # Trim trailing stops; align to minimum length
    trimmed = []
    for k in range(NUM_CODEBOOKS):
        c = audio_codes[k]
        if AUDIO_STOP_CODE in c:
            c = c[:c.index(AUDIO_STOP_CODE)]
        trimmed.append(c)
    min_len = min(len(c) for c in trimmed) if trimmed else 0
    if min_len == 0:
        log("WARNING: no valid codes produced for at least one codebook")
        return np.zeros((NUM_CODEBOOKS, 0), dtype=np.int32)

    aligned = np.array([c[:min_len] for c in trimmed], dtype=np.int32)
    return aligned


# ---------------------------------------------------------------------------
# Mimi decode
# ---------------------------------------------------------------------------

def decode_codes_to_wav(codes: np.ndarray, output_path: str) -> None:
    """Decode (8, T) Mimi codes to a 24kHz WAV via MLX Mimi codec."""
    from mlx_audio.codec import Mimi
    import soundfile as sf

    log(f"Loading Mimi codec (kyutai/moshiko-mlx-bf16)...")
    mimi = Mimi.from_pretrained("kyutai/moshiko-mlx-bf16")

    # Filter codes — Mimi vocab is 2048 (codes 0..2047). Our special codes
    # (2048..2111) are not real audio, replace with 0 if they leaked through.
    codes_safe = np.where(codes >= 2048, 0, codes).astype(np.int32)

    codes_mx = mx.array(codes_safe[None])   # (1, 8, T)
    log(f"  decoding (1, 8, {codes_safe.shape[1]}) → audio ...")
    wav = mimi.decode(codes_mx)             # expected (1, 1, samples)
    mx.eval(wav)

    wav_np = np.array(wav.tolist())
    while wav_np.ndim > 1:
        wav_np = wav_np[0]

    sf.write(output_path, wav_np.astype(np.float32), int(mimi.sample_rate))
    log(f"  → {output_path}  ({len(wav_np) / mimi.sample_rate:.2f}s @ {mimi.sample_rate}Hz)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OmniVox T2A inference (Hindi text → audio)")
    p.add_argument("--config",     default="configs/omnivox/sarvam_t2a.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="Path to epoch dir containing talker.safetensors")
    p.add_argument("--text",       required=True, help="Hindi text to speak")
    p.add_argument("--output",     default="out_hindi.wav")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p",      type=float, default=0.9)
    p.add_argument("--max_frames", type=int, default=250)
    p.add_argument("--min_frames_before_stop", type=int, default=20,
                   help="Forbid AUDIO_STOP for the first N frames per codebook")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    omni_cfg, _ = load_omnivox_config(args.config)
    log(f"Loading backbone: {omni_cfg.backbone_path}")
    model = OmniVox(omni_cfg)
    model.load_backbone()

    # Load Talker + audio_proj from checkpoint dir
    talker_path = os.path.join(args.checkpoint, "talker.safetensors")
    proj_path   = os.path.join(args.checkpoint, "audio_proj.safetensors")
    if not os.path.exists(talker_path):
        raise FileNotFoundError(f"No talker.safetensors at {talker_path}")
    log(f"Loading Talker from {talker_path} "
        f"({os.path.getsize(talker_path) / (1024*1024):.0f} MB)")
    talker_w = mx.load(talker_path)
    model.talker.load_weights(list(talker_w.items()), strict=False)

    if os.path.exists(proj_path):
        log(f"Loading audio_proj from {proj_path}")
        proj_w = mx.load(proj_path)
        model.audio_proj.load_weights(list(proj_w.items()), strict=False)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(omni_cfg.backbone_path, trust_remote_code=True)
    tokenizer.bos_token = omni_cfg.bos_token
    tokenizer.eos_token = omni_cfg.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build echo T2A prompt — same format the model trained on
    bos, eos = omni_cfg.bos_token, omni_cfg.eos_token
    system_msg = "आप एक सहायक हिंदी वॉयस असिस्टेंट हैं।"
    prompt = (
        f"{bos}system\n{system_msg}{eos}\n"
        f"{bos}user\n{args.text}{eos}\n"
        f"{bos}assistant\n{args.text}{eos}\n"
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    # Find asst_start (position right after the LAST "<bos>assistant\n" sequence)
    bos_ids = tokenizer(f"{bos}assistant\n", add_special_tokens=False).input_ids
    asst_start = -1
    for i in range(len(prompt_ids) - len(bos_ids), -1, -1):
        if prompt_ids[i:i + len(bos_ids)] == bos_ids:
            asst_start = i + len(bos_ids)
            break
    if asst_start < 0:
        raise RuntimeError("Could not locate <bos>assistant\\n in prompt")

    log(f"Prompt: '{args.text}'")
    log(f"prompt_ids length={len(prompt_ids)}  asst_start={asst_start}")
    log(f"Sampling: temperature={args.temperature}  top_p={args.top_p}  max_frames={args.max_frames}")

    t0 = time.time()
    codes = generate_audio_codes(
        model, np.array(prompt_ids, dtype=np.int32), asst_start,
        max_frames=args.max_frames, temperature=args.temperature, top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        min_frames_before_stop=args.min_frames_before_stop,
    )
    log(f"Generated {codes.shape[1]} Mimi frames "
        f"(~{codes.shape[1] / 12.5:.2f}s of audio) in {time.time() - t0:.1f}s")

    if codes.shape[1] == 0:
        log("ERROR: no codes generated — Talker emitted stop immediately or no valid codebooks")
        return

    decode_codes_to_wav(codes, args.output)
    log("Done.")


if __name__ == "__main__":
    main()
