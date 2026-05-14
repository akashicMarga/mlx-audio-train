"""
Shared utilities for MiniMind-O training scripts (Phase 1a, 1b, ...).

Provides:
  - Audio helpers (decode bytes → numpy)
  - Token construction with <|audio_pad|> markers
  - Evaluation loop (val loss, no gradient)
  - Greedy text sample generation
  - Logging: TensorBoard (tensorboardX) + JSONL (compatible with watch_training.py)
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn


MAX_AUDIO_FRAMES = 400   # 400 frames ÷ 50fps = 8s — covers full FLEURS clip, fits in 512 tokens
MAX_SEQ_LEN      = 512
AUDIO_PAD_ID     = 16    # <|audio_pad|> token id


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def decode_audio_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Raw WAV/audio bytes → 16 kHz float32 numpy array."""
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)
    except Exception:
        return np.zeros(target_sr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Token construction
# ---------------------------------------------------------------------------

def build_tokens_with_audio(
    convs: list[dict],
    T_audio: int,
    tokenizer,
    bos_ids: list[int],
    eos_ids: list[int],
) -> tuple[list[int], list[int]]:
    """
    Build (token_ids, labels) for one sample.
    The first user turn gets <|audio_pad|> × T_audio injected in place of text.
    Labels are -100 everywhere except assistant turns.
    """
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    first_user_done = False
    prompt = ""
    for turn in convs:
        role    = turn["role"]
        content = turn["content"]
        if role == "user" and T_audio > 0 and not first_user_done:
            content = "<|audio_pad|>" * T_audio
            first_user_done = True
        prompt += f"{bos}{role}\n{content}{eos}\n"

    token_ids = tokenizer(prompt, add_special_tokens=False).input_ids[:MAX_SEQ_LEN]
    token_ids += [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(token_ids))

    labels = [-100] * MAX_SEQ_LEN
    j = 0
    while j < len(token_ids):
        if token_ids[j:j+len(bos_ids)] == bos_ids:
            start = j + len(bos_ids)
            end   = start
            while end < len(token_ids):
                if token_ids[end:end+len(eos_ids)] == eos_ids:
                    break
                end += 1
            for k in range(start, min(end + len(eos_ids), MAX_SEQ_LEN)):
                labels[k] = token_ids[k]
            j = end + len(eos_ids)
        else:
            j += 1

    return token_ids, labels


# ---------------------------------------------------------------------------
# Parquet loader (supports flat large_binary + legacy list<binary>)
# ---------------------------------------------------------------------------

def load_parquet_samples(
    parquet_path: str,
    audio_enc,
    tokenizer,
    config,
    split_name: str = "train",
) -> list[dict]:
    """
    Load Parquet and pre-encode all audio once (before training).
    Returns list of {convs, enc_feats (T,H) numpy or None, T_audio}.
    """
    import pyarrow.parquet as pq

    pf   = pq.ParquetFile(parquet_path, pre_buffer=False)
    schema_names = set(pf.schema_arrow.names)
    has_flat = "question_audio" in schema_names
    col  = "question_audio" if has_flat else "question_audios"
    cols = ["conversations"] + ([col] if col in schema_names else [])
    tbl  = pf.read(columns=cols)
    rows = tbl.to_pydict()
    n    = len(rows["conversations"])

    samples = []
    for i in range(n):
        convs = json.loads(rows["conversations"][i])

        if has_flat:
            audio_bytes = rows.get("question_audio", [None] * n)[i]
        else:
            qa = rows.get("question_audios", [None] * n)[i]
            audio_bytes = qa[0] if (qa and len(qa) > 0) else None

        enc_feats = None
        T_audio   = 0
        if audio_enc is not None and audio_bytes is not None:
            wav = decode_audio_bytes(audio_bytes)
            try:
                out_np, _ = audio_enc(wav)
                T_audio   = min(out_np.shape[1], MAX_AUDIO_FRAMES)
                enc_feats = out_np[0, :T_audio].astype(np.float32)
            except Exception as e:
                pass

        samples.append({"convs": convs, "enc_feats": enc_feats, "T_audio": T_audio})

    n_audio = sum(1 for s in samples if s["enc_feats"] is not None)
    print(f"[{time.strftime('%H:%M:%S')}] {split_name}: {len(samples)} samples  (with_audio={n_audio})",
          flush=True)
    return samples


# ---------------------------------------------------------------------------
# Evaluation — val loss, no gradients
# ---------------------------------------------------------------------------

def evaluate(
    model,
    val_samples: list[dict],
    tokenizer,
    bos_ids: list[int],
    eos_ids: list[int],
    max_samples: int = 200,
) -> float:
    """
    Compute average val loss on up to max_samples samples (no gradients).
    Returns average loss over samples that have valid supervised tokens.
    """
    subset  = val_samples[:max_samples]
    total   = 0.0
    n_valid = 0

    for sample in subset:
        enc_feats_np = sample["enc_feats"]
        T_audio      = sample["T_audio"]

        mlx_projected = None
        if enc_feats_np is not None and T_audio > 0:
            raw_mx = mx.array(enc_feats_np)[None]
            proj   = model.audio_proj(raw_mx)
            mlx_projected = [proj[0]]

        token_ids, labels = build_tokens_with_audio(
            sample["convs"], T_audio, tokenizer, bos_ids, eos_ids
        )
        ids  = mx.array([token_ids], dtype=mx.int32)
        labs = mx.array([labels],    dtype=mx.int32)

        out    = model(ids, use_cache=False, mlx_audio_feats=mlx_projected)
        logits = out["logits"]

        logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets     = labs[:, 1:].reshape(-1)
        valid       = (targets != -100).astype(mx.float32)
        valid_count = float(valid.sum())

        if valid_count > 0:
            targets_safe = mx.where(targets == -100, mx.zeros_like(targets), targets)
            loss_all     = nn.losses.cross_entropy(logits_flat, targets_safe, reduction="none")
            loss_val     = float((loss_all * valid).sum() / (valid_count + 1e-9))
            total       += loss_val
            n_valid     += 1

    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Text sample generation — greedy decode, no Talker
# ---------------------------------------------------------------------------

def generate_text_sample(
    model,
    sample: dict,
    tokenizer,
    bos_ids: list[int],
    eos_ids: list[int],
    max_new: int = 80,
) -> tuple[str, str]:
    """
    Greedily decode text from one val sample.
    Returns (reference, generated) both as strings.
    """
    enc_feats_np = sample["enc_feats"]
    T_audio      = sample["T_audio"]
    convs        = sample["convs"]

    # Reference text (what the assistant should say)
    ref = next((t["content"] for t in convs if t["role"] == "assistant"), "")

    mlx_projected = None
    if enc_feats_np is not None and T_audio > 0:
        raw_mx = mx.array(enc_feats_np)[None]
        proj   = model.audio_proj(raw_mx)
        mlx_projected = [proj[0]]

    # Build prompt up to (but not including) the assistant response
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    first_user_done = False
    prompt = ""
    for turn in convs:
        if turn["role"] == "assistant":
            break
        role    = turn["role"]
        content = turn["content"]
        if role == "user" and T_audio > 0 and not first_user_done:
            content = "<|audio_pad|>" * T_audio
            first_user_done = True
        prompt += f"{bos}{role}\n{content}{eos}\n"
    prompt += f"{bos}assistant\n"

    ids_list = tokenizer(prompt, add_special_tokens=False).input_ids[-256:]
    ids      = mx.array([ids_list], dtype=mx.int32)

    eos_id = tokenizer.eos_token_id or 2
    generated_ids = []
    rep_penalty   = 1.5   # penalise already-generated tokens to prevent loops

    for _ in range(max_new):
        out    = model(ids, use_cache=False, mlx_audio_feats=mlx_projected)
        logits = out["logits"][:, -1, :]   # (1, vocab)

        # Repetition penalty: divide logits of seen tokens by rep_penalty
        if generated_ids and rep_penalty != 1.0:
            logits_np = np.array(logits.tolist(), dtype=np.float32)
            for tok in set(generated_ids):
                logits_np[0, tok] /= rep_penalty
            logits = mx.array(logits_np)

        next_id = int(mx.argmax(logits, axis=-1)[0])
        if next_id == eos_id:
            break
        generated_ids.append(next_id)
        ids = mx.concatenate([ids, mx.array([[next_id]])], axis=1)
        mlx_projected = None   # only inject audio on first forward pass

    generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return ref, generated


def save_text_samples(
    model,
    val_samples: list[dict],
    tokenizer,
    bos_ids: list[int],
    eos_ids: list[int],
    epoch: int,
    save_dir: str,
    n_samples: int = 5,
) -> None:
    """Generate text for n_samples val items and write to samples_epoch{N}.txt."""
    lines = [f"=== Epoch {epoch} Text Samples ===\n"]
    for i, sample in enumerate(val_samples[:n_samples]):
        ref, gen = generate_text_sample(model, sample, tokenizer, bos_ids, eos_ids)
        lines.append(f"[{i+1}] ref: {ref}")
        lines.append(f"[{i+1}] gen: {gen}")
        lines.append("")

    out_path = os.path.join(save_dir, f"samples_epoch{epoch}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[{time.strftime('%H:%M:%S')}]   Text samples → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Logging: TensorBoard + JSONL
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Writes to:
      {save_dir}/tensorboard/   — TensorBoard events (tensorboardX)
      {save_dir}/train_log.jsonl — JSONL (compatible with watch_training.py)
    """

    def __init__(self, save_dir: str):
        from tensorboardX import SummaryWriter
        tb_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer   = SummaryWriter(logdir=tb_dir)
        self.log_path = os.path.join(save_dir, "train_log.jsonl")
        self._log_f   = open(self.log_path, "a")
        print(f"[{time.strftime('%H:%M:%S')}] TensorBoard → {tb_dir}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] JSONL log   → {self.log_path}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] Run: tensorboard --logdir {tb_dir}", flush=True)

    def log_step(self, step: int, loss: float, epoch: int, lr: float) -> None:
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/lr",   lr,   step)
        entry = {"step": step, "loss": loss, "epoch": epoch, "lr": lr,
                 "ts": time.strftime("%H:%M:%S")}
        self._log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log_f.flush()

    def log_epoch(self, step: int, epoch: int,
                  train_loss: float, val_loss: Optional[float]) -> None:
        self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
        entry = {"step": step, "epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            entry["val_loss"] = val_loss
        self._log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log_f.flush()

    def close(self) -> None:
        self.writer.close()
        self._log_f.close()
