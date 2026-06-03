"""
OmniVox Phase 1a — Projector alignment training (pure MLX).

Trains ONLY the MMAudioProjector (~1.5M params). Everything else is frozen.

Objective: project Whisper encoder features (768d) into Qwen2.5-0.5B's hidden
space (896d) so the frozen Thinker can predict the corresponding Hindi text tokens.

Data flow:
  Hindi audio → Whisper-small (frozen, 50fps) → [numpy, no grad]
                       ↓
            MMAudioProjector(768→896)  ← ONLY trainable part
                       ↓  (injected at <|audio_pad|>=151665 positions)
  frozen Qwen2.5-0.5B → text logits → cross-entropy loss on assistant tokens

Monitoring:
  tensorboard --logdir <save_dir>/tensorboard
  tail -f <save_dir>/train_log.jsonl
  cat  <save_dir>/samples_epochN.txt

Usage:
    python scripts/omnivox_phase1a_train.py \\
        --config configs/omnivox/whisper_small_phase1a.yaml \\
        --data   data/fleurs_hindi/train.parquet \\
        --val-data data/fleurs_hindi/val.parquet
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from models.omnivox.config import load_omnivox_config
from models.omnivox.model import OmniVox
from models.minimind_o.speech_encoder import load_audio_encoder
from scripts.minimind_o_train_utils import (
    build_tokens_with_audio,
    load_parquet_samples,
    evaluate,
    save_text_samples,
    TrainingLogger,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Forward pass — gradient only through audio_proj
# ---------------------------------------------------------------------------

def forward_step(model, convs, enc_feats_np, T_audio, bos_ids, eos_ids, tokenizer,
                 audio_pad_str="<|audio_pad|>"):
    mlx_projected = None
    if enc_feats_np is not None and T_audio > 0:
        raw_mx = mx.array(enc_feats_np)[None]       # (1, T, 768) — constant, no grad
        proj   = model.audio_proj(raw_mx)           # (1, T, hidden) — TRAINABLE
        mlx_projected = [proj[0]]

    token_ids, labels = build_tokens_with_audio(
        convs, T_audio, tokenizer, bos_ids, eos_ids, audio_pad_str=audio_pad_str,
    )
    ids  = mx.array([token_ids], dtype=mx.int32)
    labs = mx.array([labels],    dtype=mx.int32)

    out    = model(ids, use_cache=False, mlx_audio_feats=mlx_projected)
    logits = out["logits"]

    logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    targets     = labs[:, 1:].reshape(-1)
    valid       = (targets != -100).astype(mx.float32)
    valid_count = valid.sum()

    targets_safe = mx.where(targets == -100, mx.zeros_like(targets), targets)
    loss_all     = nn.losses.cross_entropy(logits_flat, targets_safe, reduction="none")
    return mx.where(valid_count > 0,
                    (loss_all * valid).sum() / (valid_count + 1e-9),
                    mx.array(0.0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    omni_cfg, train_cfg = load_omnivox_config(args.config)
    if args.data:
        train_cfg.data_path = args.data
    if args.epochs > 0:
        train_cfg.epochs = args.epochs
    if args.save_dir:
        train_cfg.save_dir = args.save_dir

    log(f"OmniVox Phase 1a  encoder={omni_cfg.audio_encoder_type}  mode={train_cfg.mode}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Load HF tokenizer directly (mlx_lm's TokenizerWrapper is not callable).
    log(f"Loading tokenizer from {omni_cfg.backbone_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(omni_cfg.backbone_path, trust_remote_code=True)

    # Sarvam-1: reuse <<reserved_token_0>> (id=3) as audio_pad — already in vocab.
    # Qwen2.5:  add <|audio_pad|> as additional_special_token.
    # Choice driven by omni_cfg.audio_pad_token_str.
    audio_pad_str = omni_cfg.audio_pad_token_str
    if audio_pad_str.startswith("<|") and audio_pad_str not in tokenizer.get_vocab():
        # Legacy Qwen path — add as new special token
        tokenizer.add_special_tokens({"additional_special_tokens": [audio_pad_str]})
    audio_pad_id = tokenizer.convert_tokens_to_ids(audio_pad_str)
    omni_cfg.audio_pad_id = audio_pad_id
    log(f"  audio_pad: {audio_pad_str!r} → id={audio_pad_id}  "
        f"(model vocab={omni_cfg.vocab_size})")
    assert audio_pad_id < omni_cfg.vocab_size, \
        f"audio_pad_id {audio_pad_id} is OOV for backbone vocab {omni_cfg.vocab_size}"

    # Chat format markers (from config — ChatML for Qwen, Llama-style for Sarvam).
    tokenizer.bos_token = omni_cfg.bos_token
    tokenizer.eos_token = omni_cfg.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # bos_ids / eos_ids identify assistant turn boundaries for label masking.
    # Format: <bos>assistant\n...content...<eos>\n
    bos_ids = tokenizer(f"{omni_cfg.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_ids = tokenizer(f"{omni_cfg.eos_token}\n",          add_special_tokens=False).input_ids
    log(f"  bos_ids={bos_ids}  eos_ids={eos_ids}  vocab={len(tokenizer)}")

    # ── Audio encoder ──────────────────────────────────────────────────────────
    log(f"Loading audio encoder: {omni_cfg.audio_encoder_type} ...")
    audio_enc, _ = load_audio_encoder(omni_cfg)

    # ── Dataset ────────────────────────────────────────────────────────────────
    samples = load_parquet_samples(train_cfg.data_path, audio_enc, tokenizer, omni_cfg, "train")
    if not samples:
        log("ERROR: no training samples found")
        sys.exit(1)

    val_path = args.val_data or train_cfg.val_data_path
    if not val_path:
        val_path = train_cfg.data_path.replace("train.parquet", "val.parquet")
    val_samples = []
    if val_path and os.path.exists(val_path):
        val_samples = load_parquet_samples(val_path, audio_enc, tokenizer, omni_cfg, "val")
    else:
        log("No val set found — skipping eval")

    # ── Model ──────────────────────────────────────────────────────────────────
    log(f"Loading OmniVox backbone: {omni_cfg.backbone_path} ...")
    model = OmniVox(omni_cfg)
    model.load_backbone()           # loads Qwen2.5-0.5B via mlx_lm

    model.freeze()
    model.audio_proj.unfreeze()

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_total     = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {n_trainable/1e6:.2f}M (audio_proj only)"
        f"  |  Total: {n_total/1e6:.2f}M")

    # ── Optimizer + loss fn ────────────────────────────────────────────────────
    optimizer = optim.AdamW(learning_rate=train_cfg.learning_rate, weight_decay=0.01)
    loss_fn   = nn.value_and_grad(
        model,
        lambda m, convs, feats, T, bids, eids, tok:
            forward_step(m, convs, feats, T, bids, eids, tok,
                         audio_pad_str=audio_pad_str),
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    logger = TrainingLogger(train_cfg.save_dir)

    global_step = 0
    log(f"\nPhase 1a: {train_cfg.epochs} epochs × {len(samples)} samples"
        f"  LR={train_cfg.learning_rate}\n")

    for epoch in range(train_cfg.epochs):
        np.random.shuffle(samples)
        epoch_loss = 0.0
        n_valid    = 0
        t_epoch    = time.time()

        for step, sample in enumerate(samples):
            loss, grads = loss_fn(
                model,
                sample["convs"], sample["enc_feats"], sample["T_audio"],
                bos_ids, eos_ids, tokenizer,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val    = float(loss)
            epoch_loss += loss_val
            if loss_val > 0:
                n_valid += 1
            global_step += 1

            logger.log_step(global_step, loss_val, epoch + 1, train_cfg.learning_rate)

            if global_step % train_cfg.log_interval == 0 or step == 0:
                log(f"  ep{epoch+1}  step{step+1}/{len(samples)}"
                    f"  loss={loss_val:.4f}  global={global_step}")

        avg_train = epoch_loss / max(n_valid, 1)
        elapsed   = time.time() - t_epoch
        log(f"Epoch {epoch+1} done — train_loss={avg_train:.4f}  ({elapsed:.0f}s)")

        avg_val = None
        if val_samples:
            log("  Evaluating val set ...")
            avg_val = evaluate(model, val_samples, tokenizer, bos_ids, eos_ids,
                               audio_pad_str=audio_pad_str)
            log(f"  val_loss={avg_val:.4f}")

        logger.log_epoch(global_step, epoch + 1, avg_train, avg_val)

        if val_samples:
            save_text_samples(model, val_samples, tokenizer, bos_ids, eos_ids,
                              epoch + 1, train_cfg.save_dir,
                              audio_pad_str=audio_pad_str)

        # Save projector checkpoint
        ckpt = os.path.join(train_cfg.save_dir, f"audio_proj_epoch{epoch+1}.npz")
        proj_w = {k: np.array(v) for k, v in tree_flatten(model.audio_proj.parameters())}
        np.savez(ckpt, **proj_w)
        log(f"  Checkpoint → {ckpt}")

    logger.close()
    log("\nPhase 1a complete.")
    log(f"Next — Phase 1b:")
    log(f"  python scripts/omnivox_phase1b_train.py \\")
    log(f"      --config configs/omnivox/whisper_small_phase1b.yaml \\")
    log(f"      --projector {os.path.join(train_cfg.save_dir, f'audio_proj_epoch{train_cfg.epochs}.npz')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OmniVox Phase 1a — projector alignment")
    p.add_argument("--config",    required=True)
    p.add_argument("--data",      default="", help="Override train data_path from config")
    p.add_argument("--val-data",  default="", help="Val parquet (auto-inferred if omitted)")
    p.add_argument("--epochs",    default=0,  type=int, help="Override epochs from config")
    p.add_argument("--save-dir",  default="", dest="save_dir")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
