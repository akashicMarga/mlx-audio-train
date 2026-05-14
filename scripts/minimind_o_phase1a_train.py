"""
MiniMind-O Phase 1a — Projector alignment training (pure MLX, no PyTorch).

Trains ONLY the MMAudioProjector (~1.2M params). Everything else is frozen.

Objective: project Whisper encoder features into the Thinker's hidden space
so the (frozen) Thinker can predict the corresponding Hindi/Indic text tokens.

Data flow:
  User audio → Whisper encoder (frozen) → [numpy, no grad]
                      ↓
           MMAudioProjector  ← ONLY trainable part
                      ↓  (injected at <|audio_pad|> positions)
      frozen Thinker → text logits → cross-entropy loss on assistant tokens

Monitoring:
  tensorboard --logdir <save_dir>/tensorboard
  tail -f <save_dir>/train_log.jsonl
  cat  <save_dir>/samples_epochN.txt

Usage:
    python scripts/minimind_o_phase1a_train.py \\
        --config configs/minimind_o/whisper_small_phase1a.yaml \\
        --data   data/indicSUPERB_hindi/train.parquet \\
        --val-data data/indicSUPERB_hindi/val.parquet
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from models.minimind_o.config import OmniConfig, OmniTrainingConfig, load_config_yaml
from models.minimind_o.model import MiniMindOmni
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
# Forward pass (one training step)
# Gradient flows ONLY through model.audio_proj.
# ---------------------------------------------------------------------------

def forward_step(
    model: MiniMindOmni,
    convs: list[dict],
    enc_feats_np,
    T_audio: int,
    bos_ids: list[int],
    eos_ids: list[int],
    tokenizer,
) -> mx.array:
    mlx_projected = None
    if enc_feats_np is not None and T_audio > 0:
        raw_mx = mx.array(enc_feats_np)[None]       # (1, T, H) — constant, no grad
        proj   = model.audio_proj(raw_mx)           # (1, T, hidden) — TRAINABLE
        mlx_projected = [proj[0]]

    token_ids, labels = build_tokens_with_audio(
        convs, T_audio, tokenizer, bos_ids, eos_ids
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
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    omni_cfg, train_cfg = load_config_yaml(args.config)
    if args.data:
        train_cfg.data_path = args.data
    if args.epochs > 0:
        train_cfg.epochs = args.epochs
    if args.save_dir:
        train_cfg.save_dir = args.save_dir

    log(f"Phase 1a: encoder={omni_cfg.audio_encoder_type}  mode={train_cfg.mode}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    snap_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots/*/tokenizer.json"
    )
    snaps = sorted(glob.glob(snap_pattern))
    if not snaps:
        log("ERROR: tokenizer not found")
        sys.exit(1)
    tok_dir = os.path.dirname(snaps[-1])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    log(f"Tokenizer: vocab={len(tokenizer)}")

    bos_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_ids = tokenizer(f"{tokenizer.eos_token}\n",           add_special_tokens=False).input_ids

    # ── Audio encoder ──────────────────────────────────────────────────────────
    log(f"Loading audio encoder: {omni_cfg.audio_encoder_type} ...")
    audio_enc, _ = load_audio_encoder(omni_cfg)

    # ── Dataset ────────────────────────────────────────────────────────────────
    samples = load_parquet_samples(train_cfg.data_path, audio_enc, tokenizer, omni_cfg, "train")
    if not samples:
        log("ERROR: no samples")
        sys.exit(1)

    # Val set — try inferred path if not specified
    val_path = args.val_data
    if not val_path:
        val_path = train_cfg.data_path.replace("train.parquet", "val.parquet")
    val_samples = []
    if val_path and os.path.exists(val_path):
        val_samples = load_parquet_samples(val_path, audio_enc, tokenizer, omni_cfg, "val")
    else:
        log("No val set found — skipping eval")

    # ── Model ──────────────────────────────────────────────────────────────────
    log("Loading MiniMind-O ...")
    model = MiniMindOmni(omni_cfg, audio_encoder_path="")

    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "out", "minimind_3o", "weights.npz"
    )
    if os.path.exists(weights_path):
        weights = [
            (k, mx.array(v))
            for k, v in np.load(weights_path, allow_pickle=False).items()
            if not k.startswith("audio_proj.")
        ]
        model.load_weights(weights, strict=False)
        log(f"  Loaded backbone from {weights_path}  (audio_proj excluded)")
    else:
        log("  WARNING: no upstream weights")

    model.freeze()
    model.audio_proj.unfreeze()

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_frozen    = sum(p.size for _, p in tree_flatten(model.parameters())) - n_trainable
    log(f"  Trainable: {n_trainable/1e6:.2f}M (audio_proj)  Frozen: {n_frozen/1e6:.2f}M")

    # ── Optimizer + loss fn ────────────────────────────────────────────────────
    optimizer = optim.AdamW(learning_rate=train_cfg.learning_rate, weight_decay=0.01)
    loss_fn   = nn.value_and_grad(
        model,
        lambda m, convs, feats, T, bids, eids, tok:
            forward_step(m, convs, feats, T, bids, eids, tok),
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    logger = TrainingLogger(train_cfg.save_dir)

    global_step = 0
    total_steps = train_cfg.epochs * len(samples)
    log(f"\nPhase 1a: {train_cfg.epochs} epochs × {len(samples)} = {total_steps} steps"
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

        # ── Val loss ────────────────────────────────────────────────────────────
        avg_val = None
        if val_samples:
            log("  Evaluating on val set ...")
            avg_val = evaluate(model, val_samples, tokenizer, bos_ids, eos_ids)
            log(f"  val_loss={avg_val:.4f}")

        logger.log_epoch(global_step, epoch + 1, avg_train, avg_val)

        # ── Text samples ────────────────────────────────────────────────────────
        if val_samples:
            save_text_samples(model, val_samples, tokenizer, bos_ids, eos_ids,
                              epoch + 1, train_cfg.save_dir)

        # ── Checkpoint ─────────────────────────────────────────────────────────
        ckpt = os.path.join(train_cfg.save_dir, f"audio_proj_epoch{epoch+1}.npz")
        proj_w = {k: np.array(v) for k, v in tree_flatten(model.audio_proj.parameters())}
        np.savez(ckpt, **proj_w)
        log(f"  Checkpoint → {ckpt}")

    logger.close()
    log("\nPhase 1a complete.")
    log(f"Next: Phase 1b with: python scripts/minimind_o_phase1b_train.py "
        f"--config configs/minimind_o/whisper_small_phase1b.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MiniMind-O Phase 1a — projector alignment")
    p.add_argument("--config",    required=True)
    p.add_argument("--data",      default="", help="Override train data_path")
    p.add_argument("--val-data",  default="", help="Val parquet (auto-inferred if omitted)")
    p.add_argument("--epochs",    default=0,  type=int)
    p.add_argument("--save_dir",  default="")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
