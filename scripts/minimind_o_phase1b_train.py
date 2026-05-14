"""
MiniMind-O Phase 1b — Top-2 Thinker layers + projector fine-tuning (pure MLX).

Trainable (~16M):
  audio_proj          ~1.2M
  model.layers[-2]    ~7.4M
  model.layers[-1]    ~7.4M
  model.norm          ~0.001M

Monitoring:
  tensorboard --logdir <save_dir>/tensorboard
  tail -f <save_dir>/train_log.jsonl
  cat  <save_dir>/samples_epochN.txt

Usage:
    python scripts/minimind_o_phase1b_train.py \\
        --config   configs/minimind_o/whisper_small_phase1b.yaml \\
        --data     data/indicSUPERB_hindi/train.parquet \\
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
# Forward step — gradient through audio_proj + last 2 layers + norm
# ---------------------------------------------------------------------------

def forward_step(model, convs, enc_feats_np, T_audio, bos_ids, eos_ids, tokenizer):
    mlx_projected = None
    if enc_feats_np is not None and T_audio > 0:
        raw_mx = mx.array(enc_feats_np)[None]
        proj   = model.audio_proj(raw_mx)
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
# Freeze strategy
# ---------------------------------------------------------------------------

def apply_phase1b_freeze(model: MiniMindOmni) -> None:
    model.freeze()
    model.audio_proj.unfreeze()
    layers = model.model.layers
    n = len(layers)
    layers[n - 1].unfreeze()
    layers[n - 2].unfreeze()
    model.model.norm.unfreeze()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    omni_cfg, train_cfg = load_config_yaml(args.config)
    if args.data:
        train_cfg.data_path = args.data
    if args.epochs > 0:
        train_cfg.epochs = args.epochs
    if args.save_dir:
        train_cfg.save_dir = args.save_dir

    log(f"Phase 1b: encoder={omni_cfg.audio_encoder_type}  mode={train_cfg.mode}")

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

    val_path = args.val_data
    if not val_path:
        val_path = train_cfg.data_path.replace("train.parquet", "val.parquet")
    val_samples = []
    if val_path and os.path.exists(val_path):
        val_samples = load_parquet_samples(val_path, audio_enc, tokenizer, omni_cfg, "val")
    else:
        log("No val set — skipping eval")

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
        log(f"  Backbone loaded (audio_proj excluded)")

    # Warm-start projector from Phase 1a
    proj_path = train_cfg.load_projector
    if proj_path and os.path.exists(proj_path):
        proj_npz = np.load(proj_path, allow_pickle=False)
        saved_in = proj_npz["ln.weight"].shape[0] if "ln.weight" in proj_npz else None
        if saved_in == omni_cfg.audio_hidden_size:
            model.audio_proj.load_weights(
                [(k, mx.array(v)) for k, v in proj_npz.items()], strict=False
            )
            log(f"  Loaded Phase 1a projector from {proj_path}  (in_dim={saved_in})")
        else:
            log(f"  WARN: projector dim {saved_in} != {omni_cfg.audio_hidden_size} — skipping")
    else:
        log(f"  No Phase 1a projector at '{proj_path}' — random init")

    apply_phase1b_freeze(model)

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_total     = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M"
        f"  (audio_proj + last 2 Thinker layers + norm)")

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
    log(f"\nPhase 1b: {train_cfg.epochs} epochs × {len(samples)} = "
        f"{train_cfg.epochs * len(samples)} steps  LR={train_cfg.learning_rate}\n")

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
            log("  Evaluating val set ...")
            avg_val = evaluate(model, val_samples, tokenizer, bos_ids, eos_ids)
            log(f"  val_loss={avg_val:.4f}")

        logger.log_epoch(global_step, epoch + 1, avg_train, avg_val)

        # ── Text samples ────────────────────────────────────────────────────────
        if val_samples:
            save_text_samples(model, val_samples, tokenizer, bos_ids, eos_ids,
                              epoch + 1, train_cfg.save_dir)

        # ── Checkpoint ─────────────────────────────────────────────────────────
        ckpt_dir = os.path.join(train_cfg.save_dir, f"epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        proj_w = {k: np.array(v) for k, v in tree_flatten(model.audio_proj.parameters())}
        np.savez(os.path.join(ckpt_dir, "audio_proj.npz"), **proj_w)

        n_layers = len(model.model.layers)
        for idx in [n_layers - 2, n_layers - 1]:
            lw = {k: np.array(v) for k, v in tree_flatten(model.model.layers[idx].parameters())}
            np.savez(os.path.join(ckpt_dir, f"thinker_layer{idx}.npz"), **lw)

        nw = {k: np.array(v) for k, v in tree_flatten(model.model.norm.parameters())}
        np.savez(os.path.join(ckpt_dir, "thinker_norm.npz"), **nw)

        log(f"  Checkpoint → {ckpt_dir}/")

    logger.close()
    log("\nPhase 1b complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MiniMind-O Phase 1b — top-2 layers + projector")
    p.add_argument("--config",    required=True)
    p.add_argument("--data",      default="")
    p.add_argument("--val-data",  default="")
    p.add_argument("--epochs",    default=0, type=int)
    p.add_argument("--save_dir",  default="")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
