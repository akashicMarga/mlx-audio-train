"""
OmniVox Phase 1b — Top-N Qwen2.5 layers + projector fine-tuning (pure MLX).

Builds on Phase 1a. Loads the trained audio_proj checkpoint, then continues
training with the top N Qwen2.5 layers + final norm unfrozen.

Trainable (default N=2, fp16 backbone):
  audio_proj          ~1.5M   (projector — already aligned from Phase 1a)
  qwen.model.layers[-1]  ~7M  (last Qwen2.5-0.5B layer)
  qwen.model.layers[-2]  ~7M  (second-to-last)
  qwen.model.norm        ~1K  (final RMSNorm)
  Total:              ~15.5M params

Why fp16 backbone for Phase 1b:
  The 4-bit quantized model (Phase 1a) is fine when only audio_proj is trained
  (gradients flow through frozen quantized layers without updating them).
  In Phase 1b we actually update Qwen's top layers, so we need full-precision
  (fp16) weights: mlx-community/Qwen2.5-0.5B-Instruct (~1 GB).

Monitoring:
  tensorboard --logdir <save_dir>/tensorboard
  tail -f <save_dir>/train_log.jsonl
  cat  <save_dir>/samples_epochN.txt

Usage:
    python scripts/omnivox_phase1b_train.py \\
        --config    configs/omnivox/whisper_small_phase1b.yaml \\
        --data      data/fleurs_hindi/train.parquet \\
        --val-data  data/fleurs_hindi/val.parquet \\
        --projector out/omnivox_phase1a/audio_proj_epoch5.npz
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
# Forward pass — gradient through audio_proj + top-N Qwen layers + norm
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

    n_unfreeze = train_cfg.unfreeze_top_layers
    log(f"OmniVox Phase 1b  encoder={omni_cfg.audio_encoder_type}"
        f"  unfreeze_top={n_unfreeze}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log(f"Loading tokenizer from {omni_cfg.backbone_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(omni_cfg.backbone_path, trust_remote_code=True)

    audio_pad_str = omni_cfg.audio_pad_token_str
    if audio_pad_str.startswith("<|") and audio_pad_str not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [audio_pad_str]})
    audio_pad_id = tokenizer.convert_tokens_to_ids(audio_pad_str)
    omni_cfg.audio_pad_id = audio_pad_id
    log(f"  audio_pad: {audio_pad_str!r} → id={audio_pad_id}")
    assert audio_pad_id < omni_cfg.vocab_size

    tokenizer.bos_token = omni_cfg.bos_token
    tokenizer.eos_token = omni_cfg.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bos_ids = tokenizer(f"{omni_cfg.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_ids = tokenizer(f"{omni_cfg.eos_token}\n",          add_special_tokens=False).input_ids
    log(f"  bos_ids={bos_ids}  eos_ids={eos_ids}")

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
    log("  NOTE: Phase 1b requires fp16 backbone for weight updates.")
    log("  Set backbone_path: mlx-community/Qwen2.5-0.5B-Instruct in config if using 4-bit.")
    model = OmniVox(omni_cfg)
    model.load_backbone()

    # Freeze all, then selectively unfreeze
    model.freeze()
    model.audio_proj.unfreeze()
    model.unfreeze_top_layers(n=n_unfreeze)

    # Warm-start projector from Phase 1a
    proj_path = args.projector or train_cfg.load_projector
    if proj_path:
        loaded = model.load_projector_checkpoint(proj_path)
        if loaded:
            log(f"  Loaded Phase 1a projector from {proj_path}")
        else:
            log(f"  WARN: could not load projector from '{proj_path}' — random init")
    else:
        log("  No Phase 1a projector specified — random init")

    n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_total     = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {n_trainable/1e6:.2f}M"
        f"  (audio_proj + top-{n_unfreeze} Qwen layers + norm)"
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
    log(f"\nPhase 1b: {train_cfg.epochs} epochs × {len(samples)} samples"
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

        # Save checkpoints per epoch
        ckpt_dir = os.path.join(train_cfg.save_dir, f"epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Use safetensors for Sarvam-1 layers (~50M each) — npz silently fails
        # on large MLX tensor → numpy conversions.
        mx.save_safetensors(
            os.path.join(ckpt_dir, "audio_proj.safetensors"),
            dict(tree_flatten(model.audio_proj.parameters())),
        )

        n_layers = len(model.backbone.model.layers)
        for idx in range(n_layers - n_unfreeze, n_layers):
            mx.save_safetensors(
                os.path.join(ckpt_dir, f"backbone_layer{idx}.safetensors"),
                dict(tree_flatten(model.backbone.model.layers[idx].parameters())),
            )

        mx.save_safetensors(
            os.path.join(ckpt_dir, "backbone_norm.safetensors"),
            dict(tree_flatten(model.backbone.model.norm.parameters())),
        )

        log(f"  Checkpoint → {ckpt_dir}/")

    logger.close()
    log("\nPhase 1b complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OmniVox Phase 1b — top-N layers + projector")
    p.add_argument("--config",     required=True)
    p.add_argument("--data",       default="", help="Override train data_path from config")
    p.add_argument("--val-data",   default="", help="Val parquet (auto-inferred if omitted)")
    p.add_argument("--projector",  default="", help="Phase 1a audio_proj.npz to warm-start from")
    p.add_argument("--epochs",     default=0, type=int)
    p.add_argument("--save-dir",   default="", dest="save_dir")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
