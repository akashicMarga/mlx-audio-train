#!/usr/bin/env python3
"""
Fine-tune (or train from scratch) a Pocket TTS FlowLM in MLX with the LSD objective —
the official finetune.yaml / finetune_language.yaml / scratch.yaml recipes.

Typical use: turn the public 24-layer English teacher into a Hindi teacher (new
tokenizer, fresh text table), then distil it into students of any size with
scripts/pocket_tts_distill.py. Or adapt an existing model to new voices / a domain.

Usage:
    python scripts/pocket_tts_finetune.py --smoke-test
    python scripts/pocket_tts_finetune.py --config configs/pocket_tts/hindi/teacher_24l_from_english.yaml --benchmark 5
    python scripts/pocket_tts_finetune.py --config configs/pocket_tts/hindi/teacher_24l_from_english.yaml
    python scripts/pocket_tts_finetune.py --config ... --resume checkpoints/.../step_0010000

Checkpoints: <run_dir>/step_N/{model.safetensors, config.yaml} in the official layout
(EMA weights). A teacher samples guided: pocket-tts generate --config <ckpt>/config.yaml --cfg 2
(check your pocket-tts version's CLI for the guidance flag).
"""

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.distill import trainable_count  # noqa: E402
from models.pocket_tts.finetune import build_finetune  # noqa: E402
from models.pocket_tts.objectives import flow_lm_loss, update_latent_stats  # noqa: E402
from models.pocket_tts.train_utils import (  # noqa: E402
    TRAIN_DEFAULTS, build_datasets, merge, normalize, synthetic_batch, synthetic_manifest,
    time_steps, tiny_raw, train_loop,
)

DEFAULTS = merge(TRAIN_DEFAULTS, {
    "model": {"reset_text_embedding": False, "freeze_head": False},
    "flow": {"type": "lsd", "kwargs": {}},
    "loss": {"flow_batch_multiplier": 4, "eos_loss_weight": 0.1, "text_dropout": 0.2,
             "voice_dropout": 0.2},
    "stats_ema_steps": 0, "stats_ema_decay": 0.999,
    "optim": {"lr": 2e-4, "schedule": "constant"},
    "train": {"ema_decay": 0.999, "max_steps": 20000, "run_dir": "checkpoints/pocket-finetune"},
})


def smoke_setup(tmp: Path) -> dict:
    """Tiny model from scratch on synthetic latents (no downloads)."""
    raw = tiny_raw(layers=2, d_model=64, z_dim=None)
    (tmp / "model.yaml").write_text(yaml.safe_dump(raw))
    mx.save_safetensors(str(tmp / "mimi.safetensors"), {"mimi.placeholder": mx.zeros((1,))})
    manifest = synthetic_manifest()
    return {
        "model": {"config": str(tmp / "model.yaml"), "weights": None,
                  "mimi_weights": str(tmp / "mimi.safetensors")},
        "data": {"train": manifest, "valid": manifest[:8], "tokenizer": lambda s: [ord(c) % 64 for c in s]},
        "stats_ema_steps": 20, "stats_ema_decay": 0.9,
        "optim": {"lr": 1e-3, "warmup_steps": 10},
        "train": {"batch_size": 8, "grad_accum": 1, "max_steps": 150, "log_freq": 25,
                  "valid_freq": 75, "num_valid_batches": 2, "ckpt_freq": 150, "ema_decay": 0.99,
                  "run_dir": str(tmp / "run")},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, metavar="STEPS")
    p.add_argument("--bench-sec", type=float, default=12.0)
    args = p.parse_args()

    tmp = None
    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp(prefix="pocket_ft_smoke_"))
        user = smoke_setup(tmp)
    else:
        if not args.config:
            p.error("--config is required (or --smoke-test)")
        user = yaml.safe_load(Path(args.config).read_text())
    cfg = merge(DEFAULTS, user)
    t, L = cfg["train"], cfg["loss"]
    if args.max_steps:
        t["max_steps"] = args.max_steps
    mx.random.seed(t["seed"])
    rng = np.random.default_rng(t["seed"])

    trainable, raw = build_finetune(cfg)
    fl = trainable.flow_lm
    mimi_source = cfg["model"].get("mimi_weights") or cfg["model"]["weights"] or raw.get("weights_path")
    if not mimi_source:
        p.error("set model.mimi_weights (a pocket-tts model.safetensors) when training from scratch")
    print(f"[finetune] {fl.args.num_layers}L d={fl.args.d_model}, trainable "
          f"{trainable_count(trainable) / 1e6:.2f}M (head {'frozen' if cfg['model']['freeze_head'] else 'trained'}, "
          f"text table {'reset' if cfg['model']['reset_text_embedding'] else 'kept'})")

    def loss_fn(model, batch, training):
        batch = normalize(batch, model.flow_lm)
        return flow_lm_loss(model.flow_lm, model.objective, batch, training=training, rng=rng,
                            text_dropout=L["text_dropout"], voice_dropout=L["voice_dropout"],
                            eos_loss_weight=L["eos_loss_weight"],
                            flow_batch_multiplier=L["flow_batch_multiplier"])

    def before_step(step, micro):
        if step < cfg["stats_ema_steps"]:
            for b in micro:
                update_latent_stats(fl, b["latents"], b["mask"], cfg["stats_ema_decay"])

    if args.benchmark:
        batch = synthetic_batch(fl, t["batch_size"], args.bench_sec)
        time_steps(loss_fn, trainable, batch, args.benchmark, cfg)
        return

    train_iter, valid_ds = build_datasets(cfg["data"], t, fl.args.tokenizer_path)
    out = train_loop(cfg, trainable, fl, raw, loss_fn, train_iter, valid_ds, mimi_source,
                     args.resume, before_step)

    if args.smoke_test:
        ok = math.isfinite(out["last_loss"]) and out["last_loss"] < out["first_loss"]
        print(f"[smoke] loss {out['first_loss']:.4f} -> {out['last_loss']:.4f}: {'OK' if ok else 'FAILED'}")
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
