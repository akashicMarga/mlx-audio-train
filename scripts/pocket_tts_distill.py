#!/usr/bin/env python3
"""
Pocket TTS distillation in MLX: a frozen teacher (e.g. the public 24-layer English
model, or a 6-layer Hindi model) -> a shallower / narrower student that reproduces
the teacher's backbone output z, with classifier-free guidance baked in when the
teacher is a guided (undistilled) model. See models/pocket_tts/distill.py and
docs/pocket_tts_distillation.md.

Usage:
    # 0. no-download sanity check (tiny random teacher, synthetic data)
    python scripts/pocket_tts_distill.py --smoke-test

    # 1. how fast would a real run be on this Mac? (synthetic batches, real model sizes)
    python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml --benchmark 10

    # 2. (optional, ~3x faster steps) run the teacher once and store its targets
    python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml \
        --cache-teacher data/pocket_teacher_cache/en24l --cache-epochs 2
    #    then set data.teacher_cache: data/pocket_teacher_cache/en24l in the config

    # 3. train / resume
    python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml
    python scripts/pocket_tts_distill.py --config ... --resume checkpoints/pocket-distill-6l/step_0010000

Every checkpoint directory holds model.safetensors + config.yaml in the official
layout (EMA weights), runnable with:
    pocket-tts generate --config <ckpt>/config.yaml --voice voice.wav --text "Hello."
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

from models.pocket_tts.config import flowlm_args_from_raw  # noqa: E402
from models.pocket_tts.distill import (  # noqa: E402
    build_distillation, distill_loss, teacher_target, trainable_count,
)
from models.pocket_tts.model import FlowLM  # noqa: E402
from models.pocket_tts.teacher_cache import write_cache  # noqa: E402
from models.pocket_tts.train_utils import (  # noqa: E402
    TRAIN_DEFAULTS, build_datasets, merge, normalize, synthetic_batch, synthetic_manifest,
    time_steps, tiny_raw, train_loop,
)
from models.pocket_tts.weights import model_state  # noqa: E402

DEFAULTS = merge(TRAIN_DEFAULTS, {"distill": {"cfg_coef": 2.0},
                                  "train": {"run_dir": "checkpoints/pocket-distill"}})


def smoke_setup(tmp: Path) -> dict:
    """Tiny random teacher (8 layers, z_dim 32) on disk + synthetic latent data."""
    raw = tiny_raw()
    teacher_cfg = tmp / "teacher.yaml"
    teacher_cfg.write_text(yaml.safe_dump(raw))
    mx.random.seed(0)
    teacher = FlowLM(flowlm_args_from_raw(raw))
    rng = np.random.default_rng(0)
    state = {k: mx.array((rng.normal(size=v.shape) * 0.1).astype(np.float32)
                         + (1.0 if k.endswith(("norm1.weight", "norm2.weight", "alpha")) else 0.0))
             for k, v in model_state(teacher).items() if not k.endswith(("freqs", "emb_std", "emb_mean"))}
    teacher.load_weights(list(state.items()), strict=False)
    tensors = {f"flow_lm.{k}": v for k, v in model_state(teacher).items()}
    tensors["mimi.placeholder"] = mx.zeros((1,))
    mx.save_safetensors(str(tmp / "teacher.safetensors"), tensors)

    student = tiny_raw(layers=2, d_model=96, attn_dim=64, heads=2, num_kv_heads=1)
    student["flow_lm"]["lookup_table"]["rank"] = 32
    (tmp / "student.yaml").write_text(yaml.safe_dump(student))
    manifest = synthetic_manifest()
    return {
        "teacher": {"config": str(teacher_cfg), "weights": str(tmp / "teacher.safetensors")},
        "student": {"config": str(tmp / "student.yaml")},
        "data": {"train": manifest, "valid": manifest[:8], "tokenizer": lambda s: [ord(c) % 64 for c in s]},
        "optim": {"lr": 2e-3, "warmup_steps": 10},
        "train": {"batch_size": 8, "grad_accum": 1, "max_steps": 120, "log_freq": 20,
                  "valid_freq": 60, "num_valid_batches": 1, "ckpt_freq": 120, "ema_decay": 0.99,
                  "run_dir": str(tmp / "run")},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, metavar="STEPS",
                   help="time STEPS optimizer steps on synthetic batches (no data needed)")
    p.add_argument("--bench-sec", type=float, default=12.0, help="target seconds per benchmark row")
    p.add_argument("--cache-teacher", metavar="DIR", default=None,
                   help="write the teacher's targets for data.train to DIR and exit")
    p.add_argument("--cache-epochs", type=int, default=1, help="copies with different prompt cuts")
    p.add_argument("--benchmark-cached", action="store_true",
                   help="with --benchmark: time a step that reads cached targets (student only)")
    args = p.parse_args()

    tmp = None
    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp(prefix="pocket_smoke_"))
        user = smoke_setup(tmp)
    else:
        if not args.config:
            p.error("--config is required (or --smoke-test)")
        user = yaml.safe_load(Path(args.config).read_text())
    cfg = merge(DEFAULTS, user)
    t = cfg["train"]
    if args.max_steps:
        t["max_steps"] = args.max_steps
    mx.random.seed(t["seed"])

    teacher, student, raw, kept = build_distillation(cfg)
    cfg_coef = float(cfg["distill"]["cfg_coef"])
    print(f"[distill] teacher {teacher.args.num_layers}L d={teacher.args.d_model} "
          f"z={teacher.args.z_width}  ->  student {student.args.num_layers}L d={student.args.d_model} "
          f"(seeded from teacher layers {kept})")
    print(f"[distill] trainable student params: {trainable_count(student) / 1e6:.2f}M  cfg_coef={cfg_coef}")

    def loss_fn(model, batch, training):
        batch = normalize(batch, teacher)
        target = batch.get("target")
        if target is None:  # live teacher
            target = teacher_target(teacher, batch, cfg_coef)
            mx.eval(target)
        loss = distill_loss(model, batch, target)
        return loss, {}

    if args.benchmark:
        batch = synthetic_batch(teacher, t["batch_size"], args.bench_sec)
        if args.benchmark_cached:
            batch["target"] = teacher_target(teacher, normalize(batch, teacher), cfg_coef)
            mx.eval(batch["target"])
        time_steps(loss_fn, student, batch, args.benchmark, cfg)
        return

    if args.cache_teacher:
        d = cfg["data"]
        ds = build_datasets({**d, "teacher_cache": None}, t, student.args.tokenizer_path,
                            datasets_only=True)[0]
        n = write_cache(teacher, ds, args.cache_teacher, cfg_coef, t["batch_size"], args.cache_epochs)
        print(f"[cache] {n} rows -> {args.cache_teacher}; set data.teacher_cache to use it")
        return

    train_iter, valid_ds = build_datasets(cfg["data"], t, student.args.tokenizer_path,
                                          expect_cfg_coef=cfg_coef)
    out = train_loop(cfg, student, student, raw, loss_fn, train_iter, valid_ds,
                     cfg["teacher"]["weights"], args.resume)

    if args.smoke_test:
        ok = out["last_loss"] < 0.5 * out["first_loss"] and math.isfinite(out["last_loss"])
        print(f"[smoke] loss {out['first_loss']:.4f} -> {out['last_loss']:.4f}: {'OK' if ok else 'FAILED'}")
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
