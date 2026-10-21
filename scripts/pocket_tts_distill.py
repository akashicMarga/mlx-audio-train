#!/usr/bin/env python3
"""
Pocket TTS distillation in MLX: a frozen teacher (e.g. the public 24-layer English
model) -> a shallower / narrower student that reproduces the teacher's backbone
output z with classifier-free guidance baked in. See models/pocket_tts/distill.py
and docs/pocket_tts_distillation.md.

Usage:
    # 0. no-download sanity check (tiny random teacher, synthetic data)
    python scripts/pocket_tts_distill.py --smoke-test

    # 1. how fast would a real run be on this Mac? (synthetic batches, real model sizes)
    python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml --benchmark 20

    # 2. train
    python scripts/pocket_tts_distill.py --config configs/pocket_tts/distill_24l_to_6l.yaml

    # resume
    python scripts/pocket_tts_distill.py --config ... --resume checkpoints/pocket-distill-6l/step_0010000

Every checkpoint directory holds model.safetensors + config.yaml in the official
layout (EMA weights), runnable with:
    pocket-tts generate --config <ckpt>/config.yaml --voice voice.wav --text "Hello."
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.config import flowlm_args_from_raw  # noqa: E402
from models.pocket_tts.data import LatentDataset, batches, collate, load_tokenizer  # noqa: E402
from models.pocket_tts.distill import (  # noqa: E402
    build_distillation, distill_loss, teacher_target, trainable_count,
)
from models.pocket_tts.model import FlowLM  # noqa: E402
from models.pocket_tts.weights import export_pocket_tts, model_state  # noqa: E402

DEFAULTS = {
    "distill": {"cfg_coef": 2.0},
    "data": {"max_duration_sec": 30.0, "max_voice_prompt_sec": 5.0, "prompt_trim_max_sec": 0.5,
             "final_punct_dropout": 0.3, "bucket_batches": 5},
    "optim": {"lr": 4e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8,
              "max_norm": 1.0, "warmup_steps": 1000, "schedule": "cosine", "lr_min_ratio": 0.0},
    "train": {"batch_size": 16, "grad_accum": 4, "max_steps": 200000, "ema_decay": 0.9999,
              "log_freq": 50, "valid_freq": 2500, "num_valid_batches": 20, "ckpt_freq": 2500,
              "keep_last_n": 3, "seed": 42, "run_dir": "checkpoints/pocket-distill"},
}


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def lr_schedule(o: dict, max_steps: int):
    warm = optim.linear_schedule(1e-3 * o["lr"], o["lr"], max(o["warmup_steps"], 1))
    if o["schedule"] == "constant":
        main = lambda _: o["lr"]  # noqa: E731
    else:
        main = optim.cosine_decay(o["lr"], max(max_steps - o["warmup_steps"], 1), o["lr"] * o["lr_min_ratio"])
    return optim.join_schedules([warm, main], [o["warmup_steps"]])


def normalize(batch: dict, teacher: FlowLM) -> dict:
    batch = dict(batch)
    batch["latents"] = (batch["latents"] - teacher.emb_mean) / teacher.emb_std
    return batch


class EMA:
    """EMA of the trainable student tensors (exported weights use the shadow)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}

    def update(self, model: nn.Module):
        d = self.decay
        for k, v in tree_flatten(model.trainable_parameters()):
            self.shadow[k] = d * self.shadow[k] + (1 - d) * v


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(run_dir: Path, step: int, student, raw, ema, opt, mimi_source, keep: int):
    ckpt = run_dir / f"step_{step:07d}"
    export_pocket_tts(student, raw, mimi_source, ckpt, params=ema.shadow if ema else None)
    mx.save_safetensors(str(ckpt / "student_raw.safetensors"), model_state(student))
    if ema:
        mx.save_safetensors(str(ckpt / "ema.safetensors"), ema.shadow)
    opt_state = dict(tree_flatten(opt.state))
    mx.save_safetensors(str(ckpt / "optimizer.safetensors"),
                        {k: v for k, v in opt_state.items() if isinstance(v, mx.array)})
    (ckpt / "state.json").write_text(json.dumps({"step": step}))
    old = sorted(run_dir.glob("step_*"))
    for d in old[:-keep] if keep else []:
        shutil.rmtree(d, ignore_errors=True)
    return ckpt


def load_checkpoint(ckpt: Path, student, ema, opt) -> int:
    student.load_weights(str(ckpt / "student_raw.safetensors"), strict=True)
    if ema and (ckpt / "ema.safetensors").exists():
        ema.shadow = dict(mx.load(str(ckpt / "ema.safetensors")))
    if (ckpt / "optimizer.safetensors").exists():
        opt.init(student.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(ckpt / "optimizer.safetensors")).items()))
    return json.loads((ckpt / "state.json").read_text())["step"]


# ── Smoke test / benchmark helpers ────────────────────────────────────────────

def smoke_setup(tmp: Path) -> dict:
    """Tiny random teacher (8 layers, z_dim 32) on disk + synthetic latent data."""
    raw = {
        "flow_lm": {"dtype": "float32", "insert_bos_before_voice": True, "z_dim": 32,
                    "flow": {"dim": 64, "depth": 2, "type": "lsd"},
                    "transformer": {"d_model": 128, "hidden_scale": 2, "max_period": 10000,
                                    "num_heads": 4, "num_layers": 8},
                    "lookup_table": {"dim": 128, "n_bins": 64, "tokenizer": "sentencepiece",
                                     "tokenizer_path": "unused"}},
        "mimi": {"inner_dim": 32, "seanet": {"dimension": 64}, "quantizer": {"dimension": 32}},
    }
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

    student = merge(raw, {"flow_lm": {"transformer": {"num_layers": 2, "d_model": 96,
                                                      "attn_dim": 64, "num_heads": 2, "num_kv_heads": 1},
                                      "lookup_table": {"dim": 96, "rank": 32}}})
    (tmp / "student.yaml").write_text(yaml.safe_dump(student))
    manifest = [{"latents": rng.normal(size=(int(n), 32)).astype(np.float32),
                 "text": "x" * int(rng.integers(5, 30))} for n in rng.integers(20, 60, size=64)]
    return {
        "teacher": {"config": str(teacher_cfg), "weights": str(tmp / "teacher.safetensors")},
        "student": {"config": str(tmp / "student.yaml")},
        "data": {"train": manifest, "valid": manifest[:8], "tokenizer": lambda s: [ord(c) % 64 for c in s]},
        "optim": {"lr": 2e-3, "warmup_steps": 10},
        "train": {"batch_size": 8, "grad_accum": 1, "max_steps": 120, "log_freq": 20,
                  "valid_freq": 60, "num_valid_batches": 1, "ckpt_freq": 120, "ema_decay": 0.99,
                  "run_dir": str(tmp / "run")},
    }


def synthetic_batch(teacher: FlowLM, B: int, sec: float, text_tokens: int, voice_sec: float) -> dict:
    T = int(sec * 12.5)
    Tv = int(voice_sec * 12.5)
    a = teacher.args
    rng = np.random.default_rng(0)
    return collate([(rng.normal(size=(T, a.ldim)).astype(np.float32),
                     rng.integers(0, a.n_bins, size=text_tokens).astype(np.int32),
                     rng.normal(size=(Tv, a.voice_dim)).astype(np.float32)) for _ in range(B)])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, metavar="STEPS",
                   help="time STEPS optimizer steps on synthetic batches (no data needed)")
    p.add_argument("--bench-sec", type=float, default=12.0, help="target seconds per benchmark row")
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
    o, t, d = cfg["optim"], cfg["train"], cfg["data"]
    if args.max_steps:
        t["max_steps"] = args.max_steps
    mx.random.seed(t["seed"])

    teacher, student, raw, kept = build_distillation(cfg)
    print(f"[distill] teacher {teacher.args.num_layers}L d={teacher.args.d_model} "
          f"z={teacher.args.z_width}  ->  student {student.args.num_layers}L d={student.args.d_model} "
          f"(seeded from teacher layers {kept})")
    print(f"[distill] trainable student params: {trainable_count(student) / 1e6:.2f}M  "
          f"cfg_coef={cfg['distill']['cfg_coef']}")

    sched = lr_schedule(o, t["max_steps"])
    opt = optim.AdamW(learning_rate=sched, betas=o["betas"], eps=o["eps"], weight_decay=o["weight_decay"])
    ema = EMA(student, t["ema_decay"]) if t["ema_decay"] > 0 else None
    loss_and_grad = nn.value_and_grad(student, distill_loss)
    cfg_coef = float(cfg["distill"]["cfg_coef"])

    def micro_step(batch):
        batch = normalize(batch, teacher)
        target = teacher_target(teacher, batch, cfg_coef)
        mx.eval(target)
        return loss_and_grad(student, batch, target)

    def optimizer_step(micro_batches):
        grads_sum, losses = None, []
        for batch in micro_batches:
            loss, grads = micro_step(batch)
            grads_sum = grads if grads_sum is None else tree_map(mx.add, grads_sum, grads)
            mx.eval(grads_sum, loss)
            losses.append(loss.item())
        grads = tree_map(lambda g: g / len(micro_batches), grads_sum)
        grads, gnorm = optim.clip_grad_norm(grads, o["max_norm"])
        opt.update(student, grads)
        if ema:
            ema.update(student)
        mx.eval(student.parameters(), opt.state, *(ema.shadow.values() if ema else ()))
        return float(np.mean(losses)), gnorm.item()

    if args.benchmark:
        B = t["batch_size"]
        batch = synthetic_batch(teacher, B, args.bench_sec, int(args.bench_sec * 4), 4.0)
        optimizer_step([batch])  # warm-up
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for _ in range(args.benchmark):
            optimizer_step([batch])
        dt = (time.perf_counter() - t0) / args.benchmark
        eff = B * t["grad_accum"]
        print(f"[benchmark] micro-batch {B} x {args.bench_sec:.0f}s audio: {dt:.3f} s/step  "
              f"({B * args.bench_sec / dt:.0f} s of audio/s), peak mem {mx.get_peak_memory() / 2**30:.1f} GiB")
        print(f"[benchmark] effective batch {eff}: {dt * t['grad_accum']:.2f} s/step -> "
              f"50k steps ~{dt * t['grad_accum'] * 5e4 / 3600:.0f} h, "
              f"200k steps ~{dt * t['grad_accum'] * 2e5 / 3600:.0f} h")
        return

    tok = d.get("tokenizer") or load_tokenizer(d.get("tokenizer_path") or student.args.tokenizer_path)
    ds_kw = {k: d[k] for k in ("max_duration_sec", "max_voice_prompt_sec", "prompt_trim_max_sec",
                               "final_punct_dropout")}
    train_ds = LatentDataset(d["train"], tok, seed=t["seed"], **ds_kw)
    valid_ds = LatentDataset(d["valid"], tok, seed=1, **{**ds_kw, "final_punct_dropout": 0.0}) \
        if d.get("valid") else None
    train_iter = batches(train_ds, t["batch_size"], d["bucket_batches"], seed=t["seed"])
    print(f"[distill] {len(train_ds)} train rows, batch {t['batch_size']} x accum {t['grad_accum']}")

    run_dir = Path(t["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "distill_config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "data"} | {"data": {
            k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}}, sort_keys=False))
    step = load_checkpoint(Path(args.resume), student, ema, opt) if args.resume else 0
    log = open(run_dir / "train_log.jsonl", "a")
    mimi_source = cfg["teacher"]["weights"]

    def validate() -> float:
        vals = []
        for i, batch in enumerate(batches(valid_ds, t["batch_size"], 1, shuffle=False)):
            if i >= t["num_valid_batches"]:
                break
            batch = normalize(batch, teacher)
            target = teacher_target(teacher, batch, cfg_coef)
            vals.append(distill_loss(student, batch, target).item())
        return float(np.mean(vals)) if vals else float("nan")

    first_loss = None
    t0, frames = time.perf_counter(), 0
    while step < t["max_steps"]:
        micro = [next(train_iter) for _ in range(t["grad_accum"])]
        loss, gnorm = optimizer_step(micro)
        step += 1
        frames += sum(b["frames"] for b in micro)
        first_loss = first_loss if first_loss is not None else loss
        if step % t["log_freq"] == 0 or step == 1:
            dt = time.perf_counter() - t0
            entry = {"step": step, "loss": round(loss, 6), "grad_norm": round(gnorm, 4),
                     "lr": float(sched(mx.array(step))), "steps_per_sec": round(t["log_freq"] / dt, 3)
                     if step > 1 else None, "audio_sec_per_sec": round(frames / 12.5 / dt, 1)}
            print("  " + "  ".join(f"{k}={v}" for k, v in entry.items()), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            t0, frames = time.perf_counter(), 0
        if valid_ds and step % t["valid_freq"] == 0:
            entry = {"step": step, "val_loss": round(validate(), 6)}
            print(f"  [valid] {entry}", flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
        if step % t["ckpt_freq"] == 0 or step == t["max_steps"]:
            ckpt = save_checkpoint(run_dir, step, student, raw, ema, opt, mimi_source, t["keep_last_n"])
            print(f"  [ckpt] {ckpt}", flush=True)

    if args.smoke_test:
        ok = loss < 0.5 * first_loss and math.isfinite(loss)
        print(f"[smoke] loss {first_loss:.4f} -> {loss:.4f}: {'OK' if ok else 'FAILED'}")
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
