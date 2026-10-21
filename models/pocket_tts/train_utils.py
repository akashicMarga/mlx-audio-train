"""Shared pieces of the Pocket TTS training scripts (distillation and fine-tuning)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from models.pocket_tts.weights import export_pocket_tts, model_state


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


def build_optimizer(o: dict, max_steps: int):
    sched = lr_schedule(o, max_steps)
    return optim.AdamW(learning_rate=sched, betas=o["betas"], eps=o["eps"],
                       weight_decay=o["weight_decay"]), sched


def normalize(batch: dict, model) -> dict:
    batch = dict(batch)
    batch["latents"] = (batch["latents"] - model.emb_mean) / model.emb_std
    return batch


class EMA:
    """EMA of a FlowLM's trainable tensors (exported weights use the shadow)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}

    def update(self, model: nn.Module):
        d = self.decay
        for k, v in tree_flatten(model.trainable_parameters()):
            self.shadow[k] = d * self.shadow[k] + (1 - d) * v


def accumulate_and_step(loss_and_grad, trainable: nn.Module, opt, micro_batches, max_norm: float,
                        ema: EMA | None = None, ema_model: nn.Module | None = None):
    """One optimizer step over micro-batches; loss_and_grad(trainable, batch) -> ((loss, metrics), grads)."""
    from mlx.utils import tree_map

    grads_sum, losses, metrics = None, [], {}
    for batch in micro_batches:
        (loss, m), grads = loss_and_grad(trainable, batch)
        grads_sum = grads if grads_sum is None else tree_map(mx.add, grads_sum, grads)
        mx.eval(grads_sum, loss, m)
        losses.append(loss.item())
        for k, v in m.items():
            metrics.setdefault(k, []).append(float(v.item() if isinstance(v, mx.array) else v))
    grads = tree_map(lambda g: g / len(micro_batches), grads_sum)
    grads, gnorm = optim.clip_grad_norm(grads, max_norm)
    opt.update(trainable, grads)
    if ema:
        ema.update(ema_model if ema_model is not None else trainable)
    mx.eval(trainable.parameters(), opt.state, *(ema.shadow.values() if ema else ()))
    return sum(losses) / len(losses), gnorm.item(), {k: sum(v) / len(v) for k, v in metrics.items()}


def save_checkpoint(run_dir: Path, step: int, trainable: nn.Module, flow_lm, raw: dict,
                    ema: EMA | None, opt, mimi_source: str, keep: int) -> Path:
    """run_dir/step_N/: model.safetensors + config.yaml (official layout, EMA weights) and
    the raw training state for --resume."""
    ckpt = run_dir / f"step_{step:07d}"
    export_pocket_tts(flow_lm, raw, mimi_source, ckpt, params=ema.shadow if ema else None)
    mx.save_safetensors(str(ckpt / "trainable_raw.safetensors"), model_state(trainable))
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


def load_checkpoint(ckpt: Path, trainable: nn.Module, ema: EMA | None, opt) -> int:
    trainable.load_weights(str(ckpt / "trainable_raw.safetensors"), strict=True)
    if ema and (ckpt / "ema.safetensors").exists():
        ema.shadow = dict(mx.load(str(ckpt / "ema.safetensors")))
    if (ckpt / "optimizer.safetensors").exists():
        opt.init(trainable.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(ckpt / "optimizer.safetensors")).items()))
    return json.loads((ckpt / "state.json").read_text())["step"]


class JsonlLog:
    """train_log.jsonl in the format scripts/watch_training.py reads (loss / val_loss / lr)."""

    def __init__(self, path: Path):
        self.f = open(path, "a")

    def __call__(self, entry: dict, prefix: str = ""):
        print(f"  {prefix}" + "  ".join(f"{k}={v}" for k, v in entry.items()), flush=True)
        self.f.write(json.dumps(entry) + "\n")
        self.f.flush()


TRAIN_DEFAULTS = {
    "data": {"max_duration_sec": 30.0, "max_voice_prompt_sec": 5.0, "prompt_trim_max_sec": 0.5,
             "final_punct_dropout": 0.3, "bucket_batches": 5, "normalize_text": None,
             "teacher_cache": None},
    "optim": {"lr": 4e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8,
              "max_norm": 1.0, "warmup_steps": 1000, "schedule": "cosine", "lr_min_ratio": 0.0},
    "train": {"batch_size": 16, "grad_accum": 4, "max_steps": 200000, "ema_decay": 0.9999,
              "log_freq": 50, "valid_freq": 2500, "num_valid_batches": 20, "ckpt_freq": 2500,
              "keep_last_n": 3, "seed": 42, "run_dir": "checkpoints/pocket"},
}


def build_datasets(d: dict, t: dict, tokenizer_path: str, datasets_only: bool = False,
                   expect_cfg_coef: float | None = None):
    """(train batch iterator, valid dataset or None) from the data section.

    data.teacher_cache (distillation only) replaces data.train with precomputed targets.
    datasets_only returns (train dataset, valid dataset) instead of an iterator."""
    from models.pocket_tts.data import LatentDataset, batches, load_tokenizer
    from models.pocket_tts.text import normalizer

    tok = d.get("tokenizer") or load_tokenizer(d.get("tokenizer_path") or tokenizer_path)
    norm = normalizer(d.get("normalize_text"))
    tokenize = (lambda s: tok(norm(s))) if norm else tok
    kw = {k: d[k] for k in ("max_duration_sec", "max_voice_prompt_sec", "prompt_trim_max_sec",
                            "final_punct_dropout")}
    if d.get("teacher_cache"):
        from models.pocket_tts.teacher_cache import CachedDataset

        train_ds = CachedDataset(d["teacher_cache"], expect_cfg_coef)
    else:
        train_ds = LatentDataset(d["train"], tokenize, seed=t["seed"], **kw)
    valid_ds = LatentDataset(d["valid"], tokenize, seed=1, **{**kw, "final_punct_dropout": 0.0}) \
        if d.get("valid") else None
    print(f"[data] {len(train_ds)} train rows" + (" (cached teacher targets)" if d.get("teacher_cache") else "")
          + (f", {len(valid_ds)} valid" if valid_ds else ""))
    if datasets_only:
        return train_ds, valid_ds
    return batches(train_ds, t["batch_size"], d["bucket_batches"], seed=t["seed"]), valid_ds


def train_loop(cfg: dict, trainable: nn.Module, flow_lm, raw: dict, loss_fn, train_iter,
               valid_ds, mimi_source: str, resume: str | None = None, before_step=None) -> dict:
    """Generic loop: loss_fn(trainable, batch, training) -> (loss, metrics dict).

    flow_lm is the FlowLM inside `trainable` that gets EMA'd and exported.
    before_step(step, micro_batches) runs ahead of each optimizer step (e.g. latent stats).
    Returns {"first_loss", "last_loss", "step"}.
    """
    import time

    import numpy as np
    from models.pocket_tts.data import batches

    o, t = cfg["optim"], cfg["train"]
    opt, sched = build_optimizer(o, t["max_steps"])
    ema = EMA(flow_lm, t["ema_decay"]) if t["ema_decay"] > 0 else None
    loss_and_grad = nn.value_and_grad(trainable, lambda m, b: loss_fn(m, b, True))

    run_dir = Path(t["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    plain = {k: v for k, v in cfg.items() if k != "data"}
    plain["data"] = {k: v for k, v in cfg["data"].items() if isinstance(v, (str, int, float, bool))}
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(plain, sort_keys=False))
    step = load_checkpoint(Path(resume), trainable, ema, opt) if resume else 0
    log = JsonlLog(run_dir / "train_log.jsonl")

    def validate() -> float:
        vals = []
        for i, batch in enumerate(batches(valid_ds, t["batch_size"], 1, shuffle=False)):
            if i >= t["num_valid_batches"]:
                break
            mx.random.seed(i)  # same noise / times every validation
            vals.append(loss_fn(trainable, batch, False)[0].item())
        return float(np.mean(vals)) if vals else float("nan")

    first = last = None
    t0, frames = time.perf_counter(), 0
    while step < t["max_steps"]:
        micro = [next(train_iter) for _ in range(t["grad_accum"])]
        if before_step:
            before_step(step, micro)
        last, gnorm, metrics = accumulate_and_step(loss_and_grad, trainable, opt, micro,
                                                   o["max_norm"], ema, flow_lm)
        step += 1
        frames += sum(b["frames"] for b in micro)
        first = last if first is None else first
        if step % t["log_freq"] == 0 or step == 1:
            dt = time.perf_counter() - t0
            entry = {"step": step, "loss": round(last, 6), "grad_norm": round(gnorm, 4),
                     "lr": float(sched(mx.array(step)))}
            entry.update({k: round(v, 5) for k, v in metrics.items()})
            if step > 1:
                entry["steps_per_sec"] = round(t["log_freq"] / dt, 3)
                entry["audio_sec_per_sec"] = round(frames / 12.5 / dt, 1)
            log(entry)
            t0, frames = time.perf_counter(), 0
        if valid_ds is not None and step % t["valid_freq"] == 0:
            log({"step": step, "val_loss": round(validate(), 6)}, prefix="[valid] ")
        if step % t["ckpt_freq"] == 0 or step == t["max_steps"]:
            ckpt = save_checkpoint(run_dir, step, trainable, flow_lm, raw, ema, opt, mimi_source,
                                   t["keep_last_n"])
            print(f"  [ckpt] {ckpt}", flush=True)
    return {"first_loss": first, "last_loss": last, "step": step}


def time_steps(loss_fn, trainable: nn.Module, batch: dict, steps: int, cfg: dict):
    """--benchmark: seconds per optimizer step on a fixed synthetic batch."""
    import time

    o, t = cfg["optim"], cfg["train"]
    opt, _ = build_optimizer(o, t["max_steps"])
    loss_and_grad = nn.value_and_grad(trainable, lambda m, b: loss_fn(m, b, True))
    accumulate_and_step(loss_and_grad, trainable, opt, [batch], o["max_norm"])  # warm-up
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(steps):
        accumulate_and_step(loss_and_grad, trainable, opt, [batch], o["max_norm"])
    dt = (time.perf_counter() - t0) / steps
    B, sec = batch["latents"].shape[0], (batch["latents"].shape[1] - 1) / 12.5
    accum = t["grad_accum"]
    print(f"[benchmark] micro-batch {B} x {sec:.0f}s audio: {dt:.3f} s/step "
          f"({B * sec / dt:.0f} s of audio/s), peak mem {mx.get_peak_memory() / 2**30:.1f} GiB")
    print(f"[benchmark] effective batch {B * accum}: {dt * accum:.2f} s/step -> "
          f"15k steps ~{dt * accum * 1.5e4 / 3600:.1f} h, 50k ~{dt * accum * 5e4 / 3600:.0f} h, "
          f"200k ~{dt * accum * 2e5 / 3600:.0f} h")


def synthetic_batch(flow_lm, B: int, sec: float, voice_sec: float = 4.0) -> dict:
    import numpy as np
    from models.pocket_tts.data import collate

    a = flow_lm.args
    rng = np.random.default_rng(0)
    T, Tv = int(sec * 12.5), int(voice_sec * 12.5)
    return collate([(rng.normal(size=(T, a.ldim)).astype(np.float32),
                     rng.integers(0, a.n_bins, size=int(sec * 4)).astype(np.int32),
                     rng.normal(size=(Tv, a.voice_dim)).astype(np.float32)) for _ in range(B)])


def tiny_raw(layers: int = 8, d_model: int = 128, z_dim: int | None = 32, n_bins: int = 64,
             heads: int = 4, **transformer) -> dict:
    """A small pocket-tts config for smoke tests (no downloads)."""
    return {
        "flow_lm": {"dtype": "float32", "insert_bos_before_voice": True, "z_dim": z_dim,
                    "flow": {"dim": 64, "depth": 2, "type": "lsd"},
                    "transformer": {"d_model": d_model, "hidden_scale": 2, "max_period": 10000,
                                    "num_heads": heads, "num_layers": layers, **transformer},
                    "lookup_table": {"dim": d_model, "n_bins": n_bins, "tokenizer": "sentencepiece",
                                     "tokenizer_path": "unused"}},
        "mimi": {"inner_dim": 32, "seanet": {"dimension": 64}, "quantizer": {"dimension": 32}},
    }


def synthetic_manifest(n: int = 64, seed: int = 0) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    # Smooth random walks: learnable structure, unlike white noise.
    return [{"latents": np.cumsum(rng.normal(size=(int(k), 32)) * 0.3, axis=0).astype(np.float32),
             "text": "x" * int(rng.integers(5, 30))} for k in rng.integers(20, 60, size=n)]
