"""
trainer.py — Base training loop for all mlx-audio models.

Features:
  - AdamW optimizer with linear warmup + cosine decay
  - Gradient accumulation
  - Gradient clipping
  - Per-step + per-epoch checkpointing (saves LoRA adapters only)
  - Live metrics logging (console + optional JSON log)
  - Works with ANY model + loss_fn combo

Usage:
    trainer = Trainer(config)
    trainer.train(model, train_loader, val_loader, loss_fn)
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mxu
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    # I/O
    output_dir:        str   = "./checkpoints"
    run_name:          str   = "run"

    # Training
    num_epochs:        int   = 10
    batch_size:        int   = 4
    grad_accumulation: int   = 4      # effective batch = batch_size * grad_accumulation
    max_steps:         Optional[int] = None

    # Optimizer
    learning_rate:     float = 2e-4
    weight_decay:      float = 0.01
    grad_clip:         float = 1.0

    # LR schedule
    warmup_steps:      int   = 100
    lr_schedule:       str   = "cosine"   # "cosine" | "constant" | "linear"

    # Checkpointing
    save_every_n_steps:  int = 200
    save_every_n_epochs: int = 1
    keep_last_n:         int = 3

    # Validation
    eval_every_n_steps:  int = 100
    val_batches:         int = 20      # number of val batches per eval

    # Logging
    log_every_n_steps:   int = 10
    log_file:            Optional[str] = None   # JSON log path

    # Label smoothing
    label_smoothing:     float = 0.0

    # ── Memory optimizations ──────────────────────────────────────────────────
    # Cast inputs_embeds to bfloat16 before the transformer forward pass.
    # Halves attention-matrix (O(T²)) and hidden-state memory across all layers.
    # Requires lora_dtype: "bfloat16" in the lora config for full bf16 training.
    # Default: false (float32, more stable). Enable for 64 GB machines with long seqs.
    use_bf16:            bool  = False

    # Clear the Metal GPU buffer pool every N optimizer steps (0 = disabled).
    # Metal caches allocated buffers for reuse; over time this pool can grow to
    # several GB. Clearing it frees unused Metal memory at the cost of
    # re-allocating buffers on the next step. Useful when running OOM near the
    # end of a long training run.
    # Suggested: clear_cache_steps=50 if you see memory creeping up over time.
    clear_cache_steps:   int   = 0


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(
    step:        int,
    total_steps: int,
    base_lr:     float,
    warmup:      int,
    schedule:    str,
) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)

    if schedule == "constant":
        return base_lr

    progress = (step - warmup) / max(1, total_steps - warmup)

    if schedule == "cosine":
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    if schedule == "linear":
        return base_lr * (1 - progress)

    return base_lr


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._step         = 0
        self._epoch        = 0
        self._best_val     = float("inf")
        self._log_history: List[Dict] = []
        self._saved_ckpts: List[str]  = []

        # JSON log
        self._log_fh = None
        if config.log_file:
            self._log_fh = open(config.log_file, "w")

        # Save config
        with open(self.output_dir / "trainer_config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)

    # ── Main entry ────────────────────────────────────────────────────────

    def train(
        self,
        model:        nn.Module,
        train_loader,
        loss_fn:      Callable,
        val_loader    = None,
    ):
        """
        Full training loop.

        Args:
            model:        MLX nn.Module (with LoRA applied)
            train_loader: iterable of batch dicts
            loss_fn:      fn(model, batch) → (loss_scalar, metrics_dict)
            val_loader:   optional iterable for validation
        """
        cfg = self.cfg

        # Count total steps — use len() which is just arithmetic, never materialises data
        steps_per_epoch = max(1, len(train_loader) // cfg.grad_accumulation)
        total_steps     = steps_per_epoch * cfg.num_epochs
        if cfg.max_steps:
            total_steps = min(total_steps, cfg.max_steps)

        print(f"\n{'='*60}")
        print(f"  Training: {cfg.run_name}")
        print(f"  Epochs:   {cfg.num_epochs}  |  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}  |  Effective batch: {cfg.batch_size * cfg.grad_accumulation}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

        # Identify LoRA params — only these will be updated by optimizer
        from .lora import get_trainable_params
        lora_params = get_trainable_params(model)
        print(f"[trainer] {len(lora_params)} LoRA adapter tensors will be optimized")
        if cfg.use_bf16:
            print("[trainer] bf16 activations ENABLED — inputs_embeds cast to bfloat16 before transformer")
        if cfg.clear_cache_steps > 0:
            print(f"[trainer] Metal cache cleared every {cfg.clear_cache_steps} optimizer steps")

        optimizer = optim.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        state = [model.state, optimizer.state]

        # Custom value_and_grad that strips empty-dict subtrees before
        # model.update(). The Qwen3-TTS speech_tokenizer contains a gc_func
        # at .decoder; even after freeze(), trainable_parameters() returns
        # an empty nested skeleton for it, and model.update() crashes trying
        # to do `k in gc_func`. Stripping empty subtrees avoids that path.
        def _is_empty(x):
            """True if x is a dict/list with no mx.array leaves."""
            if isinstance(x, dict):
                return all(_is_empty(v) for v in x.values()) if x else True
            if isinstance(x, list):
                return all(_is_empty(v) for v in x) if x else True
            return False  # it's a tensor (leaf) — not empty

        def _strip_empty(d):
            if isinstance(d, dict):
                out = {}
                for k, v in d.items():
                    s = _strip_empty(v)
                    if not _is_empty(s):
                        out[k] = s
                return out
            if isinstance(d, list):
                out = [_strip_empty(v) for v in d]
                out = [v for v in out if not _is_empty(v)]
                return out
            return d  # mx.array leaf

        def _inner_fn(params, batch):
            model.update(params)
            return loss_fn(model, batch)

        _raw_vg = mx.value_and_grad(_inner_fn)

        def value_and_grad_fn(model, batch):
            # BUG FIX: use get_trainable_params (lora_a/lora_b only = 5.9M params)
            # NOT model.trainable_parameters() which returns 351.8M params and causes
            # MLX to store intermediate activations for ALL base weights → 40+ GB RAM.
            flat_lora = get_trainable_params(model)          # flat {path: tensor}
            params    = mxu.tree_unflatten(list(flat_lora.items()))  # nested for model.update
            (loss, metrics), grads = _raw_vg(params, batch)
            return (loss, metrics), _strip_empty(grads)

        accum_grads:  Dict = {}
        accum_loss   = 0.0
        accum_count  = 0

        t0 = time.time()

        for epoch in range(cfg.num_epochs):
            self._epoch = epoch
            epoch_loss  = 0.0
            epoch_steps = 0

            # Iterate lazily — never holds more than one batch in memory at a time
            for batch_idx, batch in enumerate(train_loader):
                if cfg.max_steps and self._step >= cfg.max_steps:
                    break

                # ── Forward + backward ─────────────────────────────────────
                (loss, metrics), grads = value_and_grad_fn(model, batch)
                # Evaluate BOTH loss and grads immediately — if grads are left as
                # deferred MLX arrays across grad_accumulation steps, the computation
                # graph grows unboundedly and causes OOM (exit 137).
                mx.eval(loss, grads)

                # Gradient accumulation (grads is nested dict).
                # No extra mx.eval() needed here: grads are already fully
                # materialized by the call above, so each _add_grads step only
                # creates a single-depth (a+b) deferred op per tensor.  The
                # accumulated graph is evaluated once, at the optimizer step.
                if not accum_grads:
                    accum_grads = grads
                else:
                    accum_grads = _add_grads(accum_grads, grads)

                accum_loss  += float(loss)
                accum_count += 1

                if accum_count < cfg.grad_accumulation:
                    continue

                # ── Optimizer step ─────────────────────────────────────────
                # Clip gradients (operates on flat lora grads)
                flat_grads = dict(mxu.tree_flatten(accum_grads))
                flat_grads = _clip_grads(flat_grads, cfg.grad_clip)
                nested_grads = mxu.tree_unflatten(list(flat_grads.items()))

                # Update LR
                lr = get_lr(self._step, total_steps, cfg.learning_rate,
                            cfg.warmup_steps, cfg.lr_schedule)
                optimizer.learning_rate = lr

                optimizer.update(model, nested_grads)
                mx.eval(state)

                # Optional: free the Metal GPU buffer pool to reclaim cached
                # allocations that are no longer in use. Trades re-allocation
                # overhead for lower peak memory on long runs.
                if cfg.clear_cache_steps > 0 and self._step % cfg.clear_cache_steps == 0:
                    try:
                        mx.metal.clear_cache()
                    except AttributeError:
                        pass  # Not on Metal (non-Apple platform); silently skip

                step_loss = accum_loss / accum_count
                epoch_loss += step_loss
                epoch_steps += 1
                self._step += 1

                accum_grads  = {}
                accum_loss   = 0.0
                accum_count  = 0

                # ── Logging ────────────────────────────────────────────────
                if self._step % cfg.log_every_n_steps == 0:
                    elapsed = time.time() - t0
                    log = {
                        "step":    self._step,
                        "epoch":   epoch,
                        "loss":    round(step_loss, 5),
                        "lr":      round(lr, 8),
                        "elapsed": round(elapsed, 1),
                    }
                    log.update({k: round(v, 5) for k, v in metrics.items() if k != "loss"})
                    self._log(log)

                # ── Eval ───────────────────────────────────────────────────
                if val_loader and self._step % cfg.eval_every_n_steps == 0:
                    val_loss = self._evaluate(model, val_loader, loss_fn)
                    self._log({"step": self._step, "val_loss": round(val_loss, 5)})
                    if val_loss < self._best_val:
                        self._best_val = val_loss
                        self._save_checkpoint(model, tag="best")

                # ── Checkpoint ─────────────────────────────────────────────
                if self._step % cfg.save_every_n_steps == 0:
                    self._save_checkpoint(model, tag=f"step_{self._step:07d}")

            # ── End of epoch ───────────────────────────────────────────────
            if epoch_steps > 0:
                avg = epoch_loss / epoch_steps
                print(f"\n  Epoch {epoch+1}/{cfg.num_epochs}  avg_loss={avg:.5f}\n")

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(model, tag=f"epoch_{epoch+1:04d}")

        # Final checkpoint
        self._save_checkpoint(model, tag="final")
        print(f"\n✅ Training complete. Checkpoints at: {self.output_dir}")

        if self._log_fh:
            self._log_fh.close()

    # ── Evaluation ────────────────────────────────────────────────────────

    def _evaluate(self, model, val_loader, loss_fn) -> float:
        losses = []
        for i, batch in enumerate(val_loader):
            if i >= self.cfg.val_batches:
                break
            loss, _ = loss_fn(model, batch)
            mx.eval(loss)
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else 0.0

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self, model, tag: str):
        from .lora import save_adapters

        ckpt_dir  = self.output_dir / f"checkpoint-{tag}"
        ckpt_dir.mkdir(exist_ok=True)
        adapter_path = str(ckpt_dir / "adapters.safetensors")

        save_adapters(model, adapter_path)

        # Save step info
        with open(ckpt_dir / "info.json", "w") as f:
            json.dump({"step": self._step, "epoch": self._epoch, "tag": tag}, f)

        print(f"  💾  Saved checkpoint: {ckpt_dir.name}")

        # Rotate old checkpoints
        self._saved_ckpts.append(str(ckpt_dir))
        keep = self.cfg.keep_last_n
        if keep and len(self._saved_ckpts) > keep + 2:  # +2 keeps best + final
            old = self._saved_ckpts.pop(0)
            if "best" not in old and "final" not in old:
                import shutil
                shutil.rmtree(old, ignore_errors=True)

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, entry: Dict):
        # Console
        parts = [f"{k}={v}" for k, v in entry.items()]
        print("  " + "  ".join(parts))

        # JSON file
        if self._log_fh:
            self._log_fh.write(json.dumps(entry) + "\n")
            self._log_fh.flush()

        self._log_history.append(entry)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_grads(a, b):
    """Element-wise sum of two gradient trees (nested dicts/lists of mx.array)."""
    import mlx.core as _mx
    if isinstance(a, _mx.array) and isinstance(b, _mx.array):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for k in a:
            if k in b:
                result[k] = _add_grads(a[k], b[k])
            else:
                result[k] = a[k]
        for k in b:
            if k not in result:
                result[k] = b[k]
        return result
    if isinstance(a, list) and isinstance(b, list):
        return [_add_grads(x, y) for x, y in zip(a, b)]
    return a  # fallback


def _clip_grads(grads: Dict, max_norm: float) -> Dict:
    """Global gradient norm clipping.

    Batches all squared-norm computations into a single MLX op so only
    one GPU→CPU sync is needed instead of one per gradient tensor.
    """
    if max_norm <= 0:
        return grads

    # Stack all per-tensor squared norms and sum in one MLX operation.
    # Previously used `sum(float(mx.sum(g**2)) for g in ...)` which forced
    # a separate GPU→CPU sync for every LoRA tensor.
    sq_norms = [mx.sum(g ** 2) for g in grads.values()]
    total_sq = mx.sum(mx.stack(sq_norms))
    mx.eval(total_sq)
    norm = math.sqrt(float(total_sq))

    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        return {k: v * scale for k, v in grads.items()}

    return grads
