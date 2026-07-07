#!/usr/bin/env python
"""
grpo_reward_ablation.py — the {std, none} × {linear, tanh} 2×2 (reward-todo #4).

Two switches from the GRPO reward backlog, ablated together because they interact:

  adv_norm     (train/grpo/rewards.py:group_advantages)
    std  : A = (r − mean)/std   — DeepSeek default
    none : A =  r − mean        — Dr. GRPO; drops the /std bias

  reward_shape (train/grpo/rewards.py:intelligibility_reward)
    linear : 1 − min(1, CER)
    tanh   : 1 − tanh(k·CER)    — ~2.6× wider within-group spread at CER≈0.12

WHY together: `/std` partially CANCELS reward shaping (tanh widens the spread, then
/std renormalises it away). So `none + tanh` is the only cell where shaping fully
lands — you can't read either axis alone.

THE TRAP this script exists to handle: dropping `/std` shrinks |advantage| ~10×
(reward∈[0,1], within-group dev ~±0.1, std ~0.08). At a fixed LR the `none` cells
look dead — not because the idea is bad, but because the effective step size fell
10×. So the `none` cells get their LR multiplied by `--nostd-lr-mult` (default 8×)
to match effective step size. Without this, the comparison lies.

Each cell is a clean `scripts/train.py --config <generated>.yaml` subprocess (no
cross-run state). Runs go OUTSIDE the repo by default.

Usage:
    # Dry run — print the grid + write the 4 generated configs, run nothing
    python scripts/grpo_reward_ablation.py \
        --base-config configs/qwen3_tts_hindi_grpo.yaml --dry-run

    # Full 2×2 (base LR from the config for std cells, ×8 for none cells)
    python scripts/grpo_reward_ablation.py \
        --base-config configs/qwen3_tts_hindi_grpo.yaml \
        --out-root ~/grpo_reward_ablation --steps 150

    # Override the std-cell LR and the none-cell multiplier
    python scripts/grpo_reward_ablation.py ... --lr 5e-6 --nostd-lr-mult 8

    # Collect/print an already-run sweep
    python scripts/grpo_reward_ablation.py --out-root ~/grpo_reward_ablation --collect-only

Resume: re-running the same command SKIPS cells that already finished (those with
checkpoint-final/adapters.safetensors), so an interrupted sweep is safe to restart
verbatim — only the unfinished cells re-run. Pass --force to re-run everything.
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

ADV_NORMS     = ["std", "none"]
REWARD_SHAPES = ["linear", "tanh"]


def cell_name(adv_norm: str, reward_shape: str) -> str:
    """Stable, filesystem-safe id for a grid cell."""
    return f"an-{adv_norm}__rs-{reward_shape}"


def build_grid(adv_norms, reward_shapes):
    return list(itertools.product(adv_norms, reward_shapes))


def cell_lr(adv_norm: str, base_lr: float, nostd_mult: float) -> float:
    """`none` cells get the LR bump to match effective step size (see module doc)."""
    return base_lr * nostd_mult if adv_norm == "none" else base_lr


def materialize_config(base_cfg: dict, adv_norm, reward_shape, *, base_lr, nostd_mult,
                       reward_k, out_root: Path, steps: int, eval_every: int) -> dict:
    """Deep-copy the base config and apply this cell's overrides.

    Sets the two ablation switches on the grpo block, the per-cell LR on the
    trainer block, and roots all output/TensorBoard/log paths under out_root/<cell>
    (OUTSIDE the repo). TensorBoard MUST be set — the in-loop eval only attaches
    when a tb writer exists.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("grpo", {})
    cfg.setdefault("trainer", {})

    # adv_norm is a global on the grpo block; reward_shape/reward_k are params of
    # the intelligibility reward (registry keys), set in its rewards sub-block.
    cfg["grpo"]["adv_norm"] = adv_norm
    intel = cfg["grpo"].setdefault("rewards", {}).setdefault("intelligibility", {})
    intel["reward_shape"] = reward_shape
    intel["reward_k"]     = reward_k

    lr = cell_lr(adv_norm, base_lr, nostd_mult)
    cfg["trainer"]["learning_rate"] = lr

    name    = cell_name(adv_norm, reward_shape)
    run_dir = out_root / name
    cfg["trainer"]["output_dir"]         = str(run_dir)
    cfg["trainer"]["run_name"]           = name
    cfg["trainer"]["tensorboard_dir"]    = str(run_dir / "tb")
    cfg["trainer"]["log_file"]           = str(run_dir / "train_log.jsonl")
    cfg["trainer"]["max_steps"]          = steps
    cfg["trainer"]["eval_every_n_steps"] = eval_every
    return cfg


def cell_complete(run_dir: Path) -> bool:
    """True if this cell already finished. The trainer writes
    checkpoint-final/adapters.safetensors only after the run completes, so its
    presence is a reliable done-marker for safe resume (skip finished cells)."""
    return (run_dir / "checkpoint-final" / "adapters.safetensors").exists()


def run_cell(cfg: dict, cfg_path: Path) -> int:
    """Write the generated config and run scripts/train.py on it. Returns rc."""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
           "--config", str(cfg_path)]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def collect_eval(run_dir: Path):
    """Return (final_row, best_cer_row) from a run's grpo_eval.jsonl.

    Skips reference_only (step-0) baseline rows. Returns (None, None) if the run
    produced no real eval (crashed, or no tensorboard_dir)."""
    log = run_dir / "grpo_eval.jsonl"
    if not log.exists():
        return None, None
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("reference_only"):
            continue
        rows.append(r)
    if not rows:
        return None, None
    final = rows[-1]
    best  = min(rows, key=lambda r: r.get("cer", float("inf")))
    return final, best


def actual_cell_lr(run_dir: Path, fallback: float) -> float:
    """The LR a cell was ACTUALLY run at (read from its generated config.yaml),
    not recomputed from the current --nostd-lr-mult. Matters when a cell was run
    at a different multiplier than the summary's default (e.g. a re-run)."""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            lr = yaml.safe_load(cfg_path.read_text()).get("trainer", {}).get("learning_rate")
            if lr is not None:
                return float(lr)
        except Exception:
            pass
    return fallback


def summarize(out_root: Path, grid, *, base_lr, nostd_mult):
    """Print a 2×2 comparison table (final + best CER per cell) and dump JSON."""
    hdr = (f"{'cell':<24} {'lr':>9} {'steps':>5} {'cer_final':>9} {'cer_best':>8} "
           f"{'kl':>7} {'cps':>6} {'dur_s':>6}")
    print("\n" + "=" * len(hdr))
    print("  GRPO reward 2×2: adv_norm × reward_shape  (lower CER better; "
          "watch KL small, cps not collapsing)")
    print("  NOTE: 'none' cells run at LR ×%g — do NOT compare raw LR across rows." % nostd_mult)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    rows_for_json = []
    for adv_norm, reward_shape in grid:
        name = cell_name(adv_norm, reward_shape)
        lr = actual_cell_lr(out_root / name, cell_lr(adv_norm, base_lr, nostd_mult))
        final, best = collect_eval(out_root / name)
        if final is None:
            print(f"{name:<24} {lr:>9.1e} {'—':>5} {'(no eval — not run / crashed)':>9}")
            rows_for_json.append({"cell": name, "adv_norm": adv_norm,
                                  "reward_shape": reward_shape, "lr": lr,
                                  "status": "missing"})
            continue
        print(f"{name:<24} {lr:>9.1e} {final['step']:>5} {final['cer']:>9.3f} "
              f"{best['cer']:>8.3f} {final['kl']:>7.4f} "
              f"{final['speaking_rate']:>6.1f} {final['duration_s']:>6.2f}")
        rows_for_json.append({
            "cell": name, "adv_norm": adv_norm, "reward_shape": reward_shape,
            "lr": lr, "step": final["step"],
            "cer_final": final["cer"], "cer_best": best["cer"],
            "kl_final": final["kl"], "speaking_rate_final": final["speaking_rate"],
            "duration_s_final": final["duration_s"],
        })
    print("=" * len(hdr))
    summary_path = out_root / "reward_ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows_for_json, indent=2))
    print(f"\nWrote {summary_path}")


def main():
    ap = argparse.ArgumentParser(
        description="GRPO reward ablation: adv_norm {std,none} × reward_shape {linear,tanh}")
    ap.add_argument("--base-config", help="Base GRPO YAML (pipeline: grpo)")
    ap.add_argument("--out-root", default="~/grpo_reward_ablation",
                    help="Where run dirs go (default OUTSIDE the repo)")
    ap.add_argument("--steps", type=int, default=150, help="max_steps per cell")
    ap.add_argument("--eval-every", type=int, default=25, help="in-loop eval cadence")
    ap.add_argument("--lr", type=float, default=None,
                    help="Base LR for the std cells (default: read from base config)")
    ap.add_argument("--nostd-lr-mult", type=float, default=8.0,
                    help="LR multiplier for the 'none' cells (matches effective step "
                         "size after dropping /std; memory says ~5–10×)")
    ap.add_argument("--reward-k", type=float, default=3.0, help="tanh steepness k")
    ap.add_argument("--adv-norms", nargs="+", default=ADV_NORMS, choices=ADV_NORMS)
    ap.add_argument("--reward-shapes", nargs="+", default=REWARD_SHAPES, choices=REWARD_SHAPES)
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate configs + print the plan, run nothing")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip running; just summarize an existing sweep")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cells even if already complete (default: resume — "
                         "skip cells that have checkpoint-final/adapters.safetensors)")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue the sweep even if a cell fails")
    args = ap.parse_args()

    out_root = Path(os.path.expanduser(args.out_root)).resolve()
    grid = build_grid(args.adv_norms, args.reward_shapes)

    if args.collect_only:
        # base_lr is only for display here; fall back to a placeholder if unknown.
        base_lr = args.lr if args.lr is not None else 5e-6
        summarize(out_root, grid, base_lr=base_lr, nostd_mult=args.nostd_lr_mult)
        return

    if not args.base_config:
        ap.error("--base-config is required unless --collect-only")
    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    if base_cfg.get("pipeline") != "grpo":
        ap.error(f"{args.base_config} is not a GRPO config (pipeline != grpo)")

    base_lr = args.lr if args.lr is not None \
        else float(base_cfg.get("trainer", {}).get("learning_rate", 5e-6))

    print(f"GRPO reward 2×2: {len(grid)} cells × {args.steps} steps  →  {out_root}")
    print(f"  std-cell LR = {base_lr:.1e}   none-cell LR = {base_lr * args.nostd_lr_mult:.1e} "
          f"(×{args.nostd_lr_mult:g})   tanh k = {args.reward_k:g}")
    for adv_norm, reward_shape in grid:
        print(f"  - {cell_name(adv_norm, reward_shape)}  "
              f"(lr={cell_lr(adv_norm, base_lr, args.nostd_lr_mult):.1e})")

    failures = []
    for adv_norm, reward_shape in grid:
        name = cell_name(adv_norm, reward_shape)
        run_dir = out_root / name
        if not args.force and cell_complete(run_dir):
            print(f"[resume] skip complete cell: {name} "
                  f"(has checkpoint-final; use --force to re-run)")
            continue
        cfg  = materialize_config(base_cfg, adv_norm, reward_shape,
                                  base_lr=base_lr, nostd_mult=args.nostd_lr_mult,
                                  reward_k=args.reward_k, out_root=out_root,
                                  steps=args.steps, eval_every=args.eval_every)
        cfg_path = run_dir / "config.yaml"
        if args.dry_run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[dry-run] wrote {cfg_path}")
            continue
        rc = run_cell(cfg, cfg_path)
        if rc != 0:
            failures.append((name, rc))
            print(f"[reward-ablation] cell FAILED rc={rc}: {name}")
            if not args.keep_going:
                print("[reward-ablation] stopping (use --keep-going to continue)")
                break

    if not args.dry_run:
        summarize(out_root, grid, base_lr=base_lr, nostd_mult=args.nostd_lr_mult)
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {[n for n, _ in failures]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
