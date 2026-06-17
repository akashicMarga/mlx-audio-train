#!/usr/bin/env python
"""
grpo_ablation.py — controlled GRPO ablation over layout × pg_norm × sft_lambda.

Next-step #4 from docs/grpo_post_training.md. We have partial evidence on the
layout axis (v1 concatenated lost to SFT, v3 interleaved won) but nothing
controlled, and zero data on pg_norm / sft_lambda. This sweeps the 2×2×2 grid at
small scale (~100–150 steps each, audio snapshots every eval cadence) so the
three axes can be compared under identical data/seed/budget.

Each cell is a normal `scripts/train.py --config <generated>.yaml` run (clean
subprocess → no cross-run state leakage; the model reload is negligible next to
rollout time). Every run writes `grpo_eval.jsonl` (the in-loop fixed-prompt eval);
this script collects the final + best-CER row per cell into one table.

Usage:
    # Dry run — just print the grid + generated configs, run nothing
    python scripts/grpo_ablation.py --base-config configs/qwen3_tts_hindi_grpo.yaml --dry-run

    # Full sweep (writes runs OUTSIDE the repo by default)
    python scripts/grpo_ablation.py --base-config configs/qwen3_tts_hindi_grpo.yaml \
        --out-root ~/grpo_ablation --steps 120

    # Restrict an axis (e.g. only the interleaved layout)
    python scripts/grpo_ablation.py ... --layouts interleaved

    # Collect/print results from an already-run sweep without re-running
    python scripts/grpo_ablation.py --out-root ~/grpo_ablation --collect-only
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

# Default axes — the three switches built in next-steps #2/#3 + the layout flag.
LAYOUTS      = ["interleaved", "concatenated"]
PG_NORMS     = ["token", "sequence"]
SFT_LAMBDAS  = [0.0, 0.1]


def cell_name(layout: str, pg_norm: str, sft_lambda: float) -> str:
    """Stable, filesystem-safe id for a grid cell."""
    return f"L-{layout}__pg-{pg_norm}__sft-{sft_lambda:g}"


def build_grid(layouts, pg_norms, sft_lambdas):
    return list(itertools.product(layouts, pg_norms, sft_lambdas))


def materialize_config(base_cfg: dict, layout, pg_norm, sft_lambda,
                       out_root: Path, steps: int, eval_every: int) -> dict:
    """Deep-copy the base config and apply this cell's overrides.

    Output/TensorBoard/log paths are rooted under out_root/<cell> (kept OUTSIDE
    the repo). TensorBoard MUST be set — the in-loop eval only attaches when a
    tb writer exists.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("grpo", {})
    cfg.setdefault("trainer", {})

    cfg["grpo"]["layout"]     = layout
    cfg["grpo"]["pg_norm"]    = pg_norm
    cfg["grpo"]["sft_lambda"] = sft_lambda

    name    = cell_name(layout, pg_norm, sft_lambda)
    run_dir = out_root / name
    cfg["trainer"]["output_dir"]         = str(run_dir)
    cfg["trainer"]["run_name"]           = name
    cfg["trainer"]["tensorboard_dir"]    = str(run_dir / "tb")
    cfg["trainer"]["log_file"]           = str(run_dir / "train_log.jsonl")
    cfg["trainer"]["max_steps"]          = steps
    cfg["trainer"]["eval_every_n_steps"] = eval_every
    return cfg


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

    Skips the reference_only (step-0) baseline rows. Returns (None, None) if the
    run produced no real eval (crashed, or no tensorboard_dir)."""
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


def summarize(out_root: Path, grid):
    """Print a comparison table across the grid (final + best CER per cell)."""
    hdr = (f"{'cell':<40} {'steps':>5} {'cer_final':>9} {'cer_best':>8} "
           f"{'kl':>7} {'cps':>6} {'dur_s':>6}")
    print("\n" + "=" * len(hdr))
    print("  GRPO ablation summary  (lower CER better; watch KL small, cps not collapsing)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    rows_for_json = []
    for layout, pg_norm, sft_lambda in grid:
        name = cell_name(layout, pg_norm, sft_lambda)
        final, best = collect_eval(out_root / name)
        if final is None:
            print(f"{name:<40} {'—':>5} {'(no eval — not run / crashed)':>9}")
            rows_for_json.append({"cell": name, "status": "missing"})
            continue
        print(f"{name:<40} {final['step']:>5} {final['cer']:>9.3f} "
              f"{best['cer']:>8.3f} {final['kl']:>7.4f} "
              f"{final['speaking_rate']:>6.1f} {final['duration_s']:>6.2f}")
        rows_for_json.append({
            "cell": name, "layout": layout, "pg_norm": pg_norm,
            "sft_lambda": sft_lambda, "step": final["step"],
            "cer_final": final["cer"], "cer_best": best["cer"],
            "kl_final": final["kl"], "speaking_rate_final": final["speaking_rate"],
            "duration_s_final": final["duration_s"],
        })
    print("=" * len(hdr))
    summary_path = out_root / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows_for_json, indent=2))
    print(f"\nWrote {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="GRPO ablation: layout × pg_norm × sft_lambda")
    ap.add_argument("--base-config", help="Base GRPO YAML (pipeline: grpo)")
    ap.add_argument("--out-root", default="~/grpo_ablation",
                    help="Where run dirs go (default OUTSIDE the repo)")
    ap.add_argument("--steps", type=int, default=120, help="max_steps per cell")
    ap.add_argument("--eval-every", type=int, default=25, help="in-loop eval cadence")
    ap.add_argument("--layouts", nargs="+", default=LAYOUTS, choices=LAYOUTS)
    ap.add_argument("--pg-norms", nargs="+", default=PG_NORMS, choices=PG_NORMS)
    ap.add_argument("--sft-lambdas", nargs="+", type=float, default=SFT_LAMBDAS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate configs + print the plan, run nothing")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip running; just summarize an existing sweep")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue the sweep even if a cell fails")
    args = ap.parse_args()

    out_root = Path(os.path.expanduser(args.out_root)).resolve()
    grid = build_grid(args.layouts, args.pg_norms, args.sft_lambdas)

    if args.collect_only:
        summarize(out_root, grid)
        return

    if not args.base_config:
        ap.error("--base-config is required unless --collect-only")
    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    if base_cfg.get("pipeline") != "grpo":
        ap.error(f"{args.base_config} is not a GRPO config (pipeline != grpo)")

    print(f"GRPO ablation: {len(grid)} cells × {args.steps} steps  →  {out_root}")
    for layout, pg_norm, sft_lambda in grid:
        print(f"  - {cell_name(layout, pg_norm, sft_lambda)}")

    failures = []
    for layout, pg_norm, sft_lambda in grid:
        name = cell_name(layout, pg_norm, sft_lambda)
        cfg  = materialize_config(base_cfg, layout, pg_norm, sft_lambda,
                                  out_root, args.steps, args.eval_every)
        cfg_path = out_root / name / "config.yaml"
        if args.dry_run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[dry-run] wrote {cfg_path}")
            continue
        rc = run_cell(cfg, cfg_path)
        if rc != 0:
            failures.append((name, rc))
            print(f"[ablation] cell FAILED rc={rc}: {name}")
            if not args.keep_going:
                print("[ablation] stopping (use --keep-going to continue)")
                break

    if not args.dry_run:
        summarize(out_root, grid)
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {[n for n, _ in failures]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
