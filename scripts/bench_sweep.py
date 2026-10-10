#!/usr/bin/env python
"""
bench_sweep.py — run a list of benchmark configurations, each in a fresh
process, in order. Configurations already in the ledger are skipped, so an
interrupted sweep resumes where it stopped.

Sweep YAML:

    defaults:                      # passed to every run (CLI flag names, no dashes)
      config: configs/qwen3_tts_hindi_grpo.yaml
      prompts: ~/data/hindi_heldout.jsonl
      num_prompts: 100
      seeds: "0,1,2"
    runs:
      - config_id: grpo-q8-base
        variant: hindi-grpo
        adapter: ~/grpo_ablation/gs4/L-interleaved__pg-token__sft-0/checkpoint-final/adapters.safetensors
        lever: baseline
        change: stock mlx-audio generate()
      - config_id: grpo-q4
        parent: grpo-q8-base
        variant: hindi-grpo
        adapter: ...
        model_id: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
        lever: quant
        change: q8 -> q4 base

Usage:
    python scripts/bench_sweep.py bench/sweeps/<name>.yaml [--keep-going] [--dry-run]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench import ledger  # noqa: E402

FLAGS = {"skip_ttfp", "skip_instrument", "skip_quality", "no_dnsmos", "save_wavs", "no_ledger"}


def to_argv(opts: dict) -> list:
    argv = []
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        if k in FLAGS:
            if v:
                argv.append(flag)
        elif v is not None:
            argv += [flag, str(v)]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    ap.add_argument("--dry-run", action="store_true", help="pass --dry-run to every run")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.sweep).read_text())
    defaults = spec.get("defaults", {})
    ledger_path = Path(os.path.expanduser(defaults.get("ledger", str(ledger.DEFAULT_LEDGER))))
    done = ledger.config_ids(ledger_path)

    failed = []
    for run in spec["runs"]:
        opts = {**defaults, **run}
        cid = opts["config_id"]
        if cid in done and not args.dry_run:
            print(f"[sweep] skip {cid} (already in {ledger_path})")
            continue
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "bench_tts.py"), *to_argv(opts)]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[sweep] run {cid}: {' '.join(cmd[2:])}")
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            failed.append(cid)
            print(f"[sweep] {cid} failed (exit {rc})")
            if not args.keep_going:
                break
    if failed:
        sys.exit(f"[sweep] failed: {', '.join(failed)}")
    print("[sweep] done")


if __name__ == "__main__":
    main()
