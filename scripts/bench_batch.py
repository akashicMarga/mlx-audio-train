#!/usr/bin/env python
"""
bench_batch.py — batch-size sweep (Part 4, §7): how throughput and per-request
latency change when G utterances are generated together.

GRPO-shaped workload: each batch is G copies of ONE prompt (a rollout group),
run over prompts × seeds. Baseline is stock mlx-audio `model.batch_generate()`
(Qwen3TTSBatchSession); weight levers (e.g. lora-merge-q8) apply as usual.
Loop levers from bench/fast_generate.py do NOT apply — batch_generate has its
own loop.

Per G it records:
  throughput      total audio seconds / total wall seconds, all streams together
  rtf             wall / audio (aggregate) — the ledger's rtf column
  batch latency   wall seconds per batch (the slowest stream finishes last)
  step ms         wall per batch / frames of the longest stream
  peak memory
  quality         CER (capped) and DNSMOS on up to --score-per-batch streams per batch.
                  Batched sampling consumes random numbers in a different order than
                  single streams, so items can't be paired with G=1: quality is reported
                  descriptively and the ledger decision is left blank.

One ledger row per G (batch_size column set), evidence in bench_runs/<config-id>-G<g>/.

Usage:
    python scripts/bench_batch.py --config configs/qwen3_tts_hindi_grpo.yaml \\
        --adapter ~/grpo_ablation/.../adapters.safetensors --levers lora-merge-q8 \\
        --variant hindi-grpo --config-id batch-m8 --prompts bench/prompts/hi_smoke.txt \\
        --groups 1,2,4,8,16 --seeds 0,1 --no-ledger
"""

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_tts as bt  # noqa: E402
from bench import ledger, levers  # noqa: E402
from bench.env import busy_processes, snapshot, wait_until_idle  # noqa: E402


def run_batch(model, text, g, seed, gen_kwargs, engine="stock"):
    import mlx.core as mx
    mx.random.seed(seed)
    t0 = time.perf_counter()
    if engine != "stock":
        from bench.fast_generate import generate_group
        wavs, ends = generate_group(model, text, g, compile_code_pred=(engine == "group"), **gen_kwargs)
        wall = time.perf_counter() - t0
        return [np.asarray(w, dtype=np.float32) for w in wavs], wall, max(ends) + 1
    results = list(model.batch_generate(texts=[text] * g, **gen_kwargs))
    for r in results:
        mx.eval(r.audio)
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r.sequence_idx)
    wavs = [np.asarray(r.audio, dtype=np.float32) for r in results]
    frames = max(r.token_count for r in results) if results else 0
    return wavs, wall, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--levers", default="")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config-id", required=True, help="prefix; rows get <id>-G<g>")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-prompts", type=int, default=0)
    ap.add_argument("--groups", default="1,2,4,8,16")
    ap.add_argument("--engine", default="stock", choices=["stock", "group", "group-nocompile"],
                    help="stock = mlx-audio batch_generate; group = GRPO-shaped batched loop "
                         "(plain batched KVCache, compiled code predictor, stock decoder)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lang-code", default=None)
    ap.add_argument("--score-per-batch", type=int, default=4, help="streams scored per batch (0 = none)")
    ap.add_argument("--no-dnsmos", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--ledger", default=str(ledger.DEFAULT_LEDGER))
    ap.add_argument("--evidence-root", default="bench_runs")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=600.0)
    args = ap.parse_args()

    train_mod = bt._load_train_module()
    cfg = train_mod.load_config(args.config)
    if args.model_id:
        cfg["model"]["model_id"] = args.model_id
    g_cfg, t_cfg = cfg.get("grpo", {}), cfg.get("trainer", {})
    groups = [int(x) for x in args.groups.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    prompts = bt.load_prompts(args.prompts, args.num_prompts)
    sr = cfg.get("data", {}).get("target_sr", 24000)
    gen_kwargs = dict(lang_code=args.lang_code or t_cfg.get("lang_code", "auto"),
                      temperature=g_cfg.get("temperature", 0.9), top_p=g_cfg.get("top_p", 0.95),
                      top_k=g_cfg.get("top_k", 50), repetition_penalty=1.05,
                      max_tokens=g_cfg.get("max_new_tokens", 240), verbose=False)

    still_busy = wait_until_idle(timeout_s=args.idle_timeout) if args.idle_timeout > 0 else busy_processes()
    env = snapshot()
    model = bt.load_model(train_mod, cfg, args.adapter)
    lever_objs = levers.resolve([n for n in args.levers.split(",") if n.strip()])
    for lv in lever_objs:
        lv.apply(model, gen_kwargs)
    precision = bt.precision_of(model, cfg)

    reward_cfg = None
    if args.score_per_batch:
        from train.grpo.rewards import RewardConfig
        reward_cfg = RewardConfig.from_config(g_cfg, default_language=t_cfg.get("lang_code", "auto"))

    import mlx.core as mx
    summaries = []
    for g in groups:
        run_batch(model, prompts[0]["text"], g, seeds[0], gen_kwargs, args.engine)   # warm-up at this G
        mx.reset_peak_memory()
        wall_total = audio_total = 0.0
        batch_walls, step_ms, items = [], [], []
        for seed in seeds:
            for pi, p in enumerate(prompts):
                wavs, wall, frames = run_batch(model, p["text"], g, seed, gen_kwargs, args.engine)
                wall_total += wall
                audio_total += sum(len(w) for w in wavs) / sr
                batch_walls.append(wall)
                if frames:
                    step_ms.append(1e3 * wall / frames)
                for si, w in enumerate(wavs[: args.score_per_batch]):
                    items.append({"seed": seed, "prompt_idx": pi, "stream": si, "text": p["text"],
                                  "audio_s": len(w) / sr, "_wav": w})
        peak = mx.get_peak_memory() / 1e9
        quality = {}
        if items and reward_cfg is not None:
            from bench.quality import score_items
            with_dnsmos = not args.no_dnsmos
            score_items(model, items, reward_cfg, sr, with_dnsmos=with_dnsmos, with_spk=False)
            quality["cer_capped"] = float(np.mean([i["cer_capped"] for i in items]))
            if with_dnsmos:
                quality["dnsmos_ovrl"] = float(np.mean([i["dnsmos_ovrl"] for i in items]))
        s = {"config_id": f"{args.config_id}-G{g}", "batch_size": g, "rtf": wall_total / audio_total,
             "throughput_audio_s_per_s": audio_total / wall_total, "audio_s": audio_total,
             "wall_s": wall_total, "batch_latency_s_median": statistics.median(batch_walls),
             "step_ms_median": statistics.median(step_ms) if step_ms else None,
             "peak_mem_gb": peak, "n_batches": len(batch_walls), "precision": precision,
             "levers": [l.name for l in lever_objs], "engine": args.engine, **quality}
        summaries.append(s)
        print(f"[batch] G={g:2d}  throughput {s['throughput_audio_s_per_s']:6.2f} audio-s/s  "
              f"(RTF {s['rtf']:.4f})  batch latency {s['batch_latency_s_median']:.2f}s  "
              f"step {s['step_ms_median']:.1f} ms  peak {peak:.2f} GB"
              + (f"  CER {quality['cer_capped']:.3f}" if "cer_capped" in quality else "")
              + (f"  DNSMOS {quality['dnsmos_ovrl']:.3f}" if "dnsmos_ovrl" in quality else ""))

        ev = Path(args.evidence_root) / s["config_id"]
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "summary.json").write_text(json.dumps(s, indent=2))
        (ev / "env.json").write_text(json.dumps(env, indent=2))
        with (ev / "items.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}) + "\n")
        if not args.no_ledger:
            base = summaries[0]
            notes = "; ".join(x for x in [
                args.notes, f"batch sweep: G copies of one prompt (GRPO group), engine={args.engine}",
                f"throughput {s['throughput_audio_s_per_s']:.2f} audio-s/s "
                f"({s['throughput_audio_s_per_s'] / base['throughput_audio_s_per_s']:.2f}× G={base['batch_size']})",
                f"batch latency {s['batch_latency_s_median']:.2f}s; step {s['step_ms_median']:.1f} ms",
                "quality descriptive (unpaired vs G=1)",
                ("CONTENTION: " + ", ".join(f"{b['comm']} {b['pcpu']:.0f}%" for b in still_busy[:3])) if still_busy else "",
            ] if x)
            ledger.append_row({
                "date": dt.date.today().isoformat(), "device": env["device"], "variant": args.variant,
                "config_id": s["config_id"], "parent_config_id": summaries[0]["config_id"] if g != groups[0] else "",
                "lever": "batch", "change": f"{args.engine} engine, G={g}", "model_id": cfg["model"]["model_id"],
                "precision": precision, "batch_size": g, "mlx_version": env["mlx"],
                "mlx_audio_version": env["mlx_audio"], "n_prompts": len(prompts), "n_seeds": len(seeds),
                "audio_s": audio_total, "rtf": s["rtf"], "peak_mem_gb": peak,
                "cer_capped": quality.get("cer_capped"), "dnsmos_ovrl": quality.get("dnsmos_ovrl"),
                "decision": "", "evidence_path": str(ev), "notes": notes,
            }, Path(os.path.expanduser(args.ledger)))
        mx.clear_cache()


if __name__ == "__main__":
    main()
