#!/usr/bin/env python
"""
bench_tts.py — benchmark ONE inference configuration in THIS process and append
one row to the results ledger. Run each configuration in a fresh process
(scripts/bench_sweep.py does that for a list of configurations).

Baseline is stock mlx-audio `model.generate()`. Optimisations are levers from
bench/levers.py applied on top (--levers a,b). Precision and streaming are plain
configuration (--model-id, --stream-interval).

Passes, in order (each skippable):
  1. warm-up        one generation, discarded
  2. speed          uninstrumented generate() over prompts × seeds until ≥ --min-audio-s
                    of audio: RTF, peak memory, and the audio used for quality
  3. ttfp           streaming generate() on the first --ttfp-prompts prompts:
                    time to first streamed chunk (median)
  4. instrument     component timers (talker / code predictor / codec) on the
                    first --instrument-prompts prompts — forced evals, so these
                    numbers are only used for the breakdown
  5. quality        whisper CER (capped), DNSMOS, speaker similarity; paired
                    Wilcoxon + bit-exactness vs the parent run; gate proposal

Evidence (summary.json, items.jsonl, env.json) goes to --evidence-root/<config_id>/.
Waveforms are not kept (only a sha1 per item, for bit-exactness); pass
--save-wavs to keep them.

Examples:
    # Baseline, Hindi GRPO adapter, M5 Pro
    python scripts/bench_tts.py --config configs/qwen3_tts_hindi_grpo.yaml \\
        --adapter ~/grpo_ablation/gs4/L-interleaved__pg-token__sft-0/checkpoint-final/adapters.safetensors \\
        --variant hindi-grpo --config-id grpo-q8-base --lever baseline --change "stock generate()" \\
        --prompts path/to/heldout.jsonl --num-prompts 100 --seeds 0,1,2

    # One lever on top, compared against that baseline
    python scripts/bench_tts.py ... --config-id grpo-q8-async --parent grpo-q8-base \\
        --levers async-eval --lever async-eval --change "mx.async_eval, EOS one frame late"

    # Check arguments and prompts without loading anything
    python scripts/bench_tts.py ... --dry-run
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench import ledger, levers  # noqa: E402
from bench.env import snapshot  # noqa: E402
from bench.quality import GateConfig, audio_hash, propose, wilcoxon_p  # noqa: E402


# ── inputs ────────────────────────────────────────────────────────────────────
def load_prompts(path: str, n: int):
    p = Path(os.path.expanduser(path))
    if not p.exists():
        raise SystemExit(f"prompts file not found: {p}")
    rows = []
    if p.suffix == ".jsonl":
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"text": r["text"], "lang_code": r.get("lang_code"),
                             "ref_audio": r.get("ref_audio")})
    else:
        rows = [{"text": t.strip(), "lang_code": None, "ref_audio": None}
                for t in p.read_text().splitlines() if t.strip()]
    if not rows:
        raise SystemExit(f"no prompts in {p}")
    return rows[:n] if n else rows


def _load_train_module():
    spec = importlib.util.spec_from_file_location("bench_train_entry", REPO_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(train_mod, cfg, adapter):
    """Same load path as scripts/train.py; LoRA + adapter only when --adapter is given."""
    import mlx.core as mx
    from train.lora import load_adapters

    model = train_mod.load_model(cfg)
    if adapter:
        train_mod.apply_lora(model, cfg)
        load_adapters(model, os.path.expanduser(adapter))
    # MLX modules default to training mode, which keeps LoRA dropout active.
    # Benchmarks always run in eval mode; the row's notes record it.
    model.eval()
    mx.eval(model.parameters())
    return model


def precision_of(model, cfg) -> str:
    q = getattr(model.config, "quantization", None) or getattr(model.config, "quantization_config", None)
    if isinstance(q, dict) and q.get("bits"):
        return f"q{q['bits']}"
    mid = cfg["model"]["model_id"].lower()
    for tag, name in (("bf16", "bf16"), ("fp16", "fp16"), ("8bit", "q8"), ("6bit", "q6"),
                      ("5bit", "q5"), ("4bit", "q4")):
        if tag in mid:
            return name
    return "bf16"


# ── passes ────────────────────────────────────────────────────────────────────
def prompt_kwargs(gen_kwargs, p, **extra):
    kw = dict(gen_kwargs, lang_code=p["lang_code"] or gen_kwargs["lang_code"], **extra)
    if p.get("ref_audio"):
        kw["ref_audio"] = os.path.expanduser(p["ref_audio"])
    return kw


def generate_once(model, text, gen_kwargs, seed):
    """Uninstrumented generate(); returns (wav np.float32, wall seconds)."""
    import mlx.core as mx
    mx.random.seed(seed)
    t0 = time.perf_counter()
    chunks = [r.audio for r in model.generate(text=text, **gen_kwargs)]
    audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    mx.eval(audio)
    wall = time.perf_counter() - t0
    return np.asarray(audio, dtype=np.float32), wall


def speed_pass(model, prompts, seeds, gen_kwargs, sr, min_audio_s, repeats):
    """`repeats` independent timings (each ≥ min_audio_s of audio). RTF = median;
    the spread is kept so the gate can tell a real speedup from run-to-run noise.
    Items (for quality and bit-exactness) come from the first repeat."""
    import mlx.core as mx
    mx.reset_peak_memory()
    items, rtfs, audio_all, wall_all = [], [], 0.0, 0.0
    for rep in range(repeats):
        wall_total = audio_total = 0.0
        first_round = True
        while True:
            for seed in seeds:
                for i, p in enumerate(prompts):
                    wav, wall = generate_once(model, p["text"], prompt_kwargs(gen_kwargs, p), seed)
                    dur = len(wav) / sr
                    wall_total += wall
                    audio_total += dur
                    if rep == 0 and first_round:
                        items.append({"seed": seed, "prompt_idx": i, "text": p["text"],
                                      "audio_s": dur, "wall_s": wall, "audio_sha1": audio_hash(wav),
                                      "_wav": wav})
            first_round = False
            if audio_total >= min_audio_s:
                break
        rtfs.append(wall_total / audio_total)
        audio_all += audio_total
        wall_all += wall_total
    peak = mx.get_peak_memory() / 1e9
    med = statistics.median(rtfs)
    return items, {"rtf": med, "rtf_runs": rtfs, "rtf_spread": (max(rtfs) - min(rtfs)) / med,
                   "audio_s": audio_all, "wall_s": wall_all, "peak_mem_gb": peak}


def ttfp_pass(model, prompts, seed, gen_kwargs, stream_interval):
    import mlx.core as mx
    out = []
    for p in prompts:
        kw = prompt_kwargs(gen_kwargs, p, stream=True, streaming_interval=stream_interval)
        mx.random.seed(seed)
        t0 = time.perf_counter()
        gen = model.generate(text=p["text"], **kw)
        first = next(gen)
        mx.eval(first.audio)
        out.append(1e3 * (time.perf_counter() - t0))
        for _ in gen:            # drain so the generator cleans up
            pass
    return {"ttfp_ms": statistics.median(out), "ttfp_ms_all": out}


def instrument_pass(model, prompts, seed, gen_kwargs, sr):
    from bench.instrument import instrument
    audio_s, wall = 0.0, 0.0
    with instrument(model) as times:
        for p in prompts:
            wav, w = generate_once(model, p["text"], prompt_kwargs(gen_kwargs, p), seed)
            audio_s += len(wav) / sr
            wall += w
    return times.summary(audio_s, wall)


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-levers", action="store_true", help="print the lever ladder and exit")
    ap.add_argument("--config", help="model YAML (model_id, custom_lang_ids, lora, grpo.rewards)")
    ap.add_argument("--adapter", default=None, help="LoRA adapters.safetensors (omit for stock)")
    ap.add_argument("--model-id", default=None, help="override cfg model.model_id (precision lever)")
    ap.add_argument("--variant", help="stock | hindi-sft | hindi-grpo | …")
    ap.add_argument("--config-id", help="unique id for this configuration (ledger + evidence dir)")
    ap.add_argument("--parent", default=None, help="config_id this one changes one thing from")
    ap.add_argument("--lever", help="ledger 'lever' label (baseline, async-eval, quant, stream, …)")
    ap.add_argument("--change", default="", help="one-line description of the change")
    ap.add_argument("--levers", default="", help="comma list of registered levers to apply")
    ap.add_argument("--prompts", help=".jsonl with 'text' (opt. lang_code, ref_audio) or .txt one per line")
    ap.add_argument("--num-prompts", type=int, default=0, help="first N prompts (0 = all)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lang-code", default=None, help="default: cfg trainer.lang_code or 'auto'")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05, help="mlx-audio default")
    ap.add_argument("--max-tokens", type=int, default=None, help="default: cfg grpo.max_new_tokens")
    ap.add_argument("--min-audio-s", type=float, default=10.0, help="per speed repeat")
    ap.add_argument("--speed-repeats", type=int, default=3, help="independent speed timings; RTF = median")
    ap.add_argument("--stream-interval", type=float, default=2.0, help="first/streaming chunk, seconds")
    ap.add_argument("--ttfp-prompts", type=int, default=5)
    ap.add_argument("--instrument-prompts", type=int, default=3)
    ap.add_argument("--skip-ttfp", action="store_true")
    ap.add_argument("--skip-instrument", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--no-dnsmos", action="store_true")
    ap.add_argument("--gate-alpha", type=float, default=0.05)
    ap.add_argument("--min-speedup", type=float, default=1.03)
    ap.add_argument("--max-dnsmos-drop", type=float, default=0.05,
                    help="non-inferiority margin: reject unless the 95%% CI of the per-sentence "
                         "DNSMOS OVRL change stays above −this")
    ap.add_argument("--cer-margin", type=float, default=0.02,
                    help="non-inferiority margin: reject unless the 95%% CI upper bound of the "
                         "per-sentence capped-CER change stays below +this")
    ap.add_argument("--max-spk-drop", type=float, default=None)
    ap.add_argument("--decision", default=None, choices=sorted(ledger.DECISIONS - {""}),
                    help="override the gate's proposed decision")
    ap.add_argument("--notes", default="")
    ap.add_argument("--idle-timeout", type=float, default=600.0,
                    help="wait up to N s for other CPU-heavy processes to stop (0 = don't wait)")
    ap.add_argument("--ledger", default=str(ledger.DEFAULT_LEDGER), help="CSV (env BENCH_LEDGER)")
    ap.add_argument("--evidence-root", default="bench_runs")
    ap.add_argument("--save-wavs", action="store_true")
    ap.add_argument("--no-ledger", action="store_true", help="run but don't append a row")
    ap.add_argument("--dry-run", action="store_true", help="validate inputs, load nothing")
    args = ap.parse_args(argv)
    if not args.list_levers:
        missing = [k for k in ("config", "variant", "config_id", "lever", "prompts") if not getattr(args, k)]
        if missing:
            ap.error("required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_levers:
        print(levers.describe())
        return

    train_mod = _load_train_module()
    cfg = train_mod.load_config(args.config)
    if args.model_id:
        cfg["model"]["model_id"] = args.model_id
    g, t = cfg.get("grpo", {}), cfg.get("trainer", {})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    prompts = load_prompts(args.prompts, args.num_prompts)
    lever_objs = levers.resolve([n for n in args.levers.split(",") if n.strip()])
    sr = cfg.get("data", {}).get("target_sr", 24000)

    gen_kwargs = dict(
        lang_code=args.lang_code or t.get("lang_code", "auto"),
        temperature=args.temperature if args.temperature is not None else g.get("temperature", 0.9),
        top_p=args.top_p if args.top_p is not None else g.get("top_p", 0.95),
        top_k=args.top_k if args.top_k is not None else g.get("top_k", 50),
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens or g.get("max_new_tokens", 240),
        split_pattern="",
        verbose=False,
    )

    evidence = Path(args.evidence_root) / args.config_id
    ledger_path = Path(os.path.expanduser(args.ledger))
    if args.config_id in ledger.config_ids(ledger_path) and not args.no_ledger:
        raise SystemExit(f"config_id '{args.config_id}' already in {ledger_path}; pick a new id")
    if args.adapter and not Path(os.path.expanduser(args.adapter)).exists():
        raise SystemExit(f"adapter not found: {args.adapter}")
    parent_dir = Path(args.evidence_root) / args.parent if args.parent else None
    if parent_dir and not (parent_dir / "summary.json").exists():
        msg = f"parent evidence not found: {parent_dir}/summary.json — run the parent first"
        if not args.dry_run:
            raise SystemExit(msg)
        print(f"[bench] note: {msg}")

    print(f"[bench] {args.config_id}: variant={args.variant} model={cfg['model']['model_id']} "
          f"adapter={'yes' if args.adapter else 'no'} levers={[l.name for l in lever_objs] or 'none'}")
    print(f"[bench] {len(prompts)} prompts × seeds {seeds}; gen={gen_kwargs}")
    if args.dry_run:
        print("[bench] dry run: inputs OK, nothing loaded")
        return

    from bench.env import busy_processes, wait_until_idle
    still_busy = wait_until_idle(timeout_s=args.idle_timeout) if args.idle_timeout > 0 else busy_processes()
    env = snapshot()
    model = load_model(train_mod, cfg, args.adapter)
    for lv in lever_objs:
        lv.apply(model, gen_kwargs)
    precision = precision_of(model, cfg)

    # Hardware telemetry (GPU util/clock, power, temps, swap) over the speed pass only.
    # Started before warm-up because macmon needs ~2.4 s to emit its first sample.
    from bench.telemetry import Telemetry, note as telemetry_note
    with Telemetry() as tel:
        # 1. warm-up
        generate_once(model, prompts[0]["text"], prompt_kwargs(gen_kwargs, prompts[0]), seeds[0])

        # 2. speed (uninstrumented)
        tel.mark_start()
        items, speed = speed_pass(model, prompts, seeds, gen_kwargs, sr, args.min_audio_s, args.speed_repeats)
        tel.mark_end()
    telemetry = tel.summary()
    print(f"[bench] telemetry: {telemetry_note(telemetry)}")
    print(f"[bench] speed: RTF {speed['rtf']:.4f} median of {len(speed['rtf_runs'])} "
          f"({1 / speed['rtf']:.2f}× real-time), spread {100 * speed['rtf_spread']:.1f}%, "
          f"{speed['audio_s']:.1f}s audio total, peak {speed['peak_mem_gb']:.2f} GB")

    # 3. ttfp (streaming)
    ttfp = {} if args.skip_ttfp else ttfp_pass(model, prompts[:args.ttfp_prompts], seeds[0],
                                               gen_kwargs, args.stream_interval)
    if ttfp:
        print(f"[bench] TTFP median {ttfp['ttfp_ms']:.1f} ms (stream_interval {args.stream_interval}s)")

    # 4. component breakdown (instrumented)
    comp = {} if args.skip_instrument else instrument_pass(
        model, prompts[:args.instrument_prompts], seeds[0], gen_kwargs, sr)
    if comp:
        print(f"[bench] instrumented: talker {comp['talker_ms_step']:.2f} ms/step, "
              f"code predictor {comp['mtp_ms_step']:.2f} ms/frame "
              f"({comp['code_pred_calls']} calls / {comp['frames']} frames), "
              f"codec {comp['codec_ms_per_audio_s']:.1f} ms per audio-s, "
              f"other {comp['other_ms_per_frame']:.2f} ms/frame")

    # 5. quality + gate
    parent_items, parent_summary = [], None
    if parent_dir:
        parent_summary = json.loads((parent_dir / "summary.json").read_text())
        parent_items = [json.loads(l) for l in (parent_dir / "items.jsonl").read_text().splitlines()]
    quality = {}
    if not args.skip_quality:
        from bench.quality import score_items
        from train.grpo.rewards import RewardConfig
        reward_cfg = RewardConfig.from_config(g, default_language=t.get("lang_code", "auto"))
        with_dnsmos = not args.no_dnsmos
        if with_dnsmos:
            try:
                import speechmos  # noqa: F401
            except ImportError:
                with_dnsmos = False
                print("[bench] speechmos not installed → no DNSMOS")
        score_items(model, items, reward_cfg, sr, with_dnsmos=with_dnsmos, with_spk=False)
        quality["cer_capped"] = float(np.mean([i["cer_capped"] for i in items]))
        if with_dnsmos:
            quality["dnsmos_ovrl"] = float(np.mean([i["dnsmos_ovrl"] for i in items]))
        if parent_items:
            quality["wilcoxon_p"] = wilcoxon_p(items, parent_items)
        print(f"[bench] quality: CER(capped) {quality['cer_capped']:.4f}"
              + (f", DNSMOS {quality['dnsmos_ovrl']:.3f}" if "dnsmos_ovrl" in quality else "")
              + (f", per-sentence Wilcoxon p vs parent {quality['wilcoxon_p']}" if parent_items else ""))

    busy_after = busy_processes()
    contention = still_busy + [b for b in busy_after if b["comm"] not in {x["comm"] for x in still_busy}]
    if contention:
        print(f"[bench] WARNING: other busy processes during run: "
              f"{', '.join(f'{b['comm']} {b['pcpu']:.0f}%' for b in contention[:3])}")
    summary = {"config_id": args.config_id, "parent": args.parent, **speed, **ttfp, **comp, **quality,
               "contention": contention, "telemetry": telemetry,
               "precision": precision, "gen_kwargs": gen_kwargs, "levers": [l.name for l in lever_objs],
               "n_prompts": len(prompts), "seeds": seeds, "model_eval_mode": True}
    gate = GateConfig(args.gate_alpha, args.min_speedup, args.max_dnsmos_drop, args.max_spk_drop,
                      args.cer_margin)
    decision, reason = propose(summary, parent_summary, items, parent_items, gate)
    if args.decision:
        reason = f"manual override ({decision} proposed: {reason})"
        decision = args.decision
    summary.update(decision=decision, decision_reason=reason)
    print(f"[bench] decision: {decision} — {reason}")

    # evidence
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (evidence / "env.json").write_text(json.dumps(env, indent=2))
    with (evidence / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}) + "\n")
    if args.save_wavs:
        import soundfile as sf
        wdir = evidence / "wavs"
        wdir.mkdir(exist_ok=True)
        for it in items:
            sf.write(wdir / f"s{it['seed']}_p{it['prompt_idx']:03d}.wav", it["_wav"], sr)

    if args.no_ledger:
        print(f"[bench] evidence → {evidence} (ledger skipped)")
        return
    notes = "; ".join(x for x in [
        args.notes, "codec_ms = decode ms per second of audio", "model.eval() (LoRA dropout off)",
        ("CONTENTION: " + ", ".join(f"{b['comm']} {b['pcpu']:.0f}%" for b in contention[:3])) if contention else "",
        telemetry_note(telemetry),
        f"stream_interval={args.stream_interval}s" if ttfp else "",
        f"rtf runs {','.join(f'{r:.4f}' for r in speed['rtf_runs'])} (spread {100 * speed['rtf_spread']:.1f}%)",
        f"gate: {reason}"] if x)
    row = {
        "date": dt.date.today().isoformat(), "device": env["device"], "variant": args.variant,
        "config_id": args.config_id, "parent_config_id": args.parent or "", "lever": args.lever,
        "change": args.change, "model_id": cfg["model"]["model_id"], "precision": precision,
        "batch_size": 1, "mlx_version": env["mlx"], "mlx_audio_version": env["mlx_audio"],
        "n_prompts": len(prompts), "n_seeds": len(seeds), "audio_s": speed["audio_s"],
        "rtf": speed["rtf"], "ttfp_ms": ttfp.get("ttfp_ms"),
        "talker_ms_step": comp.get("talker_ms_step"), "mtp_ms_step": comp.get("mtp_ms_step"),
        "codec_ms": comp.get("codec_ms_per_audio_s"), "snr_db": None,
        "peak_mem_gb": speed["peak_mem_gb"], "cer_capped": quality.get("cer_capped"),
        "wilcoxon_p": quality.get("wilcoxon_p"), "dnsmos_ovrl": quality.get("dnsmos_ovrl"),
        "spk_sim": None, "decision": decision, "evidence_path": str(evidence),
        "notes": notes,
    }
    path = ledger.append_row(row, ledger_path)
    print(f"[bench] ledger row → {path}; evidence → {evidence}")


if __name__ == "__main__":
    main()
