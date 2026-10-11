#!/usr/bin/env python
"""
make_ab_listening.py — blind A/B listening set for two configurations.

Generates the same sentences with the same seeds under configuration X and Y,
randomly assigns X/Y to A/B per sentence, and writes:
    <out>/clips/<n>_A.wav, <n>_B.wav
    <out>/index.html      — one row per sentence: text + two players + a notes box
    <out>/answer_key.json — which of A/B is X and which is Y (open after listening)

Usage:
    python scripts/make_ab_listening.py --adapter ~/.../adapters.safetensors \\
        --levers-x lora-merge-q8,compile-code-pred \\
        --levers-y lora-merge-q8,cp-q4-g32,compile-code-pred \\
        --label-x "code predictor q8" --label-y "code predictor q4 g32" \\
        --prompts bench/prompts/hi_indictts_100.jsonl --n 10 --out bench_runs/listen_q8_vs_q4g32
"""

import argparse
import html
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_tts as bt  # noqa: E402
from bench import levers  # noqa: E402


def generate_set(cfg, train_mod, adapter, lever_names, prompts, seed, gen_kwargs):
    import mlx.core as mx
    model = bt.load_model(train_mod, cfg, adapter)
    kw = dict(gen_kwargs)
    for lv in levers.resolve(lever_names):
        lv.apply(model, kw)
    bt.generate_once(model, prompts[0]["text"], bt.prompt_kwargs(kw, prompts[0]), seed)  # warm-up
    wavs = []
    for p in prompts:
        wav, _ = bt.generate_once(model, p["text"], bt.prompt_kwargs(kw, p), seed)
        wavs.append(wav)
    del model
    mx.clear_cache()
    return wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_tts_hindi_grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--levers-x", required=True)
    ap.add_argument("--levers-y", required=True)
    ap.add_argument("--label-x", required=True)
    ap.add_argument("--label-y", required=True)
    ap.add_argument("--prompts", default="bench/prompts/hi_indictts_100.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import soundfile as sf
    train_mod = bt._load_train_module()
    cfg = train_mod.load_config(args.config)
    g, t = cfg.get("grpo", {}), cfg.get("trainer", {})
    sr = cfg.get("data", {}).get("target_sr", 24000)
    prompts = bt.load_prompts(args.prompts, args.n)
    gen_kwargs = dict(lang_code=t.get("lang_code", "auto"), temperature=g.get("temperature", 0.9),
                      top_p=g.get("top_p", 0.95), top_k=g.get("top_k", 50), repetition_penalty=1.05,
                      max_tokens=g.get("max_new_tokens", 240), split_pattern="", verbose=False)
    adapter = os.path.expanduser(args.adapter)

    print(f"[ab] generating X = {args.label_x}")
    xs = generate_set(cfg, train_mod, adapter, args.levers_x.split(","), prompts, args.seed, gen_kwargs)
    print(f"[ab] generating Y = {args.label_y}")
    ys = generate_set(cfg, train_mod, adapter, args.levers_y.split(","), prompts, args.seed, gen_kwargs)

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.shuffle_seed)
    key, rows = [], []
    for i, (p, x, y) in enumerate(zip(prompts, xs, ys), start=1):
        x_is_a = rng.random() < 0.5
        a, b = (x, y) if x_is_a else (y, x)
        sf.write(out / "clips" / f"{i:02d}_A.wav", a, sr)
        sf.write(out / "clips" / f"{i:02d}_B.wav", b, sr)
        identical = len(x) == len(y) and np.array_equal(x, y)
        key.append({"n": i, "A": args.label_x if x_is_a else args.label_y,
                    "B": args.label_y if x_is_a else args.label_x, "identical_audio": identical,
                    "text": p["text"]})
        rows.append(f"""<tr><td class="n">{i}</td><td class="t">{html.escape(p['text'])}</td>
<td><div class="lab">A</div><audio controls preload="none" src="clips/{i:02d}_A.wav"></audio></td>
<td><div class="lab">B</div><audio controls preload="none" src="clips/{i:02d}_B.wav"></audio></td>
<td><input placeholder="A / B / same — notes"></td></tr>""")

    (out / "answer_key.json").write_text(json.dumps(
        {"X": args.label_x, "Y": args.label_y, "levers_x": args.levers_x, "levers_y": args.levers_y,
         "seed": args.seed, "pairs": key}, ensure_ascii=False, indent=2))
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Blind A/B listening</title>
<style>
body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}}
table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #d0d7de;padding:10px 8px;vertical-align:middle}}
td.n{{color:#57606a;width:2em}} td.t{{font-size:17px}} .lab{{font-weight:600;color:#57606a}}
audio{{width:210px}} input{{width:180px;padding:4px}}
p.note{{color:#57606a}}
</style>
<h1>Blind A/B listening — {len(prompts)} Hindi sentences</h1>
<p class="note">Each row is the same sentence and seed under two model configurations, randomly assigned to A and B.
For each row note which sounds better (or "same"), and any artifacts: buzz, mushy consonants, odd breathing, timing.
Only afterwards open <code>answer_key.json</code>. Notes typed here are not saved — write them down or screenshot.</p>
<table>{''.join(rows)}</table>""")
    n_ident = sum(k["identical_audio"] for k in key)
    print(f"[ab] wrote {len(key)} pairs to {out} ({n_ident} pairs bit-identical)")


if __name__ == "__main__":
    main()
