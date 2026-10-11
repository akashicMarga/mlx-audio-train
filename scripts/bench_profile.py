#!/usr/bin/env python
"""
bench_profile.py — where does a Qwen3-TTS decode frame spend its time, layer by layer?

MLX is lazy: to time a module you must force evaluation of its inputs before the
clock starts and of its outputs before it stops. Every such bracket is a
CPU↔GPU sync that costs time and serialises work, so bracketing every
projection at once would mostly measure the brackets. Instead this profiles one
LEVEL at a time (run each level in a fresh process; `--level all` does that):

  1 components   Talker forward · code predictor pass · codec decoder ·
                 sampling (cb0 / code predictor) · embedding lookups
  2 blocks       each decoder layer, final norms, heads, decoder stages
  3 sublayers    attention vs MLP vs the two norms, inside every layer
  4 ops          q/k/v/o and gate/up/down projections (+ the quantized base under
                 each LoRA wrapper, so LoRA's cost is the difference), q/k norms

Within a level, shares are of the time bracketed at that level. "unbracketed"
is everything else in the frame (Python, glue ops, the loop's own sync). The
per-bracket sync cost is calibrated and reported, so you can see how much a
level inflates its own numbers. Compare levels by share, not by absolute ms.

Usage:
    python scripts/bench_profile.py --config configs/qwen3_tts_hindi_grpo.yaml \\
        --adapter ~/grpo_ablation/gs4/L-interleaved__pg-token__sft-0/checkpoint-final/adapters.safetensors \\
        --tag grpo-q8 --prompts bench/prompts/hi_smoke.txt --num-prompts 3 --level all

Writes bench_runs/profile-<tag>/L<n>.json and prints one table per level.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

LEVELS = {
    1: [r"talker", r"talker\.code_predictor", r"decoder",
        r"talker\.model\.codec_embedding", r"talker\.code_predictor\.model\.codec_embedding\.\d+"],
    2: [r"talker\.model\.layers\.\d+", r"talker\.model\.norm", r"talker\.codec_head",
        r"talker\.model\.codec_embedding",
        r"talker\.code_predictor\.model\.layers\.\d+", r"talker\.code_predictor\.model\.norm",
        r"talker\.code_predictor\.lm_head\.\d+", r"talker\.code_predictor\.model\.codec_embedding\.\d+",
        r"talker\.code_predictor\.small_to_mtp_projection",
        r"decoder\.quantizer", r"decoder\.pre_conv", r"decoder\.pre_transformer",
        r"decoder\.upsample\.\d+\.\d+", r"decoder\.decoder\.\d+"],
    3: [r"talker(\.code_predictor)?\.model\.layers\.\d+\.(self_attn|mlp|input_layernorm|post_attention_layernorm)",
        r"talker\.codec_head", r"talker\.code_predictor\.lm_head\.\d+",
        r"decoder\.pre_transformer\.layers\.\d+\.(self_attn|mlp)", r"decoder\.decoder\.\d+\.block\.\d+",
        r"decoder\.upsample\.\d+\.\d+"],
    4: [r"talker(\.code_predictor)?\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj(\.base)?",
        r"talker(\.code_predictor)?\.model\.layers\.\d+\.mlp\.(gate|up|down)_proj(\.base)?",
        r"talker(\.code_predictor)?\.model\.layers\.\d+\.self_attn\.(q|k)_norm",
        r"talker\.codec_head", r"talker\.code_predictor\.lm_head\.\d+"],
}
LEVEL_NAMES = {1: "components", 2: "blocks", 3: "sublayers", 4: "ops"}


def norm_path(p):
    return re.sub(r"\.\d+(?=\.|$)", ".N", p)


def _arrays(o):
    import mlx.core as mx
    if isinstance(o, mx.array):
        yield o
    elif isinstance(o, (list, tuple)):
        for x in o:
            yield from _arrays(x)
    elif isinstance(o, dict):
        for x in o.values():
            yield from _arrays(x)


def _force(o):
    import mlx.core as mx
    a = list(_arrays(o))
    if a:
        mx.eval(*a)


def calibrate(n=300):
    """Cost of one bracket (eval in, tiny op, eval out) — the profiler's own tax."""
    import mlx.core as mx
    x = mx.ones((1, 1, 1024))
    mx.eval(x)
    ts = []
    for _ in range(n):
        _force(x)
        t0 = time.perf_counter()
        y = x + 1
        _force(y)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


class Profiler:
    def __init__(self, model, level):
        self.model = model
        pats = [re.compile(p + r"$") for p in LEVELS[level]]
        self.paths = {}                    # id(module) -> path, for selected modules only
        all_mods = []
        for root, mod in (("talker", model.talker), ("decoder", model.speech_tokenizer.decoder)):
            for name, sub in mod.named_modules():
                path = f"{root}.{name}" if name else root
                all_mods.append((path, sub))
                if any(p.match(path) for p in pats):
                    self.paths[id(sub)] = path
        self.stats = defaultdict(list)     # path -> [self seconds] (children subtracted)
        self.stack = []                    # [child_seconds] of open brackets: nesting from the call stack
        self.top_s = 0.0                   # time inside outermost brackets
        self.frames = 0
        self.prefills = 0
        self._patched = []
        classes = {type(m) for _, m in all_mods if id(m) in self.paths}
        classes.add(type(model.talker))    # always count frames
        self.time_sampling = level == 1

        for cls in classes:
            own = "__call__" in vars(cls)
            orig = cls.__call__
            self._patched.append((cls, orig, own))
            cls.__call__ = self._wrap(orig, count_frames=cls is type(model.talker))

        self._orig_sample = model._sample_token if self.time_sampling else None
        if self.time_sampling:
            def sample(*a, **k):
                key = "sampling.cb0" if k.get("suppress_tokens") else "sampling.code_predictor"
                return self._bracket(key, lambda: self._orig_sample(*a, **k), (a, k))
            model._sample_token = sample

    def _wrap(self, orig, count_frames):
        prof = self

        def w(mod, *a, **k):
            if count_frames:
                x = a[0] if a else k.get("inputs_embeds")
                if getattr(x, "ndim", 0) >= 2 and x.shape[1] > 1:
                    prof.prefills += 1
                else:
                    prof.frames += 1
            path = prof.paths.get(id(mod))
            if path is None:
                return orig(mod, *a, **k)
            return prof._bracket(path, lambda: orig(mod, *a, **k), (a, k))
        return w

    def _bracket(self, key, call, inputs):
        _force(inputs)
        self.stack.append(0.0)
        t0 = time.perf_counter()
        out = call()
        _force(out)
        dt = time.perf_counter() - t0
        children = self.stack.pop()
        self.stats[key].append(dt - children)
        if self.stack:
            self.stack[-1] += dt
        else:
            self.top_s += dt
        return out

    def restore(self):
        for cls, orig, own in self._patched:
            if own:
                cls.__call__ = orig
            else:
                del cls.__call__
        if self.time_sampling:
            del self.model._sample_token


def report(stats, frames, wall_s, bracket_s, level, top):
    selected = set(stats)
    self_t = {p: sum(v) for p, v in stats.items()}
    calls = {p: len(v) for p, v in stats.items()}
    groups = defaultdict(lambda: {"self_s": 0.0, "calls": 0, "per_instance": defaultdict(float)})
    for p in selected:
        g = groups[norm_path(p)]
        g["self_s"] += self_t[p]
        g["calls"] += calls[p]
        g["per_instance"][p] += self_t[p]
    rows = []
    for k, g in groups.items():
        inst = list(g["per_instance"].values())
        ms = 1e3 * g["self_s"] / frames
        rows.append({"module": k, "ms_per_frame": ms,
                     # subtract the profiler's own sync cost: calls × calibrated bracket cost
                     "net_ms_per_frame": max(0.0, ms - 1e3 * bracket_s * g["calls"] / frames),
                     "share_of_bracketed": g["self_s"] / top if top else 0,
                     "calls_per_frame": g["calls"] / frames, "instances": len(inst),
                     "instance_ms_per_frame_min": 1e3 * min(inst) / frames,
                     "instance_ms_per_frame_max": 1e3 * max(inst) / frames})
    rows.sort(key=lambda r: -r["ms_per_frame"])
    total_calls = sum(calls.values())
    out = {"level": level, "level_name": LEVEL_NAMES[level], "frames": frames,
           "instrumented_ms_per_frame": 1e3 * wall_s / frames,
           "bracketed_ms_per_frame": 1e3 * top / frames,
           "unbracketed_ms_per_frame": 1e3 * (wall_s - top) / frames,
           "brackets_per_frame": total_calls / frames,
           "bracket_cost_us": 1e6 * bracket_s,
           "est_bracket_tax_ms_per_frame": 1e3 * bracket_s * total_calls / frames,
           "rows": rows}
    return out


def print_report(r):
    print(f"\n=== Level {r['level']} — {r['level_name']} ===")
    print(f"frames {r['frames']} · instrumented {r['instrumented_ms_per_frame']:.2f} ms/frame · "
          f"bracketed {r['bracketed_ms_per_frame']:.2f} · unbracketed {r['unbracketed_ms_per_frame']:.2f}")
    print(f"{r['brackets_per_frame']:.0f} brackets/frame × {r['bracket_cost_us']:.0f} µs ≈ "
          f"{r['est_bracket_tax_ms_per_frame']:.2f} ms/frame of profiler tax (spread across rows)")
    print(f"{'module':66s} {'ms/frame':>9s} {'net':>7s} {'share':>6s} {'calls/fr':>8s} {'inst':>4s}  per-instance ms/frame")
    for row in r["rows"]:
        rng = (f"{row['instance_ms_per_frame_min']:.3f}–{row['instance_ms_per_frame_max']:.3f}"
               if row["instances"] > 1 else "")
        print(f"{row['module']:66s} {row['ms_per_frame']:9.3f} {row['net_ms_per_frame']:7.3f} {100 * row['share_of_bracketed']:5.1f}% "
              f"{row['calls_per_frame']:8.1f} {row['instances']:4d}  {rng}")


def run_level(args, level):
    import mlx.core as mx
    import bench_tts as bt
    train_mod = bt._load_train_module()
    cfg = train_mod.load_config(args.config)
    if args.model_id:
        cfg["model"]["model_id"] = args.model_id
    model = bt.load_model(train_mod, cfg, args.adapter)
    from bench import levers
    gen_kwargs = dict(lang_code=args.lang_code or cfg.get("trainer", {}).get("lang_code", "auto"),
                      temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.05,
                      max_tokens=cfg.get("grpo", {}).get("max_new_tokens", 240),
                      split_pattern="", verbose=False)
    for lv in levers.resolve([n for n in args.levers.split(",") if n.strip()]):
        lv.apply(model, gen_kwargs)
    prompts = bt.load_prompts(args.prompts, args.num_prompts)
    bt.generate_once(model, prompts[0]["text"], bt.prompt_kwargs(gen_kwargs, prompts[0]), 0)  # warm-up
    bracket = calibrate()
    prof = Profiler(model, level)
    wall = 0.0
    try:
        for p in prompts:
            _, w = bt.generate_once(model, p["text"], bt.prompt_kwargs(gen_kwargs, p), args.seed)
            wall += w
    finally:
        prof.restore()
    frames = prof.frames + prof.prefills
    r = report(prof.stats, frames, wall, bracket, level, prof.top_s)
    r.update(tag=args.tag, adapter=args.adapter, levers=args.levers, prompts=len(prompts))
    out = REPO_ROOT / "bench_runs" / f"profile-{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"L{level}.json").write_text(json.dumps(r, indent=2))
    print_report(r)
    mx.clear_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--levers", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang-code", default=None)
    ap.add_argument("--level", default="all", help="1-4 or 'all' (each level in a fresh process)")
    args = ap.parse_args()
    if args.level == "all":
        base = [a for a in sys.argv[1:]]
        i = base.index("--level") if "--level" in base else None
        if i is not None:
            del base[i:i + 2]
        for lv in LEVELS:
            rc = subprocess.run([sys.executable, __file__, *base, "--level", str(lv)], cwd=REPO_ROOT).returncode
            if rc:
                sys.exit(rc)
        return
    run_level(args, int(args.level))


if __name__ == "__main__":
    main()
