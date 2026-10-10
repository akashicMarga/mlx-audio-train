#!/usr/bin/env python
"""
bench_kernels.py — should we write a fused Metal kernel for the code predictor?

Answers, on this machine:
  1. dispatch   fixed GPU cost per kernel: chains of N tiny dependent matmuls
                (N = 1…1000), compiled and uncompiled; the slope is µs/kernel.
  2. qmv        efficiency of an 8-bit (and bf16, 4-bit) matvec vs output size:
                achieved GB/s against the chip's peak, from code-predictor shapes up.
  3. frame      the exact matmuls of one code-predictor frame (15 passes × 5 layers
                × 7 projections + 15 heads, q8) run as a dependent chain, vs the
                real compiled code-predictor frame on the merged model. The gap is
                everything that isn't a matmul (norms, RoPE, attention, sampling,
                glue) — an upper bound on what a fused kernel could recover.

Each section runs independently; a failure in one is reported and the rest continue.
Writes bench_runs/kernels/report.json and prints a summary.

Usage:
    python scripts/bench_kernels.py --config configs/qwen3_tts_hindi_grpo.yaml \\
        --adapter ~/grpo_ablation/.../adapters.safetensors [--quick]
"""

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PEAK_GBPS = 307.0          # M5 Pro spec; achieved bandwidth is reported against it


def timeit(fn, warmup=3, iters=20):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ── 1. per-kernel dispatch cost ──────────────────────────────────────────────
def section_dispatch(quick):
    w = (mx.random.normal((64, 64)) * 0.05).astype(mx.bfloat16)
    x0 = mx.random.normal((1, 64)).astype(mx.bfloat16)
    mx.eval(w, x0)
    ns = [1, 10, 100] if quick else [1, 10, 100, 300, 1000]
    out = {}
    for compiled in (False, True):
        rows = []
        for n in ns:
            def chain(x, n=n):
                for _ in range(n):
                    x = x @ w          # matmuls are not fused by mx.compile: one kernel each
                return x
            f = mx.compile(chain) if compiled else chain
            t = timeit(lambda: f(x0), iters=10 if quick else 30)
            rows.append({"n": n, "ms": 1e3 * t})
        # least-squares slope ms per kernel
        xs = [r["n"] for r in rows]
        ys = [r["ms"] for r in rows]
        mx_, my = statistics.mean(xs), statistics.mean(ys)
        slope = sum((a - mx_) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx_) ** 2 for a in xs)
        out["compiled" if compiled else "lazy"] = {"rows": rows, "us_per_kernel": 1e3 * slope,
                                                   "intercept_ms": my - slope * mx_}
    return out


# ── 2. matvec efficiency vs size ─────────────────────────────────────────────
def section_qmv(quick):
    in_f = 1024
    outs = [1024, 3072] if quick else [1024, 2048, 3072, 6144, 12288, 24576]
    chain_len = 20 if quick else 50
    res = []
    for out_f in outs:
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight = (mx.random.normal((out_f, in_f)) * 0.02).astype(mx.bfloat16)
        variants = {"bf16": lin,
                    "q8": nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8),
                    "q4": nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)}
        bpp = {"bf16": 2.0, "q8": 8.5 / 8, "q4": 4.5 / 8}
        x0 = mx.random.normal((1, 1, in_f)).astype(mx.bfloat16)
        mx.eval(x0, lin.parameters(), variants["q8"].parameters(), variants["q4"].parameters())
        for name, layer in variants.items():
            def chain(x, layer=layer):
                for _ in range(chain_len):
                    x = layer(x)[..., :in_f]         # dependent: next input is a prefix of this output
                return x
            f = mx.compile(chain)
            t = timeit(lambda: f(x0), iters=10 if quick else 20) / chain_len
            gb = out_f * in_f * bpp[name] / 1e9
            res.append({"out": out_f, "in": in_f, "dtype": name, "us_per_call": 1e6 * t,
                        "gbps": gb / t, "pct_peak": 100 * gb / t / PEAK_GBPS})
    return res


# ── 3. one code-predictor frame: matmuls only vs the real compiled frame ─────
def section_frame(args):
    import os
    import bench_tts as bt
    from bench import levers
    from bench.fast_generate import _code_pred_frame_fn

    train_mod = bt._load_train_module()
    cfg = train_mod.load_config(args.config)
    model = bt.load_model(train_mod, cfg, os.path.expanduser(args.adapter) if args.adapter else None)
    if args.adapter:
        levers.resolve(["lora-merge-q8"])[0].apply(model, {})
    cp = model.talker.code_predictor
    layers = cp.model.layers

    # (a) matmul-only chain with the real (merged, q8) weights, exact per-frame order
    projs = []
    for L in layers:
        a, m = L.self_attn, L.mlp
        projs.append([a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj])
    hidden = cp.model.layers[0].self_attn.hidden_size
    x0 = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)

    def matmul_frame(x):
        for p in range(15):
            for q, k, v, o, g, u, d in projs:
                h = q(x)
                h = h + k(x).sum() + v(x).sum()            # keep k, v live (cheap reductions)
                x = o(h)
                x = d(g(x) * u(x))
            x = x + cp.lm_head[p](x)[..., :hidden]
        return x
    matmul_frame_c = mx.compile(matmul_frame)          # compile once, time the compiled call
    t_matmul = timeit(lambda: matmul_frame_c(x0), iters=10)

    # (b) the real compiled code-predictor frame (embeddings, 5 layers × 15, heads, sampling)
    frame = _code_pred_frame_fn(model, 0.9, 50, 0.95)
    tok = mx.array([[123]], dtype=mx.int32)
    t_frame = timeit(lambda: frame(x0, tok), iters=20)

    n_mm = 15 * (5 * 7 + 1)
    return {"matmul_only_ms": 1e3 * t_matmul, "real_frame_ms": 1e3 * t_frame,
            "non_matmul_ms": 1e3 * (t_frame - t_matmul),
            "non_matmul_share": (t_frame - t_matmul) / t_frame, "matmuls_per_frame": n_mm,
            "note": "matmul-only chain adds 2 small reductions + 2 elementwise ops per layer to keep "
                    "outputs live; treat non_matmul_ms as a lower bound on removable overhead"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_tts_hindi_grpo.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for checking the script runs")
    ap.add_argument("--sections", default="dispatch,qmv,frame")
    args = ap.parse_args()

    from bench.env import snapshot
    report = {"env": snapshot(), "peak_gbps": PEAK_GBPS}
    for name in args.sections.split(","):
        print(f"== {name}")
        try:
            if name == "dispatch":
                r = section_dispatch(args.quick)
                for k, v in r.items():
                    print(f"  {k:8s} {v['us_per_kernel']:.2f} µs per kernel (intercept {v['intercept_ms']:.3f} ms)")
            elif name == "qmv":
                r = section_qmv(args.quick)
                for row in r:
                    print(f"  {row['in']}→{row['out']:<6d} {row['dtype']:5s} {row['us_per_call']:8.1f} µs  "
                          f"{row['gbps']:6.1f} GB/s  ({row['pct_peak']:.0f}% of peak)")
            elif name == "frame":
                r = section_frame(args)
                print(f"  matmuls only {r['matmul_only_ms']:.2f} ms · real compiled frame {r['real_frame_ms']:.2f} ms · "
                      f"non-matmul {r['non_matmul_ms']:.2f} ms ({100 * r['non_matmul_share']:.0f}%)")
            report[name] = r
        except Exception as e:  # noqa: BLE001
            report[name] = {"error": repr(e), "trace": traceback.format_exc()}
            print(f"  FAILED: {e!r}")
    out = REPO_ROOT / "bench_runs" / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("report_quick.json" if args.quick else "report.json")).write_text(json.dumps(report, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
