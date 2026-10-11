#!/usr/bin/env python
"""
bench_teacher_forced.py — how much does a code-predictor change move its distributions?

Free-running comparisons (CER, DNSMOS, listening) see two *different* utterances once
sampling diverges, so waveform metrics like MCD between them mostly measure the
divergence. This check removes it:

  1. Generate with the reference configuration (merged q8 + compiled code predictor)
     and record, for every frame, the code predictor's inputs (Talker hidden state,
     codebook-0 token) and the fifteen tokens it sampled.
  2. Replay those exact frames through each variant's code predictor, teacher-forced
     on the reference's tokens, and collect the logits at every pass.
  3. Compare against the reference's own teacher-forced logits, per codebook 1–15:
       kl          KL(ref ‖ variant) of the raw (temperature-1) softmax, nats
       top1        argmax agreement
       tv          total-variation distance between the *sampling* distributions
                   (temperature, top-k, top-p as deployed) — the probability that
                   an optimally coupled draw picks a different token
       dlogp       mean log-prob change of the reference's sampled token

The "unmerged" variant (the deployed LoRA before §5.1's merge) is the calibration:
the size of a change that already passed the quality gate.

Usage:
    python scripts/bench_teacher_forced.py --adapter ~/.../adapters.safetensors \\
        --prompts bench/prompts/hi_indictts_100.jsonl --n 20 \\
        --variants unmerged,cp-q4,cp-q4-g32,cp-mlp-q4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_tts as bt  # noqa: E402
from bench import levers  # noqa: E402
import bench.fast_generate as fg  # noqa: E402

REF_LEVERS = ["lora-merge-q8", "compile-code-pred"]


def load(train_mod, cfg, adapter, lever_names, gen_kwargs):
    model = bt.load_model(train_mod, cfg, adapter)
    model.eval()                       # LoRA dropout off (it defaults to training mode)
    kw = dict(gen_kwargs)
    for lv in levers.resolve(lever_names):
        lv.apply(model, kw)
    return model, kw


def record_frames(model, kw, prompts, seed):
    """Run the reference generation; return per-frame (code_hidden, cb0, codes[1,15])."""
    import mlx.core as mx
    orig = fg._code_pred_frame_fn
    frames = []

    def patched(model_, temperature, top_k, top_p, sampler="single"):
        fn = orig(model_, temperature, top_k, top_p, sampler=sampler)

        def wrapped(code_hidden, cb0):
            codes, emb = fn(code_hidden, cb0)
            frames.append((code_hidden, cb0, codes))
            return codes, emb
        return wrapped

    fg._code_pred_frame_fn = patched
    try:
        for p in prompts:
            bt.generate_once(model, p["text"], bt.prompt_kwargs(kw, p), seed)
    finally:
        fg._code_pred_frame_fn = orig
    for f in frames:
        mx.eval(*f)
    return frames


def tf_logits(model, frames):
    """Teacher-forced code-predictor logits: [n_frames, 15, V] float32 (numpy)."""
    import mlx.core as mx
    talker = model.talker
    cp = talker.code_predictor
    n_groups = model.config.talker_config.num_code_groups
    out = []
    for code_hidden, cb0, codes in frames:
        caches = [fg._ConcatCache() for _ in cp.model.layers]
        ls = []
        for i in range(n_groups - 1):
            if i == 0:
                inp = mx.concatenate([code_hidden, talker.get_input_embeddings()(cb0)], axis=1)
            else:
                inp = cp.codec_embedding[i - 1](codes[:, i - 1:i])
            logits, _, _ = cp(inp, cache=caches, generation_step=i)
            ls.append(logits[:, -1, :])
        L = mx.concatenate(ls, axis=0).astype(mx.float32)
        mx.eval(L)
        out.append(np.array(L))
    return np.stack(out)


def sampling_probs(logits, temperature, top_k, top_p):
    """The deployed code-predictor sampling distribution (same ops as the sampler)."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_top_k
    from mlx_audio.tts.models.qwen3_tts import qwen3_tts as q3
    rows = []
    for chunk in np.array_split(logits.reshape(-1, logits.shape[-1]), max(1, logits.size // 4_000_000)):
        l = mx.array(chunk) / temperature
        l = apply_top_k(l, top_k)
        l = q3._apply_probability_filters(l, top_p, 0.0)
        p = mx.softmax(l, axis=-1)
        mx.eval(p)
        rows.append(np.array(p))
    return np.concatenate(rows).reshape(logits.shape)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def compare(ref_logits, var_logits, ref_tokens, ref_sp, var_sp):
    """Per-codebook metrics (arrays of length 15) plus overall means."""
    lr, lv = log_softmax(ref_logits), log_softmax(var_logits)
    kl = (np.exp(lr) * (lr - lv)).sum(-1)                         # [F, 15]
    top1 = ref_logits.argmax(-1) == var_logits.argmax(-1)
    tv = 0.5 * np.abs(ref_sp - var_sp).sum(-1)
    tok = ref_tokens[..., None]
    dlogp = np.take_along_axis(lv, tok, -1)[..., 0] - np.take_along_axis(lr, tok, -1)[..., 0]
    per_cb = {k: v.mean(0).tolist() for k, v in
              (("kl", kl), ("top1", top1.astype(float)), ("tv", tv), ("dlogp", dlogp))}
    # chance that a frame's fifteen coupled draws all agree, treating passes as independent
    frame_same = float(np.prod(1 - tv, axis=-1).mean())
    return {"per_codebook": per_cb,
            "mean": {k: float(np.mean(v)) for k, v in per_cb.items()},
            "frame_all_same_coupled": frame_same}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_tts_hindi_grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", default="bench/prompts/hi_indictts_100.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="unmerged,cp-q4,cp-q4-g32,cp-mlp-q4",
                    help="'unmerged' = the adapter before the LoRA merge; anything else is a "
                         "lever applied after lora-merge-q8")
    ap.add_argument("--out", default="bench_runs/teacher_forced")
    args = ap.parse_args()

    import mlx.core as mx
    train_mod = bt._load_train_module()
    cfg = train_mod.load_config(args.config)
    g, t = cfg.get("grpo", {}), cfg.get("trainer", {})
    temperature, top_k, top_p = g.get("temperature", 0.9), g.get("top_k", 50), g.get("top_p", 0.95)
    gen_kwargs = dict(lang_code=t.get("lang_code", "auto"), temperature=temperature, top_p=top_p,
                      top_k=top_k, repetition_penalty=1.05, max_tokens=g.get("max_new_tokens", 240),
                      split_pattern="", verbose=False)
    adapter = os.path.expanduser(args.adapter)
    prompts = bt.load_prompts(args.prompts, args.n)

    t0 = time.time()
    model, kw = load(train_mod, cfg, adapter, REF_LEVERS, gen_kwargs)
    frames = record_frames(model, kw, prompts, args.seed)
    ref_tokens = np.concatenate([np.array(c) for _, _, c in frames])          # [F, 15]
    ref_logits = tf_logits(model, frames)
    ref_sp = sampling_probs(ref_logits, temperature, top_k, top_p)
    # self-check: every token the compiled reference sampled must be inside the support
    # of the teacher-forced, uncompiled reference distribution
    in_support = float((np.take_along_axis(ref_sp, ref_tokens[..., None], -1)[..., 0] > 0).mean())
    print(f"[tf] {len(prompts)} prompts, {len(frames)} frames recorded; "
          f"reference tokens inside teacher-forced support: {100 * in_support:.2f}%")
    del model
    mx.clear_cache()

    report = {"reference": REF_LEVERS, "prompts": args.prompts, "n_prompts": len(prompts),
              "seed": args.seed, "n_frames": len(frames), "sampling": [temperature, top_k, top_p],
              "ref_tokens_in_support": in_support, "variants": {}}
    for name in args.variants.split(","):
        lv_names = [] if name == "unmerged" else ["lora-merge-q8", name]
        model, _ = load(train_mod, cfg, adapter, lv_names, gen_kwargs)
        var_logits = tf_logits(model, frames)
        var_sp = sampling_probs(var_logits, temperature, top_k, top_p)
        r = compare(ref_logits, var_logits, ref_tokens, ref_sp, var_sp)
        report["variants"][name] = r
        m = r["mean"]
        print(f"[tf] {name:10s} KL {m['kl']:.4f} nats · top-1 agree {100 * m['top1']:.1f}% · "
              f"TV {m['tv']:.4f} · Δlogp {m['dlogp']:+.4f} · frame all-same {100 * r['frame_all_same_coupled']:.1f}%")
        del model, var_logits, var_sp
        mx.clear_cache()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"→ {out / 'report.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
