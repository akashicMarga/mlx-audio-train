#!/usr/bin/env python
"""
grpo_heldout_eval.py — post-hoc held-out CER eval for GRPO adapters.

The in-loop eval (train/grpo/eval.py) is a 6-prompt legibility signal — too noisy
to rank ablation cells. This is the real comparison: each adapter generates a
held-out sentence set via the interleaved deployment path (model.generate-style
rollout, NOT teacher-forced), over N seeds, scored by whisper CER (capped at 1.0)
— matching the protocol behind the doc's validated result table.

Compares the SFT baseline against every ablation cell's final adapter and prints
one table + writes heldout_eval.json. With scipy present, adds a Wilcoxon
signed-rank test vs the SFT baseline (the robust paired claim).

Usage:
    # Score all cells in an ablation run + the SFT baseline
    python scripts/grpo_heldout_eval.py \
        --config ~/grpo_ablation/base_hindi_grpo.local.yaml \
        --ablation-root ~/grpo_ablation/run1 \
        --heldout-jsonl /Users/akashsingh/Documents/exps/hindi/val_codes.abs.jsonl \
        --num-sentences 100 --seeds 2

    # Just print which adapters would be scored (no model load)
    python scripts/grpo_heldout_eval.py --ablation-root ~/grpo_ablation/run1 \
        --config ... --heldout-jsonl ... --dry-run
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_train_module():
    """Import scripts/train.py by path (the name `train` is the train/ package)."""
    spec = importlib.util.spec_from_file_location(
        "grpo_train_entry", REPO_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_adapters(cfg: dict, ablation_root: Path):
    """(label, adapter_path) list: SFT baseline first, then each cell's final."""
    out = []
    sft = cfg.get("grpo", {}).get("init_adapters")
    if sft and os.path.exists(sft):
        out.append(("SFT-baseline", sft))
    for cell_final in sorted(glob.glob(str(ablation_root / "L-*" / "checkpoint-final" / "adapters.safetensors"))):
        label = Path(cell_final).parents[1].name      # the L-... cell dir
        out.append((label, cell_final))
    return out


def load_heldout(train_mod, cfg: dict, model, jsonl: str, n: int):
    """Reuse build_grpo_prompts for the held-out set. `model` lets it build the
    Pipeline-2 ref fields (ref_audio_wav / ref_mel) when speaker_similarity is on;
    for Pipeline 1 it stays text-only."""
    prompts = train_mod.build_grpo_prompts(cfg, model, jsonl_override=jsonl,
                                           max_samples_override=n)
    return prompts


def eval_adapter(model, train_mod, prompts, reward_cfg, *, lang_code, seeds,
                 max_new_tokens, temperature, top_p, top_k, sample_rate, compute_mos):
    """Generate + CER-score the held-out set over `seeds` seeds. Returns dict.

    `compute_mos` also scores DNSMOS OVRL (the doc's quality metric) when the
    speechmos backend is available."""
    import mlx.core as mx
    from train.grpo.rollout import sample_rollouts_interleaved, decode_codes_to_audio
    from train.grpo.rewards import (intelligibility_reward, naturalness_reward,
                                    speaker_similarity_reward, _trailing_silence_frac,
                                    normalize_text)

    win    = max(1, int(sample_rate * 20.0 / 1000.0))
    thresh = 10.0 ** (reward_cfg.silence_rms_db / 20.0)
    cers, durs, cpss, moss, spks = [], [], [], [], []   # one entry per (sentence, seed)
    for si in range(seeds):
        for pi, prompt in enumerate(prompts):
            mx.random.seed(1000 * si + pi)
            out = sample_rollouts_interleaved(
                model, prompt["text"], lang_code=prompt.get("lang_code", lang_code),
                group_size=1, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, compute_ref=False, ref_params=None,
                ref_audio=prompt.get("ref_audio_wav"),   # Pipeline 2: clone the voice
            )
            audios = decode_codes_to_audio(model, out["full_codes"], out["codec_mask"])
            r = intelligibility_reward(audios, [prompt["text"]], reward_cfg, sample_rate=sample_rate)
            cers.append(min(1.0, float(r["cer"][0])))    # capped at 1.0
            if compute_mos:
                moss.append(naturalness_reward(audios, reward_cfg, sample_rate=sample_rate)["mos"][0])
            if prompt.get("ref_mel") is not None:        # Pipeline 2: speaker similarity
                spks.append(speaker_similarity_reward(
                    model, audios, prompt["ref_mel"], sample_rate=sample_rate)["reward"][0])
            wav = np.asarray(audios[0], dtype=np.float32)
            dur = len(wav) / sample_rate
            sil = _trailing_silence_frac(wav, win, thresh)
            durs.append(dur)
            n_chars = len(normalize_text(prompt["text"]).replace(" ", ""))
            if n_chars > 0:
                cpss.append(n_chars / max(dur * (1.0 - sil), 1e-3))
    cers = np.asarray(cers)
    return {
        "n": int(cers.size),
        "cer_mean": float(cers.mean()), "cer_median": float(np.median(cers)),
        "dnsmos_ovrl_mean": float(np.mean(moss)) if moss else None,
        "spk_sim_mean": float(np.mean(spks)) if spks else None,
        "duration_s_mean": float(np.mean(durs)), "speaking_rate_mean": float(np.mean(cpss)),
        "_cer_per_item": cers.tolist(),     # kept for paired stats; dropped from summary
    }


def main():
    ap = argparse.ArgumentParser(description="GRPO held-out CER eval across ablation cells")
    ap.add_argument("--config", required=True, help="Model/GRPO YAML (for base model + init_adapters)")
    ap.add_argument("--ablation-root", required=True, help="Sweep out-root (contains L-*/checkpoint-final)")
    ap.add_argument("--heldout-jsonl", required=True, help="Held-out sentences (text used; not in training)")
    ap.add_argument("--num-sentences", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="List adapters + held-out count, load nothing")
    args = ap.parse_args()

    train_mod = _load_train_module()
    cfg = train_mod.load_config(args.config)
    ablation_root = Path(os.path.expanduser(args.ablation_root)).resolve()
    adapters = discover_adapters(cfg, ablation_root)

    print(f"Adapters to score ({len(adapters)}):")
    for label, path in adapters:
        print(f"  {label:<38} {path}")
    if not adapters:
        print("No adapters found — is the sweep done?"); sys.exit(1)

    if args.dry_run:
        # Count held-out lines without loading the model.
        n = sum(1 for _ in open(args.heldout_jsonl)) if os.path.exists(args.heldout_jsonl) else 0
        print(f"[dry-run] held-out file has {n} lines; would score "
              f"{min(n, args.num_sentences)} × {args.seeds} seeds per adapter.")
        return

    # ── Build the model once (same path as scripts/train.py main) ───────────
    from train.grpo.rewards import RewardConfig
    from train.lora import load_adapters

    model = train_mod.load_model(cfg)
    train_mod.apply_lora(model, cfg)
    for attr in ("speech_tokenizer", "speaker_encoder"):
        sub = getattr(model, attr, None)
        if sub is not None:
            sub.freeze()
    if getattr(model, "speech_tokenizer", None) is None:
        print("[heldout] ERROR: speech_tokenizer required for decode."); sys.exit(1)

    prompts = load_heldout(train_mod, cfg, model, args.heldout_jsonl, args.num_sentences)
    is_p2 = bool(prompts) and prompts[0].get("ref_mel") is not None
    print(f"[heldout] {len(prompts)} held-out sentences × {args.seeds} seeds"
          f"{'  (Pipeline 2: cloning + speaker-sim)' if is_p2 else ''}\n")

    g  = cfg.get("grpo", {})
    rw = g.get("rewards", {}).get("intelligibility", {})
    t  = cfg.get("trainer", {})
    reward_cfg = RewardConfig(
        asr_model=rw.get("asr_model", "mlx-community/whisper-large-v3-turbo"),
        language=rw.get("language", t.get("lang_code", "auto")),
        metric=rw.get("metric", "cer"),
    )
    try:
        import speechmos  # noqa: F401
        compute_mos = True
    except ImportError:
        compute_mos = False
        print("[heldout] speechmos not installed → skipping DNSMOS column "
              "(pip install speechmos onnxruntime to enable)")

    common = dict(
        lang_code=t.get("lang_code", "auto"), seeds=args.seeds,
        max_new_tokens=g.get("max_new_tokens", 240), temperature=g.get("temperature", 0.9),
        top_p=g.get("top_p", 0.95), top_k=g.get("top_k", 50),
        sample_rate=cfg.get("data", {}).get("target_sr", 24000),
        compute_mos=compute_mos,
    )

    results = {}
    for label, path in adapters:
        print(f"=== {label} ===")
        load_adapters(model, path)
        results[label] = eval_adapter(model, train_mod, prompts, reward_cfg, **common)
        r = results[label]
        mos = f"  mos={r['dnsmos_ovrl_mean']:.3f}" if r["dnsmos_ovrl_mean"] is not None else ""
        spk = f"  spk={r['spk_sim_mean']:.4f}" if r["spk_sim_mean"] is not None else ""
        print(f"  cer_mean={r['cer_mean']:.4f}  cer_median={r['cer_median']:.4f}{mos}{spk}  "
              f"dur={r['duration_s_mean']:.2f}s  cps={r['speaking_rate_mean']:.1f}\n")
        # Free MLX's buffer cache between adapters — it grows unbounded across the
        # ~n_sentences×seeds generations and OOM-kills the process on later adapters.
        import gc
        import mlx.core as mx
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass

    # ── Table + paired stats vs SFT baseline ────────────────────────────────
    base = results.get("SFT-baseline")
    print("\n" + "=" * 92)
    print(f"  GRPO held-out eval  ({len(prompts)} sentences × {args.seeds} seeds, CER capped at 1.0)")
    print("=" * 92)
    has_spk = any(r["spk_sim_mean"] is not None for r in results.values())
    spk_hdr = f"{'spk_sim':>9}" if has_spk else ""
    print(f"{'adapter':<38}{'cer_mean':>9}{'Δ vs SFT':>10}{'cer_med':>9}{'mos':>7}"
          f"{spk_hdr}{'cps':>6}{'wilcoxon_p':>12}")
    print("-" * (92 + (9 if has_spk else 0)))
    for label, r in results.items():
        dcer = (r["cer_mean"] - base["cer_mean"]) if base else 0.0
        mos = f"{r['dnsmos_ovrl_mean']:.3f}" if r["dnsmos_ovrl_mean"] is not None else "-"
        spk = (f"{r['spk_sim_mean']:>9.4f}" if r["spk_sim_mean"] is not None
               else (f"{'-':>9}" if has_spk else ""))
        p = ""
        if base and label != "SFT-baseline":
            try:
                from scipy.stats import wilcoxon
                a, b = np.asarray(r["_cer_per_item"]), np.asarray(base["_cer_per_item"])
                if a.shape == b.shape and not np.allclose(a, b):
                    p = f"{wilcoxon(a, b).pvalue:.2e}"
            except Exception:
                p = "n/a"
        print(f"{label:<38}{r['cer_mean']:>9.4f}{dcer:>+10.4f}{r['cer_median']:>9.4f}"
              f"{mos:>7}{spk}{r['speaking_rate_mean']:>6.1f}{p:>12}")
    print("=" * (92 + (9 if has_spk else 0)))

    out = ablation_root / "heldout_eval.json"
    slim = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    out.write_text(json.dumps(slim, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
