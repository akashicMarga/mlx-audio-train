#!/usr/bin/env python
"""
curate_prompts.py — data curation by expected advantage (GRPO reward backlog #5).

The real fix for dead / low-contrast groups. GRPO learns from *within-group reward
variance*: a prompt whose rollouts all succeed (solved) or all fail (hopeless)
produces a zero-variance group → zero advantage → no gradient (the trainer already
skips these at train time, but only after paying the rollout cost every epoch).
This script front-loads that: it pre-scores the train set with a cheap rollout pass
under the STARTING policy (the SFT adapter) and keeps the "Goldilocks middle" —
prompts where the model has room AND the rollouts disagree — so every training
rollout thereafter lands on a prompt that can actually teach something.

Signal: per prompt, roll out G samples, CER-score each, and compute the **pass rate**
(fraction with CER ≤ `--pass-thresh`). pass_rate ∈ (low, high) ⇒ the group has
reward variance by construction ⇒ keep. pass_rate ≥ high ⇒ solved (drop);
pass_rate ≤ low ⇒ hopeless (drop). This is curriculum-by-pass-rate, à la RLVR.

Two stages, so the expensive part runs once:
  1. SCORE  — roll out + score every prompt → `<train>.scored.jsonl` (each original
              record + `_curate: {pass_rate, mean_cer, cer_std, mean_reward,
              reward_std, n}`). Resumable: re-running skips already-scored records.
  2. FILTER — apply the pass-rate band → `<out>` (kept records, ready for a GRPO
              config's train_jsonl). Cheap and re-runnable via `--filter-only` with
              different `--keep-low/--keep-high` — no re-rollout.

Usage:
    # Score + filter in one go (writes scored.jsonl next to the train jsonl)
    python scripts/curate_prompts.py \
        --config configs/qwen3_tts_hindi_grpo.yaml \
        --train-jsonl /path/to/train_codes.abs.jsonl \
        --out /path/to/train_curated.jsonl \
        --group-size 6 --pass-thresh 0.10 --keep-low 0.2 --keep-high 0.8

    # Peek: distribution only, no model load
    python scripts/curate_prompts.py --config ... --train-jsonl ... --dry-run

    # Re-filter an existing scored.jsonl with a different band (no rollouts)
    python scripts/curate_prompts.py --config ... --train-jsonl ... \
        --out /path/to/train_curated_tight.jsonl \
        --filter-only --keep-low 0.3 --keep-high 0.7
"""

import argparse
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


def read_records(jsonl_path: str, max_prompts=None):
    """Original records (with text), in file order — the unit we curate + write back."""
    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("text"):
                continue
            recs.append(rec)
            if max_prompts and len(recs) >= max_prompts:
                break
    return recs


def scored_path_for(train_jsonl: str) -> Path:
    p = Path(train_jsonl)
    return p.with_suffix(".scored" + p.suffix)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: score (roll out G per prompt under the SFT policy, compute pass rate)
# ──────────────────────────────────────────────────────────────────────────────

def score_records(records, model, reward_cfg, *, group_size, pass_thresh, lang_code,
                  max_new_tokens, temperature, top_p, top_k, sample_rate,
                  scored_out: Path, done_texts: set):
    """Roll out + CER-score each not-yet-scored record; append `_curate` stats to
    `scored_out` incrementally (so an interruption resumes). Returns nothing; read
    the stats back from `scored_out`."""
    import mlx.core as mx
    from train.grpo.rollout import sample_rollouts_interleaved, decode_codes_to_audio
    from train.grpo.rewards import RewardContext, score

    n_total = len(records)
    with open(scored_out, "a") as fh:
        for i, rec in enumerate(records):
            text = rec["text"]
            if text in done_texts:                 # resume: already scored
                continue
            mx.random.seed(1234 + i)               # reproducible per-prompt rollouts
            out = sample_rollouts_interleaved(
                model, text, lang_code=rec.get("lang_code", lang_code),
                group_size=group_size, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                compute_ref=False, ref_params=None,   # curation needs no KL reference
            )
            audios = decode_codes_to_audio(model, out["full_codes"], out["codec_mask"])
            ctx = RewardContext(audios=audios, texts=[text] * group_size,
                                sample_rate=sample_rate, model=model)
            r = score("intelligibility", ctx, reward_cfg)
            cers = np.minimum(1.0, np.asarray(r["cer"], dtype=np.float32))
            rewards = np.asarray(r["reward"], dtype=np.float32)
            pass_rate = float(np.mean(cers <= pass_thresh))
            rec = dict(rec)
            rec["_curate"] = {
                "pass_rate":   pass_rate,
                "mean_cer":    float(cers.mean()),
                "cer_std":     float(cers.std()),
                "mean_reward": float(rewards.mean()),
                "reward_std":  float(rewards.std()),
                "n":           int(group_size),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0 or i + 1 == n_total:
                print(f"  scored {i + 1}/{n_total}  (last: pass_rate={pass_rate:.2f} "
                      f"mean_cer={cers.mean():.3f})")


def load_scored(scored_out: Path):
    """Records that already carry `_curate` stats, keyed by text (for resume + filter)."""
    if not scored_out.exists():
        return {}
    by_text = {}
    for line in open(scored_out):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "_curate" in rec:
            by_text[rec["text"]] = rec
    return by_text


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: filter (pass-rate band) + report
# ──────────────────────────────────────────────────────────────────────────────

def classify(pass_rate: float, keep_low: float, keep_high: float) -> str:
    if pass_rate >= keep_high:
        return "solved"
    if pass_rate <= keep_low:
        return "hopeless"
    return "keep"


def histogram(vals, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    counts = [0] * (len(bins) - 1)
    for v in vals:
        for b in range(len(bins) - 1):
            if bins[b] <= v < bins[b + 1]:
                counts[b] += 1
                break
    return counts


def filter_and_report(records, scored_by_text, *, keep_low, keep_high, out_path):
    """Split scored records by pass-rate band, write the kept subset, print a report.
    `records` sets the OUTPUT ORDER + is the original source (so unscored records are
    surfaced, not silently dropped)."""
    kept, buckets = [], {"keep": 0, "solved": 0, "hopeless": 0}
    unscored = 0
    pass_rates = []
    for rec in records:
        sc = scored_by_text.get(rec["text"])
        if sc is None:
            unscored += 1
            continue
        pr = sc["_curate"]["pass_rate"]
        pass_rates.append(pr)
        cls = classify(pr, keep_low, keep_high)
        buckets[cls] += 1
        if cls == "keep":
            # write the ORIGINAL record (drop the _curate annotation from the
            # training file; it lives in scored.jsonl for the record).
            kept.append({k: v for k, v in rec.items() if k != "_curate"})

    with open(out_path, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(pass_rates)
    print("\n" + "=" * 64)
    print("  Prompt curation by pass-rate (keep the Goldilocks middle)")
    print("=" * 64)
    if unscored:
        print(f"  ⚠ {unscored} record(s) not yet scored — run without --filter-only")
    hb = histogram(pass_rates)
    labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    print(f"  pass_rate distribution over {n} scored prompts:")
    for lab, c in zip(labels, hb):
        bar = "█" * int(40 * c / max(1, max(hb)))
        print(f"    {lab}  {c:>5}  {bar}")
    print("-" * 64)
    print(f"  band: keep {keep_low} < pass_rate < {keep_high}")
    print(f"  KEEP     {buckets['keep']:>6}  (rollouts disagree → advantage signal)")
    print(f"  drop solved   {buckets['solved']:>6}  (pass_rate ≥ {keep_high})")
    print(f"  drop hopeless {buckets['hopeless']:>6}  (pass_rate ≤ {keep_low})")
    frac = buckets["keep"] / max(1, n)
    print(f"  → wrote {len(kept)} prompts ({frac:.0%} of scored) to {out_path}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="GRPO prompt curation by expected advantage")
    ap.add_argument("--config", required=True, help="GRPO YAML (base model + init_adapters = SFT policy)")
    ap.add_argument("--train-jsonl", default=None, help="Prompts to curate (default: config data.train_jsonl)")
    ap.add_argument("--out", default=None, help="Curated (kept) jsonl (default: <train>.curated.jsonl)")
    ap.add_argument("--adapter", default=None, help="Policy adapter to roll out with (default: config init_adapters)")
    ap.add_argument("--group-size", type=int, default=6, help="rollouts per prompt (variance estimate)")
    ap.add_argument("--pass-thresh", type=float, default=0.10, help="CER ≤ this counts as a 'pass'")
    ap.add_argument("--keep-low", type=float, default=0.2, help="drop if pass_rate ≤ this (hopeless)")
    ap.add_argument("--keep-high", type=float, default=0.8, help="drop if pass_rate ≥ this (solved)")
    ap.add_argument("--max-prompts", type=int, default=None, help="only the first N (debug/subset)")
    ap.add_argument("--filter-only", action="store_true", help="re-filter existing scored.jsonl; no rollouts")
    ap.add_argument("--dry-run", action="store_true", help="print plan + counts, load no model")
    args = ap.parse_args()

    train_mod = _load_train_module()
    cfg = train_mod.load_config(args.config)
    train_jsonl = args.train_jsonl or cfg.get("data", {}).get("train_jsonl")
    if not train_jsonl or not os.path.exists(train_jsonl):
        print(f"[curate] ERROR: train jsonl not found: {train_jsonl}"); sys.exit(1)
    out_path = args.out or str(Path(train_jsonl).with_suffix(".curated" + Path(train_jsonl).suffix))
    scored_out = scored_path_for(train_jsonl)

    records = read_records(train_jsonl, args.max_prompts)
    print(f"[curate] {len(records)} prompts from {train_jsonl}")
    print(f"[curate] scored file: {scored_out}  |  curated out: {out_path}")

    if args.dry_run:
        already = load_scored(scored_out)
        print(f"[dry-run] group_size={args.group_size} → ~{len(records) * args.group_size} gens "
              f"(minus {len(already)} already scored). band: keep "
              f"{args.keep_low}<pass_rate<{args.keep_high}, pass=CER≤{args.pass_thresh}.")
        return

    if not args.filter_only:
        # ── Build the model once (same path as scripts/train.py / heldout eval) ──
        from train.grpo.rewards import RewardConfig
        from train.lora import load_adapters

        model = train_mod.load_model(cfg)
        train_mod.apply_lora(model, cfg)
        for attr in ("speech_tokenizer", "speaker_encoder"):
            sub = getattr(model, attr, None)
            if sub is not None:
                sub.freeze()
        if getattr(model, "speech_tokenizer", None) is None:
            print("[curate] ERROR: speech_tokenizer required for decode."); sys.exit(1)

        # Roll out under the STARTING policy — the SFT adapter GRPO will begin from,
        # so the pass-rate reflects where training actually starts.
        adapter = args.adapter or cfg.get("grpo", {}).get("init_adapters")
        if adapter and os.path.exists(adapter):
            load_adapters(model, adapter)
            print(f"[curate] rolling out under policy: {adapter}")
        else:
            print(f"[curate] WARNING: adapter not found ({adapter}); using BASE weights — "
                  f"pass rates will NOT reflect the SFT start.")

        g, t = cfg.get("grpo", {}), cfg.get("trainer", {})
        reward_cfg = RewardConfig.from_config(g, default_language=t.get("lang_code", "auto"))
        done_texts = set(load_scored(scored_out).keys())
        if done_texts:
            print(f"[curate] resuming — {len(done_texts)} prompts already scored")
        score_records(
            records, model, reward_cfg,
            group_size=args.group_size, pass_thresh=args.pass_thresh,
            lang_code=t.get("lang_code", "auto"),
            max_new_tokens=g.get("max_new_tokens", 240), temperature=g.get("temperature", 0.9),
            top_p=g.get("top_p", 0.95), top_k=g.get("top_k", 50),
            sample_rate=cfg.get("data", {}).get("target_sr", 24000),
            scored_out=scored_out, done_texts=done_texts,
        )

    scored_by_text = load_scored(scored_out)
    if not scored_by_text:
        print(f"[curate] ERROR: no scored prompts in {scored_out} "
              f"(run without --filter-only first)."); sys.exit(1)
    filter_and_report(records, scored_by_text, keep_low=args.keep_low,
                      keep_high=args.keep_high, out_path=out_path)


if __name__ == "__main__":
    main()
