#!/usr/bin/env python3
"""
Held-out comparison of Pocket TTS checkpoints (e.g. a distilled student before / after
GRPO) — the Pocket counterpart of scripts/grpo_heldout_eval.py.

Every checkpoint generates the same prompts with the same seeds at inference settings
(sigma 0, threshold EOS); the reward stack of the GRPO config scores them (CER/WER,
UTMOS, DNSMOS, speaker similarity, length). Prints mean / median per metric and a
paired Wilcoxon p-value against the first checkpoint (the baseline).

Usage:
    python scripts/pocket_tts_grpo_heldout_eval.py --grpo-config configs/pocket_tts/hindi/grpo_student_4l.yaml \\
        --data data/pocket_latents/hindi_valid/manifest.jsonl --num-prompts 100 --seeds 0 1 \\
        --checkpoints checkpoints/pocket-hindi-4l/step_0100000/config.yaml \\
                      checkpoints/pocket-hindi-4l-grpo/step_0000400/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.data import LatentDataset, load_tokenizer  # noqa: E402
from models.pocket_tts.grpo import Prompts, RolloutConfig, Scorer, evaluate_prompts  # noqa: E402
from models.pocket_tts.text import normalizer  # noqa: E402
from models.pocket_tts.train_utils import merge  # noqa: E402
from models.pocket_tts.weights import build_flow_lm  # noqa: E402
from train.grpo.rewards import RewardConfig  # noqa: E402

LOWER_IS_BETTER = {"cer", "wer", "no_eos", "silence_frac"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="config.yaml of each checkpoint; the first is the baseline")
    ap.add_argument("--grpo-config", required=True, help="supplies rewards, data and rollout settings")
    ap.add_argument("--data", required=True, help="held-out latent manifest")
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=None, help="write per-prompt results as JSON")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pocket_tts_grpo import DEFAULTS

    cfg = merge(DEFAULTS, yaml.safe_load(Path(args.grpo_config).read_text()))
    g, d = cfg["grpo"], cfg["data"]
    rc = RolloutConfig(group_size=1, temp=g["temp"], sigma=0.0, max_frames=int(g["max_sec"] * 12.5),
                       min_frames=g["min_frames"], eos_threshold=g["eos_threshold"], train_eos=False)
    reward_cfg = RewardConfig(rewards=g["rewards"], adv_norm=g["adv_norm"])

    from models.pocket_tts.codec import Mimi

    decode = Mimi(args.checkpoints[0]).decode
    results = {}
    for n, ckpt in enumerate(args.checkpoints):
        model, _ = build_flow_lm(ckpt, weights="config")
        tok = load_tokenizer(d.get("tokenizer_path") or model.args.tokenizer_path)
        ds = LatentDataset(args.data, tok, seed=7, max_voice_prompt_sec=d["max_voice_prompt_sec"],
                           prompt_trim_max_sec=0.0, final_punct_dropout=0.0)
        prompts = Prompts(ds, tok, normalizer(d.get("normalize_text")), skip_unscorable=d["skip_unscorable"],
                          max_prompt_sec=d["max_prompt_sec"]).first(args.num_prompts)
        res = evaluate_prompts(model, prompts, Scorer(model, reward_cfg, decode), rc, seeds=args.seeds)
        label = f"[{n}] {ckpt}"
        results[label] = res
        print(f"[heldout] {label}: " + "  ".join(f"{k}={v.mean():.4f}" for k, v in sorted(res.items())))

    base_label = next(iter(results))
    base = results[base_label]
    metrics = sorted(set.intersection(*(set(r) for r in results.values())))
    print("\n| checkpoint | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for ckpt, res in results.items():
        cells = []
        for k in metrics:
            cell = f"{res[k].mean():.4f} (med {np.median(res[k]):.4f})"
            if ckpt != base_label and len(res[k]) == len(base[k]) and np.any(res[k] != base[k]):
                try:
                    from scipy.stats import wilcoxon

                    cell += f" p={wilcoxon(res[k], base[k]).pvalue:.2g}"
                except Exception:
                    pass
            cells.append(cell)
        print(f"| {ckpt} | " + " | ".join(cells) + " |")
    print("\n(lower is better: " + ", ".join(sorted(LOWER_IS_BETTER & set(metrics))) + "; higher for the rest)")
    if args.out:
        Path(args.out).write_text(json.dumps({c: {k: v.tolist() for k, v in r.items()} for c, r in results.items()},
                                             indent=1))


if __name__ == "__main__":
    main()
