#!/usr/bin/env python3
"""
EXPERIMENTAL: give a public Pocket TTS teacher a z bottleneck (z_dim) so it can be
distilled into a narrower student (d_model < teacher's).

Why: distillation regresses the student's z onto the teacher's with the heads
frozen, so the z width must match. The released 24-layer teachers have no z_proj
(their heads read d_model = 1024), so a student narrower than 1024 is impossible
without first changing the teacher. Kyutai's small models (18m/7m/2m) come from
teachers trained with z_dim=64, which are not released.

This folds the best rank-r bottleneck (reduced-rank regression on the heads'
inputs, see models/pocket_tts/distill.py:fold_z_bottleneck) into the teacher:
    z' = z_proj(out_norm(h)),  cond_embed' / out_eos' read z'
using z statistics collected on your data. It prints the relative error of the
head inputs; the folded teacher is written in the official layout so you can
listen to it / measure its WER with `pocket-tts generate` BEFORE distilling.

Usage:
    python scripts/pocket_tts_fold_bottleneck.py \\
        --config configs/pocket_tts/models/english_2026-09_24l.yaml \\
        --weights hf://kyutai/pocket-tts/languages/english_2026-09_24l/model.safetensors \\
        --data data/pocket_latents/train/manifest.jsonl --z-dim 64 \\
        --out checkpoints/pocket-teacher-24l-z64
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.config import apply_overrides, flowlm_args_from_raw  # noqa: E402
from models.pocket_tts.data import LatentDataset, batches, load_tokenizer  # noqa: E402
from models.pocket_tts.distill import fold_z_bottleneck  # noqa: E402
from models.pocket_tts.model import FlowLM  # noqa: E402
from models.pocket_tts.weights import build_flow_lm, export_pocket_tts, model_state  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="latent manifest (pocket_tts_precompute_latents.py)")
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--frames", type=int, default=20000, help="z samples used for the fit")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    teacher, raw = build_flow_lm(args.config, weights=args.weights)
    if teacher.args.z_dim is not None:
        sys.exit(f"teacher already has z_dim={teacher.args.z_dim}")
    ds = LatentDataset(args.data, load_tokenizer(teacher.args.tokenizer_path), final_punct_dropout=0.0)
    zs, n = [], 0
    for batch in batches(ds, args.batch_size, 1, shuffle=False):
        lat = (batch["latents"] - teacher.emb_mean) / teacher.emb_std
        keep = np.ones(lat.shape[0], dtype=bool)
        z = teacher.backbone_z(lat, batch["text_tokens"], batch["voice"], batch["voice_lens"], keep, keep)
        z = np.array(z)[np.array(batch["mask"])]
        zs.append(z)
        n += len(z)
        if n >= args.frames:
            break
    Z = mx.array(np.concatenate(zs)[: args.frames])
    print(f"[fold] fitting rank-{args.z_dim} bottleneck on {Z.shape[0]} frames of z (d={Z.shape[1]})")

    new, rel_err = fold_z_bottleneck(teacher, Z, args.z_dim)
    print(f"[fold] relative error of the head inputs: {rel_err:.4f}")

    raw = apply_overrides(raw, {"flow_lm.z_dim": args.z_dim})
    folded = FlowLM(flowlm_args_from_raw(raw))
    if "speaker_proj_weight" not in teacher:
        folded.pop("speaker_proj_weight")
    state = {k: v for k, v in model_state(teacher).items() if k not in new}
    folded.load_weights(list(state.items()) + list(new.items()), strict=True)
    out = export_pocket_tts(folded, raw, args.weights, args.out)
    print(f"[fold] wrote {out} and {Path(args.out) / 'config.yaml'}")
    print("       check it first:  pocket-tts generate --config "
          f"{Path(args.out) / 'config.yaml'} --text 'Hello there.'")


if __name__ == "__main__":
    main()
