#!/usr/bin/env python
"""
ttsds2_score.py — run the TTSDS2 distributional benchmark over one or more
directories of synthesized wavs against a directory of real speech.

Why a separate script: TTSDS2 (`ttsds` on PyPI) pins `numpy<2`, wants Python
3.10–3.12, and pulls a heavy torch/transformers/onnx stack. That conflicts with
the mlx-audio training env (numpy>=2, Python 3.13). Since TTSDS2 is a
*distributional* score — it only needs directories of wavs, the content need not
match — scoring is cleanly decoupled from generation. Run this inside an isolated
interpreter, e.g.:

    uv run --python 3.12 --with "numpy<2" --with ttsds \
        python scripts/ttsds2_score.py \
            --reference ~/Documents/exps/hindi/audio \
            --synth  SFT=~/ttsds2_out/SFT-baseline \
            --synth  GRPO=~/ttsds2_out/L-... \
            --multilingual --out ~/ttsds2_out/ttsds2_results.csv

TTSDS2 factors: Generic, Speaker, Prosody, Intelligibility (Environment is off by
default). The headline is the unweighted mean over factors, each computed as a
distributional distance to the *real* reference, so a higher score means the
synth distribution sits closer to real Hindi speech.

The paper (Minixhofer et al., 2506.19441) validates TTSDS2 across languages
without fine-tuning — which is exactly why it is worth trying for Hindi where
English-trained MOS predictors (DNSMOS/UTMOS) degrade zero-shot.
"""

import argparse
import os
import sys
from pathlib import Path


def _patch_hf_hub_use_auth_token():
    """pyannote-audio 3.4.0 (TTSDS2's Speaker factor) still calls
    hf_hub_download(use_auth_token=...), an argument removed in huggingface_hub
    >=1.0. transformers 5.x pins the new hub, so we can't downgrade without a
    cascade — instead translate the dead kwarg to `token` at the source. Must run
    before importing ttsds (which imports pyannote at module load)."""
    import huggingface_hub as hf

    orig = hf.hf_hub_download

    def _shim(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", tok if not isinstance(tok, bool) else None)
        return orig(*args, **kwargs)

    hf.hf_hub_download = _shim


def _patch_torch_load_weights_only():
    """PyTorch >=2.6 defaults torch.load(weights_only=True), which rejects the
    pyannote WeSpeaker checkpoint (it pickles torch.torch_version.TorchVersion).
    The checkpoint is an official pyannote HF release (trusted), so default the
    flag back to False when a caller (pyannote/lightning) doesn't set it."""
    import torch

    orig = torch.load

    def _shim(*args, **kwargs):
        # Force False: lightning's pl_load passes weights_only=True explicitly, so
        # setdefault is not enough. Safe here — this env only loads official,
        # trusted TTSDS2/pyannote checkpoints.
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _shim


def _purge_corrupt_cache(cache_dir: str):
    """TTSDS2 caches each benchmark's embeddings as .npy but does not write them
    atomically — an interrupted/OOM-killed run leaves a truncated (header-only)
    file, and *every* later run then dies with 'cannot reshape array of size 0'
    when it reads that poisoned entry. Scan and drop unreadable .npy up front so
    the suite recomputes them cleanly."""
    import glob

    import numpy as np

    removed = 0
    for f in glob.glob(os.path.join(cache_dir, "**", "*.npy"), recursive=True):
        try:
            np.load(f, allow_pickle=True)
        except Exception:
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[ttsds2] purged {removed} corrupt cache file(s) from {cache_dir}")


def _parse_synth(items):
    """['LABEL=path', ...] -> [(label, abspath)]. A bare path uses its dirname."""
    out = []
    for it in items:
        if "=" in it:
            label, path = it.split("=", 1)
        else:
            path = it
            label = Path(it).name
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            sys.exit(f"[ttsds2] synth dir not found: {path}")
        out.append((label, path))
    return out


def main():
    ap = argparse.ArgumentParser(description="TTSDS2 distributional scoring of TTS wav dirs")
    ap.add_argument("--reference", required=True,
                    help="Directory of REAL speech wavs (e.g. IndicVoices-R Hindi)")
    ap.add_argument("--synth", action="append", required=True,
                    help="LABEL=dir of synthesized wavs; repeatable")
    ap.add_argument("--out", default="ttsds2_results.csv", help="CSV to write")
    ap.add_argument("--multilingual", action="store_true",
                    help="Enable multilingual mode (required for Hindi and other non-English)")
    ap.add_argument("--cache-dir", default="~/.cache/ttsds",
                    help="Dir for TTSDS2's embedding/distribution cache (speeds up re-runs)")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Parallel distance workers. TTSDS2 defaults to cpu_count() (18 here); on "
                         "this machine >1 both OOMs (corrupting the cache) and deadlocks the "
                         "ThreadPoolExecutor distance path. Default 1 (serial, reliable).")
    ap.add_argument("--include-environment", action="store_true",
                    help="Include the noise/Environment factor (off by default)")
    args = ap.parse_args()

    _patch_hf_hub_use_auth_token()
    _patch_torch_load_weights_only()

    try:
        from ttsds import BenchmarkSuite
        from ttsds.util.dataset import DirectoryDataset
    except ImportError as e:
        sys.exit(
            f"[ttsds2] cannot import ttsds ({e}).\n"
            "Run this script inside an isolated env, e.g.:\n"
            '  uv run --python 3.12 --with "numpy<2" --with ttsds python scripts/ttsds2_score.py ...'
        )

    ref_dir = os.path.abspath(os.path.expanduser(args.reference))
    if not os.path.isdir(ref_dir):
        sys.exit(f"[ttsds2] reference dir not found: {ref_dir}")

    synth = _parse_synth(args.synth)
    datasets = [DirectoryDataset(path, name=label) for label, path in synth]
    reference_datasets = [DirectoryDataset(ref_dir, name="reference_real")]

    print(f"[ttsds2] reference : {ref_dir}")
    for label, path in synth:
        print(f"[ttsds2] synth     : {label:<24} {path}")
    print(f"[ttsds2] multilingual={args.multilingual}  environment={args.include_environment}\n")

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    _purge_corrupt_cache(cache_dir)

    suite = BenchmarkSuite(
        datasets=datasets,
        reference_datasets=reference_datasets,
        write_to_file=os.path.abspath(os.path.expanduser(args.out)),
        skip_errors=True,
        include_environment=args.include_environment,
        multilingual=args.multilingual,
        cache_dir=cache_dir,
        n_workers=args.n_workers,
    )

    suite.run()
    agg = suite.get_aggregated_results()

    # Pivot to the money table: factor (row) × adapter (column) of mean scores,
    # with OVERALL (TTSDS2's weighted headline) last. Higher = the synth
    # distribution sits closer to real Hindi speech.
    print("\n" + "=" * 72)
    print("  TTSDS2  (0–100, higher = closer to real speech; per factor + OVERALL)")
    print("=" * 72)
    try:
        pivot = agg.pivot(index="benchmark_category", columns="dataset", values="score_mean")
        order = ["GENERIC", "SPEAKER", "PROSODY", "INTELLIGIBILITY", "ENVIRONMENT", "OVERALL"]
        pivot = pivot.reindex([c for c in order if c in pivot.index])
        # Put reference_real first, then adapters as passed on the CLI.
        cols = (["reference_real"] if "reference_real" in pivot.columns else []) + \
               [lbl for lbl, _ in synth if lbl in pivot.columns]
        pivot = pivot[cols]
        print(pivot.round(2).to_string())
        print("\n  NOTE: SPEAKER uses VoxCeleb WeSpeaker/d-vector (English-trained, not "
              "swapped by\n  multilingual); read it as a *relative* signal. GENERIC / "
              "INTELLIGIBILITY / most of\n  PROSODY are genuinely multilingual (mHuBERT-147, "
              "XLSR, multilingual Whisper).")
    except Exception as e:
        print(f"[ttsds2] (pivot failed: {e}; raw table below)\n{agg}")
    print(f"\n[ttsds2] wrote {args.out}")


if __name__ == "__main__":
    main()
