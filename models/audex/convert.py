"""
Disk-frugal converter: nvidia/Nemotron-Labs-Audex-2B (HF) -> MLX checkpoint.

Produces an output dir with:
    lm.safetensors                 (NemotronDense LM, bf16, audio-encoder dropped)
    speech_decoder.safetensors     (Audex causal speech decoder, bf16)
    lm_config.json / speech_decoder_config.json
    tokenizer.json / tokenizer_config.json / special_tokens_map.json

The full HF checkpoint (~5.8 GB) bundles a Whisper audio encoder we don't need
for text/TTS. To respect tight disk, we download one shard at a time, extract
only the tensors we keep, then delete the shard before fetching the next.

Usage:
    python -m models.audex.convert --out /path/to/audex_mlx
    # or a specific HF revision / local snapshot via --repo / --local-src
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

REPO = "nvidia/Nemotron-Labs-Audex-2B"
FULL = "checkpoint_folder_full"
DECODER = "audex_causal_speech_decoder"
LM_SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def _hf_get(repo, filename, scratch):
    from huggingface_hub import hf_hub_download
    # local_dir download in hf_hub>=1.0 writes directly (no cache-blob duplication),
    # which keeps peak disk low — important on this tight volume.
    return hf_hub_download(repo_id=repo, filename=filename, local_dir=str(scratch))


def _keep_lm_key(k: str) -> bool:
    return (k.startswith("model.") or k == "lm_head.weight") and not k.startswith("model.audio")


def convert_lm(out: Path, scratch: Path, repo: str):
    # Source tensors are bf16; load natively with mx.load (numpy can't hold bf16).
    weights = {}
    for shard in LM_SHARDS:
        print(f"[convert] LM shard {shard} ...", flush=True)
        path = _hf_get(repo, f"{FULL}/{shard}", scratch)
        shard_w = mx.load(path, format="safetensors")
        for k, v in shard_w.items():
            if _keep_lm_key(k):
                weights[k] = v.astype(mx.bfloat16)
        mx.eval(list(weights.values()))
        del shard_w
        os.remove(path)                              # free disk before next shard
        print(f"[convert]   kept {len(weights)} tensors so far; shard removed", flush=True)
    mx.save_safetensors(str(out / "lm.safetensors"), weights)
    print(f"[convert] wrote lm.safetensors ({len(weights)} tensors)", flush=True)


def convert_decoder(out: Path, scratch: Path, repo: str):
    print("[convert] speech decoder ...", flush=True)
    path = _hf_get(repo, f"{DECODER}/model.safetensors", scratch)
    src = mx.load(path, format="safetensors")
    weights = {}
    for k, v in src.items():
        # PatchHead: torch `module.head.proj.weight` -> mlx `module.head_proj.weight`
        if k == "module.head.proj.weight":
            k = "module.head_proj.weight"
        # Conv1d weights: torch [out, in/groups, k] -> mlx [out, k, in/groups]
        if k in ("module.lookahead_conv.weight", "module.lookahead_proj.weight"):
            v = mx.swapaxes(v, 1, 2)
        weights[k] = v.astype(mx.bfloat16)
    mx.eval(list(weights.values()))
    del src
    os.remove(path)
    mx.save_safetensors(str(out / "speech_decoder.safetensors"), weights)
    print(f"[convert] wrote speech_decoder.safetensors ({len(weights)} tensors)", flush=True)


def write_configs(out: Path, scratch: Path, repo: str):
    from .lm import LMConfig
    from .speech_decoder import SpeechDecoderConfig
    import dataclasses

    lm_cfg = json.loads(Path(_hf_get(repo, f"{FULL}/config.json", scratch)).read_text())
    dec_cfg = json.loads(Path(_hf_get(repo, f"{DECODER}/config.json", scratch)).read_text())

    (out / "lm_config.json").write_text(json.dumps(dataclasses.asdict(LMConfig.from_hf(lm_cfg)), indent=2))
    (out / "speech_decoder_config.json").write_text(
        json.dumps(dataclasses.asdict(SpeechDecoderConfig.from_hf(dec_cfg)), indent=2))

    for fn in TOKENIZER_FILES:
        src = _hf_get(repo, f"{FULL}/{fn}", scratch)
        shutil.copy(src, out / fn)
    print("[convert] wrote configs + tokenizer")


def main():
    ap = argparse.ArgumentParser(description="Convert Nemotron-Labs-Audex-2B to MLX (text + TTS)")
    ap.add_argument("--out", required=True, help="output MLX checkpoint dir")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--scratch", default=None, help="temp dir for shard downloads (default: <out>/.scratch)")
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else out / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    write_configs(out, scratch, args.repo)
    if not args.skip_lm:
        convert_lm(out, scratch, args.repo)
    if not args.skip_decoder:
        convert_decoder(out, scratch, args.repo)

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[convert] done -> {out}")


if __name__ == "__main__":
    main()
