"""
Load / export Pocket TTS weights in the official `model.safetensors` layout.

A released file holds `flow_lm.*` (backbone + heads) and `mimi.*` (codec).
Training checkpoints written here keep that layout, so they run unchanged with
`pocket-tts generate --config <exported yaml>`.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import yaml
from mlx.utils import tree_flatten

from models.pocket_tts.config import flowlm_args_from_raw, load_raw_config
from models.pocket_tts.model import FlowLM


def resolve_path(path: str) -> str:
    """Local path for a pocket-tts style reference: local file or hf://repo/file[@rev]."""
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    spec, _, revision = path.removeprefix("hf://").partition("@")
    owner, repo, *rest = spec.split("/")
    return hf_hub_download(f"{owner}/{repo}", "/".join(rest), revision=revision or None)


def load_flow_lm_state(path: str) -> dict[str, mx.array]:
    """flow_lm tensors without the prefix, from a model.safetensors (or our checkpoints)."""
    state = mx.load(resolve_path(path))
    return {k.removeprefix("flow_lm."): v for k, v in state.items() if k.startswith("flow_lm.")}


def load_mimi_state(path: str) -> dict[str, mx.array]:
    state = mx.load(resolve_path(path))
    return {k: v for k, v in state.items() if k.startswith("mimi.")}


def model_state(model: FlowLM) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def build_flow_lm(config_path: str, overrides: dict | None = None,
                  weights: str | None = None, dtype=mx.float32) -> tuple[FlowLM, dict]:
    """(FlowLM, raw config). With `weights`, loads them strictly (voice projection optional)."""
    raw = load_raw_config(config_path, overrides)
    model = FlowLM(flowlm_args_from_raw(raw))
    if weights:
        state = load_flow_lm_state(weights)
        if "speaker_proj_weight" not in state:
            # *-without-voice-cloning releases ship no voice projection: run text-only.
            model.pop("speaker_proj_weight")
        model.load_weights(list(state.items()), strict=True)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    model.freeze_buffers()
    return model, raw


def export_pocket_tts(model: FlowLM, raw_config: dict, mimi_source: str, out_dir: str | Path,
                      params: dict[str, mx.array] | None = None) -> Path:
    """Write out_dir/model.safetensors (flow_lm + mimi) and out_dir/config.yaml.

    `params` overrides the model's own tensors (e.g. the EMA shadow).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model_state(model)
    if params:
        state.update(params)
    tensors = {f"flow_lm.{k}": v.astype(mx.float32) for k, v in state.items()}
    tensors.update(load_mimi_state(mimi_source))
    weights_file = out_dir / "model.safetensors"
    mx.save_safetensors(str(weights_file), tensors)

    cfg = dict(raw_config)
    cfg["weights_path"] = str(weights_file.resolve())
    cfg.pop("weights_path_without_voice_cloning", None)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return weights_file
