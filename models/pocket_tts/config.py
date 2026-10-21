"""
Pocket TTS FlowLM architecture, read from an official pocket-tts YAML config.

The official configs (kyutai-labs/pocket-tts, pocket_tts/config/*.yaml) describe
both the FlowLM (backbone + sampler head) and the Mimi codec. Only the FlowLM
part is needed here: distillation runs on precomputed Mimi latents, and the
codec weights are copied verbatim into the exported checkpoint.

Dotted overrides let one released config serve every depth/width variant, the
same way the official trainer does it:

    load_flowlm_args("english_2026-09_24l.yaml", {"flow_lm.transformer.num_layers": 6})
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Sampler-head types -> number of time conditions the head takes.
FLOW_TIME_CONDS = {"lsd": 2, "flow_matching": 1, "drifting": 0, "pointwise": 0}


@dataclass
class FlowLMArgs:
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    max_period: float
    attn_dim: int | None
    num_kv_heads: int | None
    n_bins: int
    lut_rank: int | None
    z_dim: int | None
    flow_dim: int
    flow_depth: int
    flow_type: str
    ldim: int               # latent width the flow models (mimi.quantizer.dimension)
    voice_dim: int          # width of voice-prompt latents (mimi.inner_dim or seanet.dimension)
    insert_bos_before_voice: bool
    tokenizer: str
    tokenizer_path: str

    @property
    def head_dim(self) -> int:
        return (self.attn_dim or self.d_model) // self.num_heads

    @property
    def z_width(self) -> int:
        """Width of the backbone output z that the sampler and EOS heads read."""
        return self.z_dim or self.d_model

    @property
    def num_time_conds(self) -> int:
        return FLOW_TIME_CONDS[self.flow_type]


def apply_overrides(raw: dict, overrides: dict[str, Any] | None) -> dict:
    raw = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"config has no section {key!r} (from override {dotted!r})")
            node = node[key]
        node[leaf] = value
    return raw


def resolve_path(path: str) -> str:
    """Local path for a pocket-tts style reference: local file or hf://owner/repo/file[@rev]."""
    path = str(path)
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    spec, _, revision = path.removeprefix("hf://").partition("@")
    owner, repo, *rest = spec.split("/")
    return hf_hub_download(f"{owner}/{repo}", "/".join(rest), revision=revision or None)


def load_raw_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """A pocket-tts YAML config (local or hf://), with dotted overrides applied.

    Relative weights/tokenizer paths inside an hf:// config are made hf:// paths of the
    same repo (community configs often say `tokenizer_path: tokenizer.model`)."""
    raw = _read_yaml(resolve_path(path))
    if str(path).startswith("hf://"):
        spec, _, rev = str(path).removeprefix("hf://").partition("@")
        repo = "/".join(spec.split("/")[:2])
        fix = lambda v: v if not v or "://" in v or v.startswith("/") else f"hf://{repo}/{v}" + (f"@{rev}" if rev else "")  # noqa: E731
        for key in ("weights_path", "weights_path_without_voice_cloning"):
            if isinstance(raw.get(key), str):
                raw[key] = fix(raw[key])
        lut = raw["flow_lm"]["lookup_table"]
        lut["tokenizer_path"] = fix(lut["tokenizer_path"])
    return apply_overrides(raw, overrides)


def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flowlm_args_from_raw(raw: dict) -> FlowLMArgs:
    fl, mimi = raw["flow_lm"], raw["mimi"]
    tr, lut, flow = fl["transformer"], fl["lookup_table"], fl["flow"]
    if lut["dim"] != tr["d_model"]:
        raise ValueError("lookup_table.dim must equal transformer.d_model")
    ff = tr.get("dim_feedforward") or int(tr["d_model"] * tr["hidden_scale"])
    flow_type = flow.get("type", "lsd")
    if flow_type not in FLOW_TIME_CONDS:
        raise ValueError(f"unknown flow type {flow_type!r}")
    return FlowLMArgs(
        d_model=tr["d_model"],
        num_heads=tr["num_heads"],
        num_layers=tr["num_layers"],
        dim_feedforward=ff,
        max_period=float(tr["max_period"]),
        attn_dim=tr.get("attn_dim"),
        num_kv_heads=tr.get("num_kv_heads"),
        n_bins=lut["n_bins"],
        lut_rank=lut.get("rank"),
        z_dim=fl.get("z_dim"),
        flow_dim=flow["dim"],
        flow_depth=flow["depth"],
        flow_type=flow_type,
        ldim=mimi["quantizer"]["dimension"],
        voice_dim=mimi.get("inner_dim") or mimi["seanet"]["dimension"],
        insert_bos_before_voice=bool(fl.get("insert_bos_before_voice", False)),
        tokenizer=lut.get("tokenizer", "sentencepiece"),
        tokenizer_path=lut["tokenizer_path"],
    )


def load_flowlm_args(path: str | Path, overrides: dict[str, Any] | None = None) -> FlowLMArgs:
    return flowlm_args_from_raw(load_raw_config(path, overrides))
