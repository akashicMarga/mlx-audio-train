# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import posixpath
import random
import tarfile
from pathlib import Path

import huggingface_hub
from huggingface_hub.utils import EntryNotFoundError
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from .lm import Lm, LmConfig, config_personaplex_7b_v1, config_v0_1

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_TEXT_TOKENIZER = "tokenizer_spm_32k_3.model"
DEFAULT_MOSHI_WEIGHT = "model.safetensors"
DEFAULT_MIMI_WEIGHT = "tokenizer-e351c8d8-checkpoint125.safetensors"


def hf_hub_download(repo: str | None, path: str) -> str:
    if repo is None or repo == "":
        raise ValueError(f"the --hf-repo flag is required to retrieve {path}")
    return huggingface_hub.hf_hub_download(repo, path)


def hf_get(filename: str, hf_repo: str | None = None) -> str:
    if filename.startswith("hf://"):
        parts = filename[5:].split("/")
        if len(parts) < 3:
            raise ValueError(f"invalid hf:// URL, expected hf://org/repo/path: {filename!r}")
        repo_name = parts[0] + "/" + parts[1]
        rel_path = posixpath.normpath("/".join(parts[2:]))
        if rel_path.startswith("..") or rel_path.startswith("/"):
            raise ValueError(f"path traversal detected in hf:// URL: {filename!r}")
        return hf_hub_download(repo_name, rel_path)
    if filename.startswith("file://"):
        return filename[7:]
    if hf_repo is not None and not Path(filename).exists():
        return hf_hub_download(hf_repo, filename)
    return filename


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def seed_all(seed: int | None) -> None:
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def get_lm_config(config_path: str | None, hf_repo: str) -> LmConfig:
    if config_path is None:
        config_path = hf_hub_download(hf_repo, "config.json")
    config_path = hf_get(config_path, hf_repo)
    import json

    with open(config_path, "r", encoding="utf-8") as fobj:
        data = json.load(fobj)
    if "dim" not in data:
        if hf_repo == "nvidia/personaplex-7b-v1":
            return config_personaplex_7b_v1()
        return config_v0_1()
    return LmConfig.from_config_dict(data)


def get_or_download_model_file(
    hf_repo: str,
    quantized: int | None,
    explicit_model_file: str | None,
) -> tuple[str, bool]:
    if explicit_model_file is not None:
        return hf_get(explicit_model_file, hf_repo), False
    if quantized in (4, 8):
        quantized_name = f"model.q{quantized}.safetensors"
        try:
            return hf_hub_download(hf_repo, quantized_name), True
        except EntryNotFoundError:
            pass
    return hf_hub_download(hf_repo, DEFAULT_MOSHI_WEIGHT), False


def get_or_download_tokenizer(hf_repo: str, tokenizer_file: str | None) -> str:
    if tokenizer_file is None:
        return hf_hub_download(hf_repo, DEFAULT_TEXT_TOKENIZER)
    return hf_get(tokenizer_file, hf_repo)


def get_or_download_mimi(hf_repo: str, mimi_file: str | None) -> str:
    if mimi_file is None:
        return hf_hub_download(hf_repo, DEFAULT_MIMI_WEIGHT)
    return hf_get(mimi_file, hf_repo)


def is_pytorch_weights(file: str) -> bool:
    with safe_open(file, framework="np") as sf:
        keys = list(sf.keys())
    if "out_norm.alpha" in keys:
        return True
    return any(k.endswith("in_proj_weight") for k in keys)


def load_lm_weights(
    model: Lm,
    lm_config: LmConfig,
    model_file: str,
    quantized: int | None,
) -> None:
    file_name = Path(model_file).name
    has_prequantized = f".q{quantized}." in file_name if quantized in (4, 8) else False
    if has_prequantized:
        group_size = 32 if quantized == 4 else 64
        nn.quantize(model, bits=quantized, group_size=group_size)

    if is_pytorch_weights(model_file):
        model.load_pytorch_weights(model_file, lm_config, strict=True)
    else:
        model.load_weights(model_file, strict=True)

    if quantized in (4, 8) and not has_prequantized:
        group_size = 32 if quantized == 4 else 64
        nn.quantize(model, bits=quantized, group_size=group_size)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar contents safely, rejecting path traversal attempts."""
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest_resolved)):
            raise ValueError(f"tar member {member.name!r} escapes destination directory")
    tar.extractall(path=dest)


def get_voice_prompt_dir(voice_prompt_dir: str | None, hf_repo: str) -> str:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz_path = Path(voices_tgz)
    voices_dir = voices_tgz_path.parent / "voices"
    if not voices_dir.exists():
        with tarfile.open(voices_tgz_path, "r:gz") as tar:
            _safe_extractall(tar, voices_tgz_path.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a voices/ directory")
    return str(voices_dir)


def resolve_voice_prompt(
    voice: str | None,
    voice_prompt: str | None,
    voice_prompt_dir: str,
) -> str:
    if voice is None and voice_prompt is None:
        raise ValueError("one of --voice or --voice-prompt is required")
    selected = voice_prompt if voice_prompt is not None else voice
    assert selected is not None
    if "." not in Path(selected).name:
        selected = f"{selected}.pt"
    path = Path(selected)
    if not path.is_absolute():
        path = Path(voice_prompt_dir) / selected
    if not path.exists():
        raise FileNotFoundError(f"voice prompt not found: {path}")
    return str(path)
