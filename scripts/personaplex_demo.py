#!/usr/bin/env python3
"""
personaplex_demo.py — Gradio demo for PersonaPlex Hindi checkpoint preview.

Lets you load the base PersonaPlex model or a selected Hindi LoRA checkpoint,
feed a WAV file, and listen to the generated response while training continues.

Usage:
    python scripts/personaplex_demo.py
    python scripts/personaplex_demo.py --adapter checkpoints/personaplex-hindi/checkpoint-best
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr
import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn
import yaml

from models.personaplex import Lm, LmGen
from models.personaplex.persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    get_or_download_tokenizer,
    get_voice_prompt_dir,
    load_lm_weights,
    resolve_voice_prompt,
    seed_all,
    wrap_with_system_tags,
)
from models.personaplex.training import apply_lora_to_transformer
from models.personaplex.utils import Sampler, reshape_input_tokens
from train.lora import load_personaplex_adapters


DEFAULT_ARTIFACTS_ROOT = Path(
    os.environ.get(
        "MLX_AUDIO_ARTIFACTS_DIR",
        "/Users/akashsingh/Documents/mlx-audio-artifacts",
    )
)
CHECKPOINTS_ROOT = Path(
    os.environ.get(
        "MLX_AUDIO_CHECKPOINTS_DIR",
        str(DEFAULT_ARTIFACTS_ROOT / "checkpoints"),
    )
)
LOCAL_VOICES_ROOT = Path(
    os.environ.get(
        "MLX_AUDIO_VOICES_DIR",
        str(DEFAULT_ARTIFACTS_ROOT / "voices"),
    )
)
REPO_CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
REPO_VOICES_ROOT = REPO_ROOT / "voices"
BASE_LABEL = "── Base model (no Hindi adapter) ──"
FRAME_SIZE = 1920
SAMPLE_RATE = 24000

_CURRENT_MODEL_KEY: Optional[str] = None
_CURRENT_MODEL_BUNDLE: Optional[dict] = None


def scan_checkpoints() -> list[str]:
    choices = [BASE_LABEL]
    roots = []
    for root in (CHECKPOINTS_ROOT, REPO_CHECKPOINTS_ROOT):
        if root.exists() and root not in roots:
            roots.append(root)

    for root in roots:
        for adapter_file in sorted(root.rglob("adapters.npz")):
            rel = adapter_file.parent.relative_to(root)
            label = " / ".join(rel.parts)
            if label not in choices:
                choices.append(label)
    return choices


def label_to_adapter_path(label: str) -> Optional[str]:
    if not label or label == BASE_LABEL:
        return None
    rel = Path(*[part.strip() for part in label.split("/")])
    for root in (CHECKPOINTS_ROOT, REPO_CHECKPOINTS_ROOT):
        candidate = root / rel
        if candidate.exists():
            return str(candidate)
    return str(CHECKPOINTS_ROOT / rel)


def _resolve_adapter_file(adapter_path: str | None) -> Optional[str]:
    if not adapter_path:
        return None

    p = Path(adapter_path.strip())
    if p.is_file() and p.name == "adapters.npz":
        return str(p)

    if p.is_dir():
        direct = p / "adapters.npz"
        if direct.exists():
            return str(direct)

        checkpoints = sorted(p.glob("checkpoint-*/adapters.npz"))
        if checkpoints:
            chosen = checkpoints[-1]
            print(f"[personaplex-demo] Found {len(checkpoints)} checkpoint(s), using: {chosen.parent.name}")
            return str(chosen)

    print(f"[personaplex-demo] WARNING: no adapters.npz found at '{adapter_path}'")
    return None


def _load_model_config(adapter_path: str | None) -> dict:
    if not adapter_path:
        return {}

    candidates = [
        Path(adapter_path),
        Path(adapter_path).parent,
        Path(adapter_path).parent.parent,
    ]
    for candidate in candidates:
        cfg_path = candidate / "model_config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    return {}


def _infer_lora_params(adapter_file: str, saved_cfg: dict) -> tuple[int, float, float]:
    rank = 16
    with np.load(adapter_file) as data:
        for key in data.files:
            if key.endswith("lora_a"):
                rank = int(data[key].shape[0])
                break

    lora_cfg = saved_cfg.get("lora", {}) if isinstance(saved_cfg, dict) else {}
    alpha = float(lora_cfg.get("alpha", rank))
    dropout = float(lora_cfg.get("dropout", 0.0))
    return rank, alpha, dropout


def _get_voice_prompt_dir(adapter_path: str | None) -> str:
    saved_cfg = _load_model_config(adapter_path)
    model_cfg = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}
    hf_repo = model_cfg.get("hf_repo", DEFAULT_HF_REPO)
    return get_voice_prompt_dir(model_cfg.get("voice_prompt_dir"), hf_repo)


def _has_text_supervision(adapter_path: str | None) -> bool:
    if not adapter_path:
        return True

    path = Path(adapter_path)
    candidates = [
        path / "train_log.jsonl",
        path.parent / "train_log.jsonl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if "text_loss" in entry and float(entry["text_loss"]) > 0:
                        return True
            return False
        except Exception:
            continue
    return True


def _load_bundle(adapter_path: str | None) -> dict:
    global _CURRENT_MODEL_KEY, _CURRENT_MODEL_BUNDLE

    adapter_file = _resolve_adapter_file(adapter_path)
    cache_key = adapter_file or "__base__"
    if _CURRENT_MODEL_KEY == cache_key and _CURRENT_MODEL_BUNDLE is not None:
        return _CURRENT_MODEL_BUNDLE

    saved_cfg = _load_model_config(adapter_path)
    model_cfg = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}

    hf_repo = model_cfg.get("hf_repo", DEFAULT_HF_REPO)
    lm_config_name = model_cfg.get("lm_config")
    quantized = model_cfg.get("quantized")
    model_file = model_cfg.get("model_file")

    print(f"[personaplex-demo] Loading base PersonaPlex from: {hf_repo}")
    lm_config = get_lm_config(lm_config_name, hf_repo)
    tokenizer_file = get_or_download_tokenizer(hf_repo, model_cfg.get("tokenizer"))
    mimi_file = get_or_download_mimi(hf_repo, model_cfg.get("mimi_weight"))
    base_file, _ = get_or_download_model_file(hf_repo, quantized, model_file)

    model = Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    load_lm_weights(model, lm_config, base_file, quantized)

    if adapter_file:
        rank, alpha, dropout = _infer_lora_params(adapter_file, saved_cfg)
        apply_lora_to_transformer(model, rank=rank, alpha=alpha, dropout=dropout)
        load_personaplex_adapters(model, adapter_file)
        print(f"[personaplex-demo] Loaded checkpoint: {adapter_file} (rank={rank}, alpha={alpha})")

    voice_prompt_dir = get_voice_prompt_dir(model_cfg.get("voice_prompt_dir"), hf_repo)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)

    _CURRENT_MODEL_KEY = cache_key
    _CURRENT_MODEL_BUNDLE = {
        "model": model,
        "hf_repo": hf_repo,
        "mimi_file": mimi_file,
        "voice_prompt_dir": voice_prompt_dir,
        "text_tokenizer": text_tokenizer,
        "adapter_file": adapter_file,
    }
    return _CURRENT_MODEL_BUNDLE


def scan_voices(adapter_path: str | None = None) -> list[str]:
    voices: list[str] = []

    builtin_dir = Path(_get_voice_prompt_dir(adapter_path))
    voices.extend(p.stem for p in sorted(builtin_dir.glob("*.pt")))

    for local_dir in (LOCAL_VOICES_ROOT, REPO_VOICES_ROOT):
        if local_dir.exists():
            voices.extend(str(p.resolve()) for p in sorted(local_dir.glob("*.pt")))
            voices.extend(str(p.resolve()) for p in sorted(local_dir.glob("*.npz")))

    seen = set()
    deduped = []
    for voice in voices:
        if voice in seen:
            continue
        seen.add(voice)
        deduped.append(voice)

    return deduped or ["NATF2"]


def generate_preview(
    input_audio: str,
    adapter_label: str,
    voice_name: str,
    custom_voice_prompt: str,
    text_prompt: str,
    seed: int,
    audio_temp: float,
    text_temp: float,
    audio_topk: int,
    text_topk: int,
):
    if not input_audio:
        raise gr.Error("Provide an input WAV or MP3 file.")

    adapter_path = label_to_adapter_path(adapter_label)
    bundle = _load_bundle(adapter_path)
    model = bundle["model"]
    text_tokenizer = bundle["text_tokenizer"]
    has_text_supervision = _has_text_supervision(adapter_path)

    seed_all(seed)

    gen = LmGen(
        model=model,
        max_steps=100000,
        text_sampler=Sampler(temp=text_temp, top_k=text_topk),
        audio_sampler=Sampler(temp=audio_temp, top_k=audio_topk),
        check=False,
        audio_silence_frame_cnt=int(0.5 * 12.5),
    )

    voice_prompt_path = resolve_voice_prompt(
        voice=voice_name or None,
        voice_prompt=custom_voice_prompt.strip() or None,
        voice_prompt_dir=bundle["voice_prompt_dir"],
    )
    gen.load_voice_prompt_embeddings(voice_prompt_path)
    gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt))
        if text_prompt.strip()
        else None
    )
    gen.reset_streaming()
    gen.step_system_prompts()

    audio_tokenizer = rustymimi.Tokenizer(bundle["mimi_file"], num_codebooks=8)
    in_pcms, _ = sphn.read(input_audio, sample_rate=SAMPLE_RATE)
    if in_pcms.ndim == 1:
        in_pcms = in_pcms[None, :]
    total_samples = in_pcms.shape[-1]
    steps = (total_samples + FRAME_SIZE - 1) // FRAME_SIZE

    all_out_pcm = []
    generated_text_tokens = []
    text_token_map = ["EPAD", "BOS", "EOS", "PAD"]

    for idx in range(steps):
        start = idx * FRAME_SIZE
        end = min((idx + 1) * FRAME_SIZE, total_samples)
        pcm_data = in_pcms[:, start:end]
        if pcm_data.shape[-1] < FRAME_SIZE:
            pad = FRAME_SIZE - pcm_data.shape[-1]
            pcm_data = np.pad(pcm_data, ((0, 0), (0, pad)), mode="constant")

        encoded = audio_tokenizer.encode_step(pcm_data[None, 0:1])
        model_input = reshape_input_tokens(encoded, gen.user_codebooks)
        text_token = gen.step(input_tokens=model_input)

        if text_token is not None:
            token_id = int(text_token[0].item())
            if token_id in (0, 1, 2, 3):
                generated_text_tokens.append(text_token_map[token_id])
            else:
                piece = text_tokenizer.id_to_piece(token_id)
                generated_text_tokens.append(piece.replace("▁", " "))

        audio_tokens = gen.last_audio_tokens()
        if audio_tokens is not None:
            decode_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(decode_tokens)
            all_out_pcm.append(out_pcm)

    if not all_out_pcm:
        raise gr.Error("No output audio was generated.")

    all_out_pcm_np = np.concatenate(all_out_pcm, axis=-1)
    out_samples = min(total_samples, all_out_pcm_np.shape[-1])
    preview_audio = all_out_pcm_np[0, 0, :out_samples].astype(np.float32)
    preview_audio = np.clip(preview_audio, -1.0, 1.0)
    generated_text = "".join(generated_text_tokens).strip()
    if not has_text_supervision:
        generated_text = (
            "This checkpoint was trained with empty text_tokens, so decoded text here is expected "
            "to be unreliable. Use the audio preview to judge Hindi adaptation."
        )

    adapter_status = adapter_label if adapter_label and adapter_label != BASE_LABEL else "base model"
    status = f"Generated with {adapter_status} | voice={Path(voice_prompt_path).stem} | frames={steps}"
    if not has_text_supervision:
        status += " | text stream not trained"
    return (SAMPLE_RATE, preview_audio), generated_text, status


def build_ui(default_adapter: str | None = None):
    checkpoint_choices = scan_checkpoints()
    default_label = BASE_LABEL
    if default_adapter:
        resolved = _resolve_adapter_file(default_adapter)
        if resolved:
            try:
                rel = Path(resolved).parent.relative_to(CHECKPOINTS_ROOT)
                default_label = " / ".join(rel.parts)
            except ValueError:
                default_label = BASE_LABEL

    initial_voice_choices = scan_voices(label_to_adapter_path(default_label))

    with gr.Blocks(title="PersonaPlex Hindi Checkpoint Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# PersonaPlex Hindi Checkpoint Demo

Load the base PersonaPlex model or any Hindi LoRA checkpoint and audition how the model responds to spoken Hindi input.
            """.strip()
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_audio = gr.Audio(label="Input speech", type="filepath", sources=["upload", "microphone"])
                adapter_input = gr.Dropdown(
                    label="Checkpoint",
                    choices=checkpoint_choices,
                    value=default_label,
                    allow_custom_value=False,
                )
                refresh_btn = gr.Button("Refresh checkpoints", variant="secondary", size="sm")
                voice_input = gr.Dropdown(
                    label="Voice prompt",
                    choices=initial_voice_choices,
                    value=initial_voice_choices[0] if initial_voice_choices else None,
                    allow_custom_value=True,
                )
                custom_voice_prompt = gr.Textbox(
                    label="Custom voice prompt path",
                    placeholder="Optional absolute path to a .pt voice prompt",
                )
                text_prompt = gr.Textbox(
                    label="System prompt",
                    lines=3,
                    value="You are a helpful assistant that speaks natural Hindi. Answer in Hindi.",
                )
                seed = gr.Slider(label="Seed", minimum=0, maximum=99999999, value=42424242, step=1)
                audio_temp = gr.Slider(label="Audio temperature", minimum=0.1, maximum=1.5, value=0.8, step=0.05)
                text_temp = gr.Slider(label="Text temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.05)
                audio_topk = gr.Slider(label="Audio top-k", minimum=10, maximum=500, value=250, step=5)
                text_topk = gr.Slider(label="Text top-k", minimum=5, maximum=100, value=25, step=1)
                generate_btn = gr.Button("Generate preview", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Generated output", type="numpy")
                output_text = gr.Textbox(label="Decoded generated text", lines=6)
                status = gr.Textbox(label="Status", lines=2)

        def _refresh_checkpoints():
            choices = scan_checkpoints()
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def _refresh_voices(adapter_label: str):
            adapter_path = label_to_adapter_path(adapter_label)
            voices = scan_voices(adapter_path)
            return gr.update(choices=voices, value=voices[0] if voices else None)

        refresh_btn.click(fn=_refresh_checkpoints, outputs=[adapter_input])
        adapter_input.change(fn=_refresh_voices, inputs=[adapter_input], outputs=[voice_input])
        generate_btn.click(
            fn=generate_preview,
            inputs=[
                input_audio,
                adapter_input,
                voice_input,
                custom_voice_prompt,
                text_prompt,
                seed,
                audio_temp,
                text_temp,
                audio_topk,
                text_topk,
            ],
            outputs=[output_audio, output_text, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to checkpoint dir or adapters.npz")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ui = build_ui(default_adapter=args.adapter)
    ui.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
