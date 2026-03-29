#!/usr/bin/env python3
"""
demo.py — Qwen3-TTS Gradio demo with voice cloning.

Modes:
  1. Base (no ref audio)     → random voice every time
  2. Voice cloning (ref audio provided) → clones the uploaded speaker

Launch:
    python scripts/demo.py
    python scripts/demo.py --adapter checkpoints/qwen3-hindi/checkpoint-best
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import soundfile as sf

# ── Model IDs ────────────────────────────────────────────────────────────────
BASE_MODEL          = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
CUSTOM_VOICE_MODEL  = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

_models = {}   # cache: {"base": model, "custom": model}

CHECKPOINTS_ROOT = Path(__file__).parent.parent / "checkpoints"
BASE_LABEL       = "── Base model (no adapter) ──"


def scan_checkpoints() -> list:
    """Return sorted list of (label, path) for all available checkpoints."""
    choices = [BASE_LABEL]
    if CHECKPOINTS_ROOT.exists():
        for adapter_file in sorted(CHECKPOINTS_ROOT.rglob("adapters.safetensors")):
            # e.g.  checkpoints/qwen3-hindi/checkpoint-best  →  "qwen3-hindi / checkpoint-best"
            rel = adapter_file.parent.relative_to(CHECKPOINTS_ROOT)
            label = str(rel).replace("/", " / ").replace("\\", " / ")
            choices.append(label)
    return choices


def label_to_adapter_path(label: str) -> Optional[str]:
    """Convert dropdown label back to the checkpoint directory path."""
    if label == BASE_LABEL or not label:
        return None
    rel = label.replace(" / ", os.sep)
    return str(CHECKPOINTS_ROOT / rel)


# ── Model loading ─────────────────────────────────────────────────────────────

def _resolve_adapter_file(adapter_path: str) -> Optional[str]:
    """
    Resolve adapter_path to the actual adapters.safetensors file path.

    Accepts any of:
      - /path/to/checkpoint-best/                (dir containing adapters.safetensors)
      - /path/to/checkpoint-best/adapters.safetensors  (file directly)
      - /path/to/checkpoints/qwen3-hindi/        (parent dir → picks latest checkpoint)
    Returns the resolved file path, or None with a printed warning.
    """
    p = Path(adapter_path.strip())

    # Direct file
    if p.is_file() and p.suffix == ".safetensors":
        return str(p)

    # Directory containing adapters.safetensors
    if p.is_dir():
        direct = p / "adapters.safetensors"
        if direct.exists():
            return str(direct)

        # Parent dir: look for checkpoint-* subdirs and pick the latest
        checkpoints = sorted(p.glob("checkpoint-*/adapters.safetensors"))
        if checkpoints:
            chosen = checkpoints[-1]  # last = lexicographically latest
            print(f"[demo] Found {len(checkpoints)} checkpoint(s), using: {chosen.parent.name}")
            return str(chosen)

    print(f"[demo] WARNING: no adapters.safetensors found at '{adapter_path}'")
    print(f"[demo]   Expected one of:")
    print(f"[demo]     {adapter_path}/adapters.safetensors")
    print(f"[demo]     {adapter_path}/checkpoint-*/adapters.safetensors")
    return None


def _load_model_config(adapter_path: str) -> dict:
    """Read model_config.yaml saved alongside the checkpoint. Returns {} if not found."""
    import yaml
    search_dirs = [
        Path(adapter_path),
        Path(adapter_path).parent,
        Path(adapter_path).parent.parent,
    ]
    for d in search_dirs:
        cfg_file = d / "model_config.yaml"
        if cfg_file.exists():
            with open(cfg_file) as f:
                return yaml.safe_load(f)
    return {}


def get_model(mode: str, adapter_path: str = None):
    """Load and cache base or custom-voice model."""
    key = f"{mode}_{adapter_path}"
    if key in _models:
        return _models[key]

    from mlx_audio.tts.utils import load_model as mlx_load
    model_id = CUSTOM_VOICE_MODEL if mode == "custom" else BASE_MODEL
    print(f"[demo] Loading {mode} model: {model_id}")
    model = mlx_load(model_id)

    if adapter_path:
        adapter_file = _resolve_adapter_file(adapter_path)
        if adapter_file:
            from train.lora import load_adapters, apply_lora, LoRAConfig
            import mlx.core as _mx
            # Infer rank from the saved adapter shapes so rank-16 (pipe2)
            # and rank-8 (pipe1) adapters both load correctly.
            _weights = _mx.load(adapter_file)
            _lora_a  = next((v for k, v in _weights.items() if k.endswith("lora_a")), None)
            _rank    = int(_lora_a.shape[1]) if _lora_a is not None else 8
            # Detect model_type from saved config (qwen3_tts vs qwen3_tts_speaker)
            _saved_cfg  = _load_model_config(adapter_path)
            _model_type = _saved_cfg.get("model", {}).get("model_type", "qwen3_tts")
            apply_lora(model, LoRAConfig(model_type=_model_type, rank=_rank))
            load_adapters(model, adapter_file)
            print(f"[demo] ✅ Loaded adapters (model_type={_model_type}, rank={_rank}): {adapter_file}")
            # Patch custom language token IDs (e.g. hi→2051) so lang_code works at inference
            custom_lang_ids = _saved_cfg.get("model", {}).get("custom_lang_ids", {})
            if custom_lang_ids:
                model.talker.config.codec_language_id.update(custom_lang_ids)
                print(f"[demo] Registered custom lang IDs: {custom_lang_ids}")
        # else: warning already printed inside _resolve_adapter_file

    _models[key] = model
    return model


# ── Core synthesis ────────────────────────────────────────────────────────────

def synthesise(
    text:         str,
    ref_audio,          # numpy array from gr.Audio or None
    ref_text:     str,
    adapter_dir:  str,
    lang_code:    str,
    speed:        float,
    temperature:  float,
):
    if not text.strip():
        return None, None, "⚠️ Please enter some text."

    from mlx_audio.tts.generate import generate_audio

    results    = {}   # {"base": (sr, audio), "cloned": (sr, audio)}
    status_log = []

    # ── Determine which modes to run ─────────────────────────────────────────
    run_base   = True
    run_cloned = ref_audio is not None

    adapter = adapter_dir.strip() or None

    # Show adapter status in the UI
    if adapter:
        resolved = _resolve_adapter_file(adapter)
        if resolved:
            adapter_label = f"adapter: {Path(resolved).parent.name}"
        else:
            status_log.append(f"⚠️ Adapter path set but adapters.safetensors not found: {adapter}")
            adapter = None
            adapter_label = "base weights (adapter not found)"
    else:
        adapter_label = "base weights"

    # ── Save ref audio to temp file if provided ───────────────────────────────
    ref_path = None
    if run_cloned:
        sr_ref, audio_ref = ref_audio
        tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_ref.name, audio_ref, sr_ref)
        ref_path = tmp_ref.name

    # ── Base generation (no speaker) ─────────────────────────────────────────
    if run_base:
        try:
            model    = get_model("base", adapter)
            out_dir  = tempfile.mkdtemp()
            generate_audio(
                text        = text,
                model       = model,
                output_path = out_dir,
                file_prefix = "base",
                speed       = speed,
                temperature = temperature,
                voice       = None,
                lang_code   = lang_code,
                verbose     = False,
                stt_model   = None,
            )
            matches = sorted(glob.glob(os.path.join(out_dir, "base*.wav")))
            if matches:
                audio, sr = sf.read(matches[0])
                results["base"] = (sr, audio.astype(np.float32))
                status_log.append("✅ Base: generated")
            else:
                status_log.append("❌ Base: no file saved")
        except Exception as e:
            status_log.append(f"❌ Base: {e}")

    # ── Cloned generation (ICL mode — Base model + ref_audio + ref_text) ────────
    # Real voice cloning uses the Base model in ICL (In-Context Learning) mode.
    # The CustomVoice model only supports its built-in speaker list, not arbitrary cloning.
    # ICL mode: Base model prefills with [ref_audio codec tokens | ref_text] then
    # generates the target text in the same voice.
    if run_cloned:
        try:
            model   = get_model("base", adapter)
            out_dir = tempfile.mkdtemp()
            kwargs  = dict(
                text        = text,
                model       = model,
                output_path = out_dir,
                file_prefix = "cloned",
                speed       = speed,
                temperature = temperature,
                ref_audio   = ref_path,
                voice       = None,
                lang_code   = lang_code,
                verbose     = False,
            )
            if ref_text.strip():
                # User provided transcript — use it directly, skip Whisper
                kwargs["ref_text"]  = ref_text.strip()
                kwargs["stt_model"] = None
            # else: ref_text omitted → generate_audio auto-transcribes via Whisper

            generate_audio(**kwargs)
            matches = sorted(glob.glob(os.path.join(out_dir, "cloned*.wav")))
            if matches:
                audio, sr = sf.read(matches[0])
                results["cloned"] = (sr, audio.astype(np.float32))
                status_log.append("✅ Cloned: generated")
            else:
                status_log.append("❌ Cloned: no file saved")
        except Exception as e:
            status_log.append(f"❌ Cloned: {e}")

    base_out   = results.get("base")
    cloned_out = results.get("cloned")
    status     = f"[{adapter_label}]  " + "  |  ".join(status_log)

    return base_out, cloned_out, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(adapter_path: str = None):
    with gr.Blocks(title="Qwen3-TTS — Voice Cloning Demo", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
# 🎙️ Qwen3-TTS — Voice Cloning Demo
**Left:** Base model (random voice) &nbsp;|&nbsp; **Right:** Your voice cloned from reference audio

Upload any 3–10 second audio clip as reference → the model speaks in that voice.
""")

        # ── Inputs ───────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label    = "Text to synthesise",
                    placeholder = "नमस्ते! आज का दिन बहुत अच्छा है।",
                    lines    = 3,
                )
            with gr.Column(scale=1):
                speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                temp_slider  = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature (variety)")

        with gr.Row():
            with gr.Column():
                ref_audio_input = gr.Audio(
                    label   = "🎤 Reference audio for voice cloning (3–10 sec)",
                    type    = "numpy",
                    sources = ["upload", "microphone"],
                )
                ref_text_input = gr.Textbox(
                    label       = "Reference text — what is said in the clip (optional, leave blank to auto-transcribe)",
                    placeholder = "e.g. Hello, I want to test voice cloning with this recording.",
                    lines       = 2,
                )
                with gr.Row():
                    adapter_input = gr.Dropdown(
                        label   = "Finetuned checkpoint",
                        choices = scan_checkpoints(),
                        value   = BASE_LABEL,
                        scale   = 4,
                    )
                    refresh_btn = gr.Button("🔄", size="sm", scale=1, min_width=40)
                lang_input = gr.Dropdown(
                    label   = "Language",
                    choices = ["auto", "hi", "kn", "mr", "ta", "te", "en"],
                    value   = "auto",
                    info    = "auto = base English. Select a language when using a multilingual adapter.",
                )

        gen_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

        # ── Outputs ──────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎲 Base (random voice)")
                base_out   = gr.Audio(label="Base output", type="numpy")
            with gr.Column():
                gr.Markdown("### 🎯 Cloned (your reference voice)")
                cloned_out = gr.Audio(label="Cloned output", type="numpy")

        status_box = gr.Textbox(label="Status", interactive=False)

        # ── Examples ─────────────────────────────────────────────────────────
        gr.Markdown("### 📝 Example texts")
        gr.Examples(
            examples=[
                ["नमस्ते! मेरा नाम राज है और मैं दिल्ली से हूँ।",                    None, "", BASE_LABEL, "hi", 1.0, 0.7],
                ["ಕನ್ನಡ ನಾಡಿನ ಸಂಸ್ಕೃತಿ ಬಹಳ ಶ್ರೀಮಂತವಾಗಿದೆ.",                          None, "", BASE_LABEL, "kn", 1.0, 0.7],
                ["मराठी भाषा महाराष्ट्राची अभिमानाची भाषा आहे.",                      None, "", BASE_LABEL, "mr", 1.0, 0.7],
                ["தமிழ் மொழி உலகின் தொன்மையான மொழிகளில் ஒன்றாகும்.",               None, "", BASE_LABEL, "ta", 1.0, 0.7],
                ["తెలుగు భాష చాలా మధురమైనది.",                                        None, "", BASE_LABEL, "te", 1.0, 0.7],
                ["The quick brown fox jumps over the lazy dog.",                      None, "", BASE_LABEL, "auto", 1.0, 0.8],
            ],
            inputs=[text_input, ref_audio_input, ref_text_input, adapter_input, lang_input, speed_slider, temp_slider],
        )

        # ── How to use ───────────────────────────────────────────────────────
        with gr.Accordion("ℹ️ How voice cloning works", open=False):
            gr.Markdown("""
**Without reference audio:**
- Uses the Base model → generates a random voice each time
- Useful for seeing raw Hindi quality

**With reference audio (3–10 seconds):**
- Uses the Base model in **ICL (In-Context Learning)** mode
- Prefills the LM with your reference audio codec tokens + reference text transcript
- Then generates the target text continuing in the same voice
- Works with ANY voice: record yourself or upload any clip
- **Reference text is required** — type exactly what is said in your clip

**Tips for better cloning:**
- Clear audio, no background noise
- 5–10 seconds works better than 3 seconds
- Reference text must match what is actually said in the clip (used as codec prefix)
- After Hindi finetuning, the cloned voice should speak Hindi more naturally

**Checkpoint dropdown:**
- Select a finetuned checkpoint from the dropdown to compare before/after training
- Click 🔄 to rescan if you just finished training
""")

        # ── Refresh dropdown ─────────────────────────────────────────────────
        refresh_btn.click(
            fn      = lambda: gr.update(choices=scan_checkpoints()),
            inputs  = [],
            outputs = [adapter_input],
        )

        # ── Wire up ──────────────────────────────────────────────────────────
        def synthesise_ui(text, ref_audio, ref_text, adapter_label, lang_code, speed, temperature):
            adapter_path = label_to_adapter_path(adapter_label)
            return synthesise(text, ref_audio, ref_text, adapter_path or "", lang_code, speed, temperature)

        gen_btn.click(
            fn      = synthesise_ui,
            inputs  = [text_input, ref_audio_input, ref_text_input, adapter_input, lang_input, speed_slider, temp_slider],
            outputs = [base_out, cloned_out, status_box],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to finetuned checkpoint dir")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true")
    args = parser.parse_args()

    ui = build_ui(adapter_path=args.adapter)
    ui.launch(server_port=args.port, share=args.share)
