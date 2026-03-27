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

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import soundfile as sf

# ── Model IDs ────────────────────────────────────────────────────────────────
BASE_MODEL          = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
CUSTOM_VOICE_MODEL  = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

_models = {}   # cache: {"base": model, "custom": model}


# ── Model loading ─────────────────────────────────────────────────────────────

def get_model(mode: str, adapter_path: str = None):
    """Load and cache base or custom-voice model."""
    key = f"{mode}_{adapter_path}"
    if key in _models:
        return _models[key]

    from mlx_audio.tts.utils import load_model as mlx_load
    model_id = CUSTOM_VOICE_MODEL if mode == "custom" else BASE_MODEL
    print(f"[demo] Loading {mode} model: {model_id}")
    model = mlx_load(model_id)

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapters.safetensors")):
        from train.lora import load_adapters, apply_lora, LoRAConfig
        apply_lora(model, LoRAConfig(model_type="qwen3_tts"))
        load_adapters(model, os.path.join(adapter_path, "adapters.safetensors"))
        print(f"[demo] Loaded adapters from {adapter_path}")

    _models[key] = model
    return model


# ── Core synthesis ────────────────────────────────────────────────────────────

def synthesise(
    text:         str,
    ref_audio,          # numpy array from gr.Audio or None
    ref_text:     str,
    adapter_dir:  str,
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
    status     = "  |  ".join(status_log)

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
                adapter_input = gr.Textbox(
                    label       = "Finetuned adapter path (blank = base weights)",
                    placeholder = "./checkpoints/qwen3-hindi/checkpoint-best",
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
                ["नमस्ते! मेरा नाम राज है और मैं दिल्ली से हूँ।",             None, "", "", 1.0, 0.7],
                ["आज बाज़ार में बहुत भीड़ थी, लेकिन मैंने सब ख़रीद लिया।",    None, "", "", 1.0, 0.7],
                ["Hello, मैं ठीक हूँ। आप कैसे हैं? Thank you for asking.",    None, "", "", 1.0, 0.7],
                ["भारत एक विविधताओं से भरा देश है।",                          None, "", "", 0.9, 0.7],
                ["The quick brown fox jumps over the lazy dog.",              None, "", "", 1.0, 0.8],
            ],
            inputs=[text_input, ref_audio_input, ref_text_input, adapter_input, speed_slider, temp_slider],
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

**Adapter path:**
- Leave blank to use base weights
- Point to a finetuned checkpoint to compare quality before/after training
""")

        # ── Wire up ──────────────────────────────────────────────────────────
        gen_btn.click(
            fn      = synthesise,
            inputs  = [text_input, ref_audio_input, ref_text_input, adapter_input, speed_slider, temp_slider],
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
