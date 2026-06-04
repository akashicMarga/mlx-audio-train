#!/usr/bin/env python3
"""
lfm_asr_demo.py — LFM 2.5 Audio ASR demo (Flask, single-threaded).

MLX streams are thread-local on Metal; Gradio/threaded servers break them.
Flask with threaded=False runs everything on the main thread where MLX
initialised its GPU context.

Usage:
    pip install flask flask-cors
    python scripts/lfm_asr_demo.py
    python scripts/lfm_asr_demo.py --adapter checkpoints/lfm-audio-asr-test/checkpoint-best
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────────────

WEBUI_DIR        = Path(__file__).parent / "lfm_asr_webui"
CHECKPOINTS_ROOT = Path(__file__).parent.parent / "checkpoints"
DEFAULT_MODEL_ID = "mlx-community/LFM2.5-Audio-1.5B-8bit"
SYSTEM_PROMPT    = "Perform ASR."
BASE_LABEL       = "__base__"

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(WEBUI_DIR))
CORS(app)

M = {}   # model, processor

# ── Model loading ─────────────────────────────────────────────────────────────

def load_base(model_id: str):
    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor

    print(f"[demo] Loading base model: {model_id}")
    model     = LFM2AudioModel.from_pretrained(model_id)
    processor = LFM2AudioProcessor.from_pretrained(model_id)
    mx.eval(model.parameters())
    model.eval()
    M["base_model"]     = model
    M["processor"]      = processor
    M["model_id"]       = model_id
    M["lora_model"]     = None
    M["lora_label"]     = None
    print("[demo] Base model ready.")


def get_model(adapter_label: str):
    """Return (model, processor). Loads LoRA model on demand and caches it."""
    if adapter_label == BASE_LABEL or not adapter_label:
        return M["base_model"], M["processor"]

    # Already cached for this label
    if M.get("lora_label") == adapter_label and M.get("lora_model") is not None:
        return M["lora_model"], M["processor"]

    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio import LFM2AudioModel
    from train.lora import apply_lora, load_adapters, LoRAConfig

    adapter_path = label_to_path(adapter_label)
    print(f"[demo] Applying adapter: {adapter_path}")
    lora_model = LFM2AudioModel.from_pretrained(M["model_id"])
    mx.eval(lora_model.parameters())

    cfg = LoRAConfig(model_type="lfm_audio", rank=8, alpha=16.0)
    apply_lora(lora_model, cfg)
    load_adapters(lora_model, str(Path(adapter_path) / "adapters.safetensors"))
    mx.eval(lora_model.parameters())
    lora_model.eval()

    M["lora_model"] = lora_model
    M["lora_label"] = adapter_label
    return lora_model, M["processor"]


# ── Adapter listing ───────────────────────────────────────────────────────────

def scan_adapters():
    entries = [{"label": "Base model (no adapter)", "value": BASE_LABEL}]
    if CHECKPOINTS_ROOT.exists():
        for f in sorted(CHECKPOINTS_ROOT.rglob("adapters.safetensors")):
            rel   = f.parent.relative_to(CHECKPOINTS_ROOT)
            label = str(rel).replace("/", " / ").replace("\\", " / ")
            entries.append({"label": label, "value": str(rel)})
    return entries


def label_to_path(value: str) -> str:
    return str(CHECKPOINTS_ROOT / value)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_asr(audio_np: np.ndarray, sr: int, adapter_label: str,
            max_new_tokens: int = 64, temperature: float = 0.0, top_k: int = 1) -> dict:
    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio.processor import ChatState
    from mlx_audio.sts.models.lfm_audio.model import LFMModality, IM_END_TOKEN

    # Mono float32 [-1, 1]
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if audio_np.max() > 1.0:
        audio_np /= 32768.0

    model, processor = get_model(adapter_label)

    audio_mx   = mx.array(audio_np)
    chat_state = ChatState(processor)
    chat_state.new_turn("system");    chat_state.add_text(SYSTEM_PROMPT); chat_state.end_turn()
    chat_state.new_turn("user");      chat_state.add_audio(audio_mx, sample_rate=sr); chat_state.end_turn()
    chat_state.new_turn("assistant")

    t0   = time.time()
    toks = []
    for token, modality in model.generate_from_chat_state(
        chat_state, mode="sequential",
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k,
    ):
        if modality == LFMModality.TEXT:
            tid = int(token.item())
            if tid == IM_END_TOKEN:
                break
            toks.append(tid)

    text    = processor.decode_text(toks).strip()
    elapsed = time.time() - t0
    dur     = len(audio_np) / max(sr, 1)
    return {"text": text, "elapsed": round(elapsed, 2), "rtf": round(elapsed / max(dur, 0.01), 2)}


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(WEBUI_DIR), "index.html")


@app.route("/adapters")
def list_adapters():
    return jsonify(scan_adapters())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data         = request.json
    audio_b64    = data.get("audio_b64", "")
    sr           = int(data.get("sample_rate", 16000))
    adapter      = data.get("adapter", BASE_LABEL)
    max_tokens   = int(data.get("max_tokens", 200))
    temperature  = float(data.get("temperature", 0.3))
    top_k        = int(data.get("top_k", 50))

    # Decode raw PCM float32 bytes (sent as base64)
    raw     = base64.b64decode(audio_b64)
    audio   = np.frombuffer(raw, dtype=np.float32).copy()

    try:
        result = run_asr(audio, sr, adapter, max_tokens, temperature, top_k)
        return jsonify({"ok": True, **result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    data        = request.json
    audio_b64   = data.get("audio_b64", "")
    sr          = int(data.get("sample_rate", 16000))
    adapter     = data.get("adapter", BASE_LABEL)
    max_tokens  = int(data.get("max_tokens", 200))
    temperature = float(data.get("temperature", 0.3))
    top_k       = int(data.get("top_k", 50))

    raw   = base64.b64decode(audio_b64)
    audio = np.frombuffer(raw, dtype=np.float32).copy()

    try:
        base_res = run_asr(audio, sr, BASE_LABEL, max_tokens, temperature, top_k)
        lora_res = run_asr(audio, sr, adapter,     max_tokens, temperature, top_k)
        return jsonify({"ok": True, "base": base_res, "lora": lora_res})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LFM 2.5 Audio ASR demo")
    p.add_argument("--model",   default=DEFAULT_MODEL_ID)
    p.add_argument("--adapter", default=None, help="Default adapter dir (optional)")
    p.add_argument("--port",    type=int, default=7860)
    args = p.parse_args()

    WEBUI_DIR.mkdir(exist_ok=True)
    _write_html(args.adapter)

    load_base(args.model)
    # Pre-load default adapter if specified
    if args.adapter:
        ap = Path(args.adapter).expanduser().resolve()
        if not ap.is_absolute():
            ap = Path.cwd() / ap
        try:
            rel = str(ap.relative_to(CHECKPOINTS_ROOT.resolve()))
        except ValueError:
            rel = str(ap)   # already an absolute path outside CHECKPOINTS_ROOT
        get_model(rel)

    print(f"\n  Open: http://localhost:{args.port}\n")
    # threaded=False: keeps all inference on the main thread where MLX
    # initialized its Metal GPU stream. Worker threads don't have that stream.
    app.run(host="0.0.0.0", port=args.port, threaded=False, use_reloader=False)


# ── HTML generation ───────────────────────────────────────────────────────────

def _write_html(default_adapter):
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LFM 2.5 Audio — ASR Demo</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e2e8f0; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem;
  }
  header { text-align: center; margin-bottom: 2rem; }
  header h1 { font-size: 1.8rem; font-weight: 700; color: #fff; }
  header p  { color: #94a3b8; margin-top: .4rem; font-size: .95rem; }
  .card {
    background: #1e2433; border: 1px solid #2d3748; border-radius: 12px;
    padding: 1.5rem; width: 100%; max-width: 780px; margin-bottom: 1.2rem;
  }
  .card h2 { font-size: 1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem; }
  label { display: block; font-size: .85rem; color: #94a3b8; margin-bottom: .3rem; }
  select, input[type=range] {
    width: 100%; background: #131720; border: 1px solid #374151;
    border-radius: 6px; color: #e2e8f0; padding: .5rem .75rem; font-size: .9rem;
  }
  select:focus { outline: none; border-color: #6366f1; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  .col { flex: 1; min-width: 160px; }
  .range-row { display: flex; align-items: center; gap: .5rem; }
  .range-row input { flex: 1; }
  .range-val { font-size: .85rem; color: #a5b4fc; width: 2.5rem; text-align: right; }
  /* Audio controls */
  .audio-zone {
    border: 2px dashed #374151; border-radius: 10px; padding: 1.2rem;
    text-align: center; cursor: pointer; transition: border-color .2s;
  }
  .audio-zone:hover { border-color: #6366f1; }
  .audio-zone p { color: #64748b; font-size: .9rem; }
  #waveform { width: 100%; height: 60px; background: #131720; border-radius: 6px;
               display: none; margin-top: .75rem; }
  .btn-row { display: flex; gap: .75rem; margin-top: .75rem; flex-wrap: wrap; }
  button {
    border: none; border-radius: 8px; padding: .55rem 1.1rem;
    font-size: .9rem; font-weight: 600; cursor: pointer; transition: opacity .15s;
  }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .btn-primary  { background: #6366f1; color: #fff; }
  .btn-danger   { background: #ef4444; color: #fff; }
  .btn-secondary{ background: #334155; color: #e2e8f0; }
  .btn-compare  { background: #0891b2; color: #fff; }
  button:not(:disabled):hover { opacity: .85; }
  /* Transcript */
  .transcript {
    background: #131720; border: 1px solid #1e2d3d; border-radius: 8px;
    padding: 1rem; min-height: 80px; font-size: 1rem; line-height: 1.6;
    color: #e2e8f0; white-space: pre-wrap; word-break: break-word;
    position: relative;
  }
  .transcript .placeholder { color: #4b5563; font-style: italic; }
  .meta { font-size: .78rem; color: #475569; margin-top: .5rem; }
  /* Comparison */
  .cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media(max-width:600px){ .cmp-grid { grid-template-columns: 1fr; } }
  .cmp-label { font-size: .8rem; font-weight: 600; color: #6366f1; margin-bottom: .4rem; }
  .cmp-label.lora { color: #10b981; }
  /* Spinner */
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #6366f1; border-top-color: transparent;
    border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status { font-size: .85rem; color: #64748b; margin-top: .5rem; min-height: 1.2em; }
  .err { color: #f87171; }
  audio { width: 100%; margin-top: .75rem; display: none; border-radius: 6px; }
  details > summary { cursor: pointer; font-size: .85rem; color: #64748b;
                       padding: .4rem 0; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; font-size: .7rem; }
  details[open] > summary::before { content: "▼ "; }
  details .inner { padding-top: .75rem; }
</style>
</head>
<body>

<header>
  <h1>🎙️ LFM 2.5 Audio — ASR</h1>
  <p>Transcribe speech with the base model or a fine-tuned LoRA adapter</p>
</header>

<!-- Audio input card -->
<div class="card">
  <h2>Audio input</h2>
  <div class="audio-zone" id="dropZone">
    <p>Drop a WAV / MP3 here, or click to browse</p>
    <input type="file" id="fileInput" accept="audio/*" style="display:none">
  </div>
  <audio id="audioPreview" controls></audio>
  <canvas id="waveform"></canvas>

  <div class="btn-row">
    <button class="btn-primary"    id="recBtn">⏺ Record</button>
    <button class="btn-danger"     id="stopBtn"    disabled>⏹ Stop</button>
    <button class="btn-secondary"  id="clearBtn">✕ Clear</button>
  </div>
  <div class="status" id="recStatus"></div>
</div>

<!-- Settings -->
<div class="card">
  <h2>Settings</h2>
  <div class="row" style="margin-bottom:.8rem">
    <div class="col">
      <label>LoRA adapter</label>
      <select id="adapterSel"></select>
    </div>
  </div>

  <details>
    <summary>Generation parameters</summary>
    <div class="inner">
      <div class="row">
        <div class="col">
          <label>Max tokens</label>
          <div class="range-row">
            <input type="range" id="maxTokens" min="20" max="200" value="64" step="4">
            <span class="range-val" id="maxTokensVal">64</span>
          </div>
        </div>
        <div class="col">
          <label>Temperature</label>
          <div class="range-row">
            <input type="range" id="temperature" min="0" max="1.5" value="0" step="0.05">
            <span class="range-val" id="temperatureVal">0.00</span>
          </div>
        </div>
        <div class="col">
          <label>Top-k</label>
          <div class="range-row">
            <input type="range" id="topK" min="1" max="100" value="50" step="1">
            <span class="range-val" id="topKVal">50</span>
          </div>
        </div>
      </div>
    </div>
  </details>
</div>

<!-- Transcription -->
<div class="card">
  <h2>Transcription</h2>
  <div class="transcript" id="transcript">
    <span class="placeholder">Transcription will appear here…</span>
  </div>
  <div class="meta" id="transcrMeta"></div>
  <div class="btn-row" style="margin-top:.75rem">
    <button class="btn-primary"  id="transcribeBtn" disabled>▶ Transcribe</button>
    <button class="btn-compare"  id="compareBtn"    disabled>⇄ Compare base vs adapter</button>
  </div>
  <div class="status" id="transcribeStatus"></div>
</div>

<!-- Comparison result -->
<div class="card" id="compareCard" style="display:none">
  <h2>Side-by-side comparison</h2>
  <div class="cmp-grid">
    <div>
      <div class="cmp-label">Base model</div>
      <div class="transcript" id="cmpBase"></div>
      <div class="meta" id="cmpBaseMeta"></div>
    </div>
    <div>
      <div class="cmp-label lora">Fine-tuned adapter</div>
      <div class="transcript" id="cmpLora"></div>
      <div class="meta" id="cmpLoraMeta"></div>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
let audioBuffer = null;   // Float32Array, 16 kHz mono
let mediaRecorder = null;
let recChunks = [];
const TARGET_SR = 16000;

// ── Adapter list ─────────────────────────────────────────────────────────────
fetch("/adapters").then(r => r.json()).then(list => {
  const sel = document.getElementById("adapterSel");
  sel.innerHTML = "";
  list.forEach(({label, value}) => {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  });
  // Pre-select first non-base if available
  if (list.length > 1) sel.selectedIndex = 1;
});

// ── Range display ─────────────────────────────────────────────────────────────
function linkRange(id, valId, fmt) {
  const el = document.getElementById(id);
  const vl = document.getElementById(valId);
  vl.textContent = fmt(el.value);
  el.addEventListener("input", () => vl.textContent = fmt(el.value));
}
linkRange("maxTokens",   "maxTokensVal",   v => v);
linkRange("temperature", "temperatureVal", v => parseFloat(v).toFixed(2));
linkRange("topK",        "topKVal",        v => v);

// ── File drop / browse ────────────────────────────────────────────────────────
const dropZone  = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.style.borderColor = "#6366f1"; });
dropZone.addEventListener("dragleave", ()  => dropZone.style.borderColor = "");
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.style.borderColor = "";
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  const url = URL.createObjectURL(file);
  const preview = document.getElementById("audioPreview");
  preview.src = url; preview.style.display = "block";
  setStatus("recStatus", "");

  const reader = new FileReader();
  reader.onload = e => {
    const ctx = new AudioContext({ sampleRate: TARGET_SR });
    ctx.decodeAudioData(e.target.result).then(buf => {
      // Mix down to mono at TARGET_SR (AudioContext already resampled)
      const ch0 = buf.getChannelData(0);
      if (buf.numberOfChannels === 1) {
        audioBuffer = new Float32Array(ch0);
      } else {
        const ch1 = buf.getChannelData(1);
        audioBuffer = new Float32Array(ch0.length);
        for (let i = 0; i < ch0.length; i++) audioBuffer[i] = (ch0[i] + ch1[i]) * 0.5;
      }
      drawWaveform(audioBuffer);
      enableButtons();
      setStatus("recStatus", `Loaded: ${(audioBuffer.length / TARGET_SR).toFixed(1)}s`);
    });
  };
  reader.readAsArrayBuffer(file);
}

// ── Recording ─────────────────────────────────────────────────────────────────
document.getElementById("recBtn").addEventListener("click", startRec);
document.getElementById("stopBtn").addEventListener("click", stopRec);
document.getElementById("clearBtn").addEventListener("click", clearAudio);

async function startRec() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: TARGET_SR } });
  recChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => recChunks.push(e.data);
  mediaRecorder.onstop = () => {
    const blob = new Blob(recChunks, { type: "audio/webm" });
    loadFile(blob);
    stream.getTracks().forEach(t => t.stop());
  };
  mediaRecorder.start();
  document.getElementById("recBtn").disabled  = true;
  document.getElementById("stopBtn").disabled = false;
  setStatus("recStatus", "⏺ Recording…");
}

function stopRec() {
  if (mediaRecorder) mediaRecorder.stop();
  document.getElementById("recBtn").disabled  = false;
  document.getElementById("stopBtn").disabled = true;
}

function clearAudio() {
  audioBuffer = null;
  document.getElementById("audioPreview").style.display = "none";
  document.getElementById("audioPreview").src = "";
  document.getElementById("waveform").style.display = "none";
  disableButtons();
  document.getElementById("transcript").innerHTML = '<span class="placeholder">Transcription will appear here…</span>';
  document.getElementById("transcrMeta").textContent = "";
  document.getElementById("compareCard").style.display = "none";
  setStatus("recStatus", "");
}

// ── Waveform ──────────────────────────────────────────────────────────────────
function drawWaveform(samples) {
  const canvas = document.getElementById("waveform");
  canvas.style.display = "block";
  const ctx = canvas.getContext("2d");
  canvas.width  = canvas.offsetWidth;
  canvas.height = 60;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#131720";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 1;
  const step = Math.ceil(samples.length / canvas.width);
  const mid  = canvas.height / 2;
  ctx.beginPath();
  for (let x = 0; x < canvas.width; x++) {
    let max = 0;
    for (let j = 0; j < step; j++) { const v = Math.abs(samples[x * step + j] || 0); if (v > max) max = v; }
    ctx.moveTo(x, mid - max * mid);
    ctx.lineTo(x, mid + max * mid);
  }
  ctx.stroke();
}

// ── Buttons ───────────────────────────────────────────────────────────────────
function enableButtons() {
  document.getElementById("transcribeBtn").disabled = false;
  document.getElementById("compareBtn").disabled    = false;
}
function disableButtons() {
  document.getElementById("transcribeBtn").disabled = true;
  document.getElementById("compareBtn").disabled    = true;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(id, msg, isErr) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className   = "status" + (isErr ? " err" : "");
}

function audioToBase64(buf) {
  // Float32Array → base64
  const bytes = new Uint8Array(buf.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function params() {
  return {
    audio_b64:   audioToBase64(audioBuffer),
    sample_rate: TARGET_SR,
    adapter:     document.getElementById("adapterSel").value,
    max_tokens:  parseInt(document.getElementById("maxTokens").value),
    temperature: parseFloat(document.getElementById("temperature").value),
    top_k:       1,
  };
}

function metaStr(res) {
  return `${res.elapsed}s elapsed · RTF ${res.rtf}×`;
}

// ── Transcribe ────────────────────────────────────────────────────────────────
document.getElementById("transcribeBtn").addEventListener("click", async () => {
  if (!audioBuffer) return;
  const btn = document.getElementById("transcribeBtn");
  btn.disabled = true;
  document.getElementById("transcript").innerHTML = '<span class="spinner"></span> Transcribing…';
  document.getElementById("transcrMeta").textContent = "";
  setStatus("transcribeStatus", "Running inference…");

  try {
    const res = await fetch("/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params()),
    }).then(r => r.json());

    if (res.ok) {
      document.getElementById("transcript").textContent = res.text || "(empty)";
      document.getElementById("transcrMeta").textContent = metaStr(res);
      setStatus("transcribeStatus", "Done.");
    } else {
      document.getElementById("transcript").innerHTML = `<span class="placeholder">(error)</span>`;
      setStatus("transcribeStatus", res.error, true);
    }
  } catch(e) {
    setStatus("transcribeStatus", String(e), true);
  } finally {
    btn.disabled = false;
  }
});

// ── Compare ───────────────────────────────────────────────────────────────────
document.getElementById("compareBtn").addEventListener("click", async () => {
  if (!audioBuffer) return;
  const btn = document.getElementById("compareBtn");
  btn.disabled = true;
  document.getElementById("compareCard").style.display = "block";
  document.getElementById("cmpBase").innerHTML = '<span class="spinner"></span>';
  document.getElementById("cmpLora").innerHTML = '<span class="spinner"></span>';
  document.getElementById("cmpBaseMeta").textContent = "";
  document.getElementById("cmpLoraMeta").textContent = "";
  setStatus("transcribeStatus", "Running comparison (2× inference)…");

  try {
    const res = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params()),
    }).then(r => r.json());

    if (res.ok) {
      document.getElementById("cmpBase").textContent = res.base.text || "(empty)";
      document.getElementById("cmpLora").textContent = res.lora.text || "(empty)";
      document.getElementById("cmpBaseMeta").textContent = metaStr(res.base);
      document.getElementById("cmpLoraMeta").textContent = metaStr(res.lora);
      setStatus("transcribeStatus", "Comparison done.");
    } else {
      document.getElementById("cmpBase").textContent = "";
      document.getElementById("cmpLora").textContent = "";
      setStatus("transcribeStatus", res.error, true);
    }
  } catch(e) {
    setStatus("transcribeStatus", String(e), true);
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>"""
    (WEBUI_DIR / "index.html").write_text(html)


if __name__ == "__main__":
    main()
