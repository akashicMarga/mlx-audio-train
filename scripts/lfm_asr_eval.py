"""
Evaluate a fine-tuned LFM 2.5 Audio checkpoint on OHF-Voice function-calling.

Matches the cookbook's three-metric evaluation (format compliance, function-name
accuracy, argument accuracy) using our MLX model instead of the GGUF server.

Usage:
    python scripts/lfm_asr_eval.py \
        --adapter checkpoints/lfm-audio-asr-10k/checkpoint-best \
        --samples-per-function 10 \
        --output evals/
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


MODEL_ID   = "mlx-community/LFM2.5-Audio-1.5B-8bit"
DATASET    = "Paulescu/OHF-Voice-audio-20260504"
SYS_PROMPT = "Perform ASR."


# ── Parser (mirrors cookbook's _parser.py) ──────────────────────────────────

def parse_call(text: str):
    text = text.strip()
    if not text:
        return None
    parts = text.split("|")
    fn = parts[0].strip()
    if not fn or not fn.replace("_", "").isalnum():
        return None
    args: dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            return None
        k, v = p.split("=", 1)
        k = k.strip()
        if not k.startswith("$") or not k[1:].replace("_", "").isalnum():
            return None
        args[k] = v.strip()
    return fn, args


def score(pred: str, gt: str):
    gt_parsed = parse_call(gt)
    if gt_parsed is None:
        raise ValueError(f"unparseable ground truth: {gt!r}")
    pred_parsed = parse_call(pred)
    if pred_parsed is None:
        return False, False, False
    fn_match   = pred_parsed[0] == gt_parsed[0]
    args_match = fn_match and pred_parsed[1] == gt_parsed[1]
    return True, fn_match, args_match


# ── Audio helpers ────────────────────────────────────────────────────────────

def audio_field_to_numpy(audio_field) -> tuple[np.ndarray, int]:
    if isinstance(audio_field, (bytes, bytearray)):
        buf = io.BytesIO(bytes(audio_field))
        arr, sr = sf.read(buf, dtype="float32")
        return arr, sr
    if isinstance(audio_field, dict):
        if audio_field.get("bytes") is not None:
            buf = io.BytesIO(bytes(audio_field["bytes"]))
            arr, sr = sf.read(buf, dtype="float32")
            return arr, sr
        if "array" in audio_field and "sampling_rate" in audio_field:
            return np.array(audio_field["array"], dtype=np.float32), audio_field["sampling_rate"]
    raise TypeError(f"Unrecognised audio field: {type(audio_field).__name__}")


# ── Inference ────────────────────────────────────────────────────────────────

def load_model(adapter_path: str | None):
    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train.lora import apply_lora, load_adapters, LoRAConfig

    print(f"[eval] Loading base model: {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    model     = LFM2AudioModel.from_pretrained(MODEL_ID)
    model.eval()

    if adapter_path:
        print(f"[eval] Applying adapter: {adapter_path}")
        apply_lora(model, LoRAConfig(model_type="lfm_audio", rank=16, alpha=32.0))
        safetensors = Path(adapter_path) / "adapters.safetensors"
        load_adapters(model, str(safetensors))

    mx.eval(model.parameters())
    return model, processor


def transcribe(model, processor, audio: np.ndarray, sr: int,
               max_new_tokens: int = 64) -> str:
    import mlx.core as mx
    from mlx_audio.sts.models.lfm_audio.processor import ChatState
    from mlx_audio.sts.models.lfm_audio.model import LFMModality

    audio_mx = mx.array(audio)
    chat = ChatState(processor)
    chat.new_turn("system"); chat.add_text(SYS_PROMPT); chat.end_turn()
    chat.new_turn("user");   chat.add_audio(audio_mx, sample_rate=sr); chat.end_turn()
    chat.new_turn("assistant")

    IM_END_ID = 7  # <|im_end|> — stop token in the LFM chat template

    text = ""
    for token, modality in model.generate_from_chat_state(
        chat, mode="sequential",
        max_new_tokens=max_new_tokens,
        temperature=0.0, top_k=1,
    ):
        if modality == LFMModality.TEXT:
            tok_id = int(token.item())
            if tok_id == IM_END_ID:
                break
            text += processor.tokenizer.decode([tok_id])
    return text.strip()


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_samples(samples_per_fn: int):
    import sys
    # Hide stub torch entirely so datasets never tries to integrate with PyTorch.
    # mlx-audio installs an incomplete torch stub; datasets' streaming mode
    # repeatedly breaks on missing attributes (Generator, nn, utils.data, ...).
    # Removing torch from sys.modules makes datasets treat it as absent.
    _hidden = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k == "torch" or k.startswith("torch.")}
    try:
        from datasets import load_dataset
        # TORCH_AVAILABLE is set at datasets import time; flip it off so
        # IterableDataset._maybe_share_with_torch_persistent_workers returns early
        import datasets.config as _dcfg
        _dcfg.TORCH_AVAILABLE = False

        print(f"[eval] Loading {DATASET} (test split, streaming)...")
        ds = load_dataset(DATASET, split="test", streaming=True)

        by_fn: dict[str, list[dict]] = defaultdict(list)
        total = 0
        for row in ds:
            total += 1
            gt  = row["text_chat"][1]["content"][0]["text"]
            fn  = gt.split("|")[0]
            if len(by_fn[fn]) >= samples_per_fn:
                continue
            audio_raw = row["audio_chat"][0]["content"][0]["audio"]
            by_fn[fn].append({
                "id":           str(row.get("id", "?")),
                "function":     fn,
                "ground_truth": gt,
                "audio":        audio_raw,
            })
    finally:
        sys.modules.update(_hidden)

    samples = []
    for fn in sorted(by_fn):
        samples.extend(by_fn[fn])
    print(f"[eval] {len(samples)} samples across {len(by_fn)} functions (scanned {total} rows)")
    return samples


# ── Report ───────────────────────────────────────────────────────────────────

def render_report(adapter: str | None, results: list[dict]) -> str:
    n = len(results)
    fmt_n  = sum(1 for r in results if r["parseable"])
    fn_n   = sum(1 for r in results if r["function_match"])
    args_n = sum(1 for r in results if r["args_match"])

    def pct(x): return f"{100*x/n:.1f}%" if n else "n/a"

    by_fn: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_fn[r["function"]].append(r)

    lines = [
        f"# Eval report",
        f"",
        f"- adapter: `{adapter or 'none (base model)'}`",
        f"- dataset: `{DATASET}` (test split)",
        f"- samples: {n}",
        f"- system_prompt: `{SYS_PROMPT}`",
        f"",
        f"## Layered metrics",
        f"",
        f"| metric | count | pct |",
        f"|---|---|---|",
        f"| Format compliance (parseable) | {fmt_n}/{n} | {pct(fmt_n)} |",
        f"| Function-name accuracy        | {fn_n}/{n} | {pct(fn_n)} |",
        f"| Argument accuracy             | {args_n}/{n} | {pct(args_n)} |",
        f"",
        f"## Per-function breakdown",
        f"",
        f"| function | n | fmt | name | args |",
        f"|---|---|---|---|---|",
    ]
    for fn in sorted(by_fn):
        items = by_fn[fn]
        m    = len(items)
        p_n  = sum(1 for r in items if r["parseable"])
        f_n  = sum(1 for r in items if r["function_match"])
        a_n  = sum(1 for r in items if r["args_match"])
        lines.append(f"| {fn} | {m} | {p_n}/{m} | {f_n}/{m} | {a_n}/{m} |")

    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",              default=None,
                   help="Path to adapter checkpoint (omit for base model baseline)")
    p.add_argument("--samples-per-function", type=int, default=10)
    p.add_argument("--max-new-tokens",       type=int, default=64)
    p.add_argument("--output",               default="evals/")
    args = p.parse_args()

    model, processor = load_model(args.adapter)
    samples          = load_samples(args.samples_per_function)

    results = []
    for i, s in enumerate(samples, 1):
        try:
            audio, sr = audio_field_to_numpy(s["audio"])
            pred      = transcribe(model, processor, audio, sr, args.max_new_tokens)
        except Exception as e:
            pred = f"<ERROR: {e}>"

        parseable, fn_match, args_match = score(pred, s["ground_truth"])
        results.append({
            "id":             s["id"],
            "function":       s["function"],
            "ground_truth":   s["ground_truth"],
            "prediction":     pred,
            "parseable":      parseable,
            "function_match": fn_match,
            "args_match":     args_match,
        })
        status = "args" if args_match else ("name" if fn_match else ("fmt" if parseable else " no"))
        print(f"[{i:3}/{len(samples)}] {status} | gt:   {s['ground_truth'][:60]}")
        print(f"              pred: {pred[:60]}")

    report = render_report(args.adapter, results)
    print("\n" + report)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = Path(args.adapter).name if args.adapter else "base"
    out_dir = Path(args.output) / f"{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Report: {out_dir}/report.md")


if __name__ == "__main__":
    main()
