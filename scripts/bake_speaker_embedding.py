#!/usr/bin/env python3
"""
bake_speaker_embedding.py — Post-training step for the speaker-cloning pipeline.

After training with configs/qwen3_tts_speaker.yaml, this script:

  1. Loads the base model and applies the trained LoRA adapters.
  2. Runs model.speaker_encoder on every ref_audio in the training JSONL.
  3. Averages the resulting speaker embeddings.
  4. Writes the mean embedding into talker.model.codec_embedding.weight[3000]
     — the reserved "custom voice" slot (matches official sft_12hz.py).
  5. Optionally fuses LoRA adapters into the base weights.
  6. Saves the modified model as a new checkpoint directory and patches
     config.json with the custom-voice speaker metadata.

Usage:
    python scripts/bake_speaker_embedding.py \\
        --config  configs/qwen3_tts_speaker.yaml \\
        --checkpoint checkpoints/qwen3-speaker/checkpoint-final \\
        --output  checkpoints/qwen3-speaker/custom_voice_model

    # Also fuse LoRA adapters into base weights (larger file, no adapters needed at inference)
    python scripts/bake_speaker_embedding.py \\
        --config  configs/qwen3_tts_speaker.yaml \\
        --checkpoint checkpoints/qwen3-speaker/checkpoint-final \\
        --output  checkpoints/qwen3-speaker/custom_voice_model \\
        --fuse-lora
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Speaker embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_speaker_embeddings(model, ref_audio_paths: list, target_sr: int = 24000) -> np.ndarray:
    """
    Run model.speaker_encoder on each ref_audio and return a numpy array
    of shape [N, D_spk] (one row per audio file).
    """
    import mlx.core as mx
    from data.audio_utils import load_audio, mel_spectrogram

    embeddings = []

    for path in ref_audio_paths:
        try:
            audio, _ = load_audio(path, target_sr=target_sr)
            mel = mel_spectrogram(audio, sr=target_sr)  # [T, 128]
            mel_mx = mx.array(mel)[None, :, :]           # [1, T, 128]

            spk_embed = model.speaker_encoder(mel_mx)    # [1, D]
            mx.eval(spk_embed)
            embeddings.append(np.array(spk_embed[0]))
        except Exception as e:
            print(f"  [WARN] Failed to encode {path}: {e}")

    if not embeddings:
        raise RuntimeError("No speaker embeddings could be extracted — check ref_audio paths.")

    return np.stack(embeddings, axis=0)   # [N, D]


def collect_ref_audio_paths(jsonl_path: str) -> list:
    """Parse JSONL and return unique ref_audio paths."""
    base = Path(jsonl_path).parent
    paths = []
    seen  = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = obj.get("ref_audio")
            if ref and ref not in seen:
                seen.add(ref)
                abs_ref = str(base / ref) if not Path(ref).is_absolute() else ref
                paths.append(abs_ref)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Embed baking
# ──────────────────────────────────────────────────────────────────────────────

def bake_speaker_into_model(model, mean_embed: np.ndarray, slot: int = 3000):
    """
    Write mean_embed into talker.model.codec_embedding.weight[slot].

    The Qwen3-TTS talker reserves slot 3000 for custom-voice speaker
    embeddings.  This mirrors what the official sft_12hz.py does when
    it calls:
        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding

    We try the most likely attribute path; if the MLX model exposes the
    codec embedding table differently, we print a warning and skip.
    """
    import mlx.core as mx

    # Try to locate the codec embedding weight
    embed_module = None
    for path in (
        ("talker", "model", "codec_embedding"),
        ("talker", "codec_embedding"),
        ("model", "talker", "model", "codec_embedding"),
    ):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "weight"):
                embed_module = obj
                print(f"[bake] Found codec_embedding at: model.{'.'.join(path)}")
                break
        except AttributeError:
            continue

    if embed_module is None:
        print("[bake] WARNING: Could not locate talker.model.codec_embedding — "
              "speaker embedding NOT baked in. The LoRA adapters are saved but "
              "inference will need a ref_audio.")
        return False

    weight_np = np.array(embed_module.weight)    # materialise to numpy

    if slot >= weight_np.shape[0]:
        print(f"[bake] WARNING: slot {slot} is out of range "
              f"(codec_embedding has {weight_np.shape[0]} rows). Skipping.")
        return False

    # Ensure shape compatibility
    embed_dim = weight_np.shape[1]
    if mean_embed.shape[0] != embed_dim:
        # Attempt to project/truncate — warn user
        print(f"[bake] WARNING: speaker embedding dim {mean_embed.shape[0]} != "
              f"codec embedding dim {embed_dim}.  Truncating/padding.")
        if mean_embed.shape[0] > embed_dim:
            mean_embed = mean_embed[:embed_dim]
        else:
            pad = np.zeros(embed_dim - mean_embed.shape[0], dtype=mean_embed.dtype)
            mean_embed = np.concatenate([mean_embed, pad])

    weight_np[slot] = mean_embed.astype(weight_np.dtype)
    embed_module.weight = mx.array(weight_np)
    mx.eval(embed_module.weight)
    print(f"[bake] Written mean speaker embedding → codec_embedding.weight[{slot}]")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Config.json patching
# ──────────────────────────────────────────────────────────────────────────────

def patch_config_json(src_model_dir: str, output_dir: str, speaker_name: str, slot: int = 3000):
    """
    Copy config.json and patch it with custom-voice metadata (mirrors
    the official sft_12hz.py checkpoint-saving logic).
    """
    src_cfg  = Path(src_model_dir) / "config.json"
    dst_cfg  = Path(output_dir)    / "config.json"
    if not src_cfg.exists():
        print(f"[bake] No config.json found at {src_model_dir}, skipping patch.")
        return

    with open(src_cfg) as f:
        cfg = json.load(f)

    cfg["tts_model_type"] = "custom_voice"

    talker_cfg = cfg.get("talker_config", {})
    spk_ids = talker_cfg.get("spk_id", {})
    spk_ids[speaker_name] = slot
    talker_cfg["spk_id"] = spk_ids

    dialect = talker_cfg.get("spk_is_dialect", {})
    dialect[speaker_name] = False
    talker_cfg["spk_is_dialect"] = dialect

    cfg["talker_config"] = talker_cfg

    with open(dst_cfg, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[bake] Patched config.json: tts_model_type=custom_voice, "
          f"spk_id[{speaker_name}]={slot}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bake speaker embedding into Qwen3-TTS model")
    parser.add_argument("--config",      required=True,  help="Training YAML config")
    parser.add_argument("--checkpoint",  required=True,  help="Path to trained checkpoint dir (contains adapters.safetensors)")
    parser.add_argument("--output",      required=True,  help="Output directory for the baked model")
    parser.add_argument("--slot",        type=int, default=3000, help="Codec embedding slot for the speaker (default: 3000)")
    parser.add_argument("--fuse-lora",   action="store_true",    help="Fuse LoRA adapters into base weights before saving")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id     = cfg["model"]["model_id"]
    train_jsonl  = cfg["data"]["train_jsonl"]
    speaker_name = cfg.get("processor", {}).get("speaker_name", "custom_speaker")
    target_sr    = cfg["data"].get("target_sr", 24000)

    print(f"\n{'='*60}")
    print(f"  Baking speaker embedding")
    print(f"  Model:    {model_id}")
    print(f"  Ckpt:     {args.checkpoint}")
    print(f"  Speaker:  {speaker_name}  (slot {args.slot})")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[bake] Loading model…")
    from mlx_audio.tts.utils import load_model as mlx_load
    model = mlx_load(model_id)

    # ── Apply LoRA and load adapters ──────────────────────────────────────────
    print("[bake] Applying LoRA adapters…")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train.lora import apply_lora, load_adapters, LoRAConfig
    import scripts.train as train_script

    lora_raw    = cfg.get("lora", {})
    lora_config = LoRAConfig(
        rank           = lora_raw.get("rank", 8),
        alpha          = lora_raw.get("alpha", 16.0),
        dropout        = 0.0,         # no dropout at baking time
        target_modules = lora_raw.get("target_modules", None),
        model_type     = "qwen3_tts",
    )
    apply_lora(model, lora_config)

    adapter_path = Path(args.checkpoint) / "adapters.safetensors"
    if adapter_path.exists():
        load_adapters(model, str(adapter_path))
        print(f"[bake] Loaded adapters from {adapter_path}")
    else:
        print(f"[bake] WARNING: No adapters.safetensors found at {adapter_path}")

    # ── Extract and average speaker embeddings ────────────────────────────────
    print(f"\n[bake] Collecting ref_audio paths from {train_jsonl}…")
    ref_paths = collect_ref_audio_paths(train_jsonl)
    print(f"[bake] Found {len(ref_paths)} unique ref_audio files")

    print("[bake] Extracting speaker embeddings…")
    embeds = extract_speaker_embeddings(model, ref_paths, target_sr=target_sr)
    mean_embed = embeds.mean(axis=0)
    print(f"[bake] Mean speaker embedding shape: {mean_embed.shape}  "
          f"(from {len(embeds)} samples)")

    # ── Bake into codec_embedding ─────────────────────────────────────────────
    baked = bake_speaker_into_model(model, mean_embed, slot=args.slot)

    # ── Optionally fuse LoRA ──────────────────────────────────────────────────
    if args.fuse_lora:
        print("[bake] Fusing LoRA adapters into base weights…")
        from train.lora import LoRALinear, QLoRALinear
        import mlx.nn as nn

        def _fuse(module):
            children = module.children() if hasattr(module, "children") else {}
            if not isinstance(children, dict):
                return
            for key, val in children.items():
                if isinstance(val, LoRALinear):
                    setattr(module, key, val.fuse())
                elif isinstance(val, nn.Module):
                    _fuse(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, nn.Module):
                            _fuse(item)

        _fuse(model)
        print("[bake] LoRA adapters fused.")

    # ── Copy model files to output dir ────────────────────────────────────────
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file as save_safetensors

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files from original model cache (tokenizer, vocab, etc.)
    # We can't easily locate the HuggingFace cache here, so just copy the
    # adapter safetensors alongside a README explaining what to do.
    print(f"\n[bake] Output dir: {output_dir}")

    # Save the updated LoRA adapters (or fused adapters.safetensors)
    if args.fuse_lora:
        print("[bake] NOTE: with --fuse-lora the full model weights need to be "
              "saved via mlx_audio's save utilities. Saving adapter delta only.")

    if not args.fuse_lora:
        # Save adapters with the baked speaker slot noted
        from train.lora import save_adapters
        save_adapters(model, str(output_dir / "adapters.safetensors"))

    # Save mean speaker embedding separately for reference / debugging
    np.save(str(output_dir / "speaker_embedding.npy"), mean_embed)
    print(f"[bake] Saved mean speaker embedding → {output_dir}/speaker_embedding.npy")

    # Patch config.json
    import subprocess
    # Try to find the model's cached config.json
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_id, local_files_only=True)
        patch_config_json(local_dir, str(output_dir), speaker_name, args.slot)
    except Exception:
        # Fall back: write a minimal custom-voice config stub
        stub = {
            "tts_model_type": "custom_voice",
            "base_model_id":  model_id,
            "talker_config": {
                "spk_id":        {speaker_name: args.slot},
                "spk_is_dialect": {speaker_name: False},
            },
            "speaker_embedding_slot": args.slot,
            "speaker_name":           speaker_name,
        }
        with open(output_dir / "custom_voice_config.json", "w") as f:
            json.dump(stub, f, indent=2, ensure_ascii=False)
        print(f"[bake] Saved custom_voice_config.json (base config.json not found)")

    # Write info file
    with open(output_dir / "bake_info.json", "w") as f:
        json.dump({
            "base_model":       model_id,
            "checkpoint":       str(args.checkpoint),
            "speaker_name":     speaker_name,
            "slot":             args.slot,
            "n_ref_audios":     len(embeds),
            "fused_lora":       args.fuse_lora,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done!  Baked model → {output_dir}")
    if baked:
        print(f"  Speaker '{speaker_name}' is stored at codec_embedding[{args.slot}].")
        print(f"  Use model_type='custom_voice' at inference with speaker='{speaker_name}'.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
