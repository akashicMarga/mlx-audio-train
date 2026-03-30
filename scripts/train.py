#!/usr/bin/env python3
"""
train.py — Universal MLX TTS finetuning entry point.

Supports: qwen3_tts | csm | kokoro | chatterbox  (model_type in config)

Usage:
    # Qwen3-TTS Hindi LoRA
    python scripts/train.py --config configs/qwen3_tts_hindi.yaml

    # Quick smoke test (5 steps, dummy data)
    python scripts/train.py --config configs/qwen3_tts_hindi.yaml --smoke-test

    # Resume from checkpoint
    python scripts/train.py --config configs/qwen3_tts_hindi.yaml \
        --resume checkpoints/qwen3-hindi/checkpoint-step_0000200
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import mlx.core as mx
import mlx.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str = "train", model=None, max_seq_len: int = None):
    """Build TTSDataset + processor for the configured model type."""
    from data.base_dataset import TTSDataset, DatasetConfig, BatchIterator
    from data.processors.qwen3_tts import Qwen3TTSProcessor, Qwen3TTSProcessorConfig, collate_qwen3
    from data.processors.csm import CSMProcessor, CSMProcessorConfig, collate_csm

    model_type = cfg["model"]["model_type"]
    data_cfg   = cfg["data"]
    proc_cfg   = cfg.get("processor", {})

    jsonl_key = f"{split}_jsonl"
    jsonl_path = data_cfg.get(jsonl_key)
    if not jsonl_path or not Path(jsonl_path).exists():
        print(f"[train] No {split} data found at: {jsonl_path}")
        return None, None

    # Dataset config
    ds_config = DatasetConfig(
        jsonl_path   = jsonl_path,
        target_sr    = data_cfg.get("target_sr",    24000),
        min_duration = data_cfg.get("min_duration", 0.5),
        max_duration = data_cfg.get("max_duration", 20.0),
        normalize    = data_cfg.get("normalize",    True),
        trim         = data_cfg.get("trim",         True),
        shuffle      = (split == "train"),
        max_samples  = data_cfg.get("max_samples",  None),
    )

    # Model-specific processor
    if model_type in ("qwen3_tts", "qwen3_tts_speaker"):
        # Pass model's pre-loaded speech_tokenizer if available
        speech_tok = getattr(model, "speech_tokenizer", None) if model is not None else None
        # max_seq_len CLI arg overrides the config value
        codec_len = max_seq_len or proc_cfg.get("max_codec_len", 1500)
        processor = Qwen3TTSProcessor(Qwen3TTSProcessorConfig(
            model_id          = cfg["model"]["model_id"],
            tokenizer_id      = cfg["model"]["tokenizer_id"],
            max_text_len      = proc_cfg.get("max_text_len",  256),
            max_codec_len     = codec_len,
            speaker_name      = proc_cfg.get("speaker_name",  "speaker_0"),
            speech_tokenizer  = speech_tok,
            # Speaker-cloning pipeline: extract ref mel for speaker_encoder
            include_ref_mel   = proc_cfg.get("include_ref_mel", False),
            # Fallback lang_code for samples that don't have one in the JSONL
            lang_code         = cfg["trainer"].get("lang_code", "auto"),
        ))
        collate_fn = collate_qwen3

    elif model_type == "csm":
        processor = CSMProcessor(CSMProcessorConfig(
            model_id   = cfg["model"]["model_id"],
            max_seq_len = proc_cfg.get("max_text_len",  2048),
        ))
        collate_fn = collate_csm

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: qwen3_tts, qwen3_tts_speaker, csm")

    dataset = TTSDataset(ds_config, processor=processor)
    t_cfg   = cfg["trainer"]

    loader = BatchIterator(
        dataset,
        batch_size     = t_cfg["batch_size"],
        drop_last      = (split == "train"),
        collate_fn     = collate_fn,
        sort_by_length = t_cfg.get("sort_by_length", False),
        prefetch       = t_cfg.get("prefetch", 2),
    )

    return dataset, loader


def load_model(cfg: dict):
    """Load model via mlx-audio."""
    model_type = cfg["model"]["model_type"]
    model_id   = cfg["model"]["model_id"]

    print(f"[train] Loading model: {model_id}")

    if model_type in ("qwen3_tts", "qwen3_tts_speaker"):
        from mlx_audio.tts.utils import load_model as mlx_load
        model = mlx_load(model_id)
        # Register any custom language token IDs defined in the config.
        # Use unused slots in the codec embedding table (IDs 2051–2147 are free).
        # See MODEL_CARD.md for the full list of reserved IDs.
        custom_lang_ids = cfg["model"].get("custom_lang_ids", {})
        if custom_lang_ids:
            model.talker.config.codec_language_id.update(custom_lang_ids)
            print(f"[train] Registered custom lang IDs: {custom_lang_ids}")
        return model

    elif model_type == "csm":
        from mlx_audio.tts.utils import load_model as mlx_load
        model = mlx_load(model_id)
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_loss_fn(cfg: dict):
    """Return the appropriate loss function for the model type."""
    model_type = cfg["model"]["model_type"]
    use_bf16   = cfg["trainer"].get("use_bf16", False)

    if model_type == "qwen3_tts":
        from train.losses.codec_loss import qwen3_tts_loss
        label_smoothing = cfg["trainer"].get("label_smoothing", 0.0)
        lang_code       = cfg["trainer"].get("lang_code", "auto")

        def loss_fn(model, batch):
            return qwen3_tts_loss(
                model, batch,
                label_smoothing=label_smoothing,
                lang_code=lang_code,
                use_bf16=use_bf16,
            )
        return loss_fn

    elif model_type == "qwen3_tts_speaker":
        from train.losses.codec_loss import qwen3_tts_speaker_loss
        label_smoothing = cfg["trainer"].get("label_smoothing", 0.0)
        lang_code       = cfg["trainer"].get("lang_code", "auto")

        def loss_fn(model, batch):
            return qwen3_tts_speaker_loss(
                model, batch,
                label_smoothing=label_smoothing,
                lang_code=lang_code,
                use_bf16=use_bf16,
            )
        return loss_fn

    elif model_type == "csm":
        from train.losses.codec_loss import csm_loss
        def loss_fn(model, batch):
            return csm_loss(model, batch)
        return loss_fn

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def apply_lora(model, cfg: dict) -> int:
    import mlx.core as mx
    from train.lora import apply_lora as _apply, LoRAConfig

    lora_cfg_raw = cfg.get("lora", {})

    # Parse optional lora_dtype (e.g. "bfloat16" → mx.bfloat16)
    lora_dtype_str = lora_cfg_raw.get("lora_dtype", None)
    lora_dtype = None
    if lora_dtype_str == "bfloat16":
        lora_dtype = mx.bfloat16
    elif lora_dtype_str == "float16":
        lora_dtype = mx.float16

    lora_config = LoRAConfig(
        rank           = lora_cfg_raw.get("rank",    8),
        alpha          = lora_cfg_raw.get("alpha",   16.0),
        dropout        = lora_cfg_raw.get("dropout", 0.05),
        target_modules = lora_cfg_raw.get("target_modules", None),
        model_type     = cfg["model"]["model_type"],
        lora_dtype     = lora_dtype,
    )
    n = _apply(model, lora_config)
    return n


def print_param_count(model):
    from train.lora import count_params
    trainable, total = count_params(model)
    pct = 100 * trainable / max(total, 1)
    print(f"[train] Parameters: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test (no real data needed)
# ──────────────────────────────────────────────────────────────────────────────

def run_smoke_test(model, loss_fn, cfg: dict):
    """5-step sanity check with random dummy data."""
    import numpy as np
    import mlx.core as mx

    print("\n[smoke-test] Running 5 steps with dummy data...")

    batch = {
        "text_ids":      mx.array(np.random.randint(0, 1000, (2, 20), dtype=np.int32)),
        "codec_ids":     mx.array(np.random.randint(0, 4096, (2, 50), dtype=np.int32)),
        "text_lengths":  mx.array(np.array([20, 18], dtype=np.int32)),
        "codec_lengths": mx.array(np.array([50, 45], dtype=np.int32)),
        "text_mask":     mx.array(np.ones((2, 20), dtype=bool)),
        "codec_mask":    mx.array(np.ones((2, 50), dtype=bool)),
    }

    for step in range(5):
        try:
            loss, metrics = loss_fn(model, batch)
            mx.eval(loss)
            print(f"  step {step+1}: loss={float(loss):.4f}  ✅")
        except Exception as e:
            print(f"  step {step+1}: ERROR — {e}  ❌")
            import traceback
            traceback.print_exc()
            return False

    print("[smoke-test] PASSED ✅\n")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLX Audio Finetuning")
    parser.add_argument("--config",      required=True,        help="Path to YAML config")
    parser.add_argument("--smoke-test",  action="store_true",  help="Run 5 steps with dummy data")
    parser.add_argument("--resume",      default=None,         help="Path to checkpoint dir to resume from")
    parser.add_argument("--lora-rank",   type=int, default=None, help="Override LoRA rank from config")
    parser.add_argument("--lr",          type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs",      type=int,   default=None, help="Override num_epochs")
    parser.add_argument("--max-steps",   type=int,   default=None, help="Max training steps")
    parser.add_argument("--max-seq-len", type=int,   default=None,
                        help="Override max_codec_len (tokens). Reducing from 1200 to 800 saves ~50%% attention memory.")
    parser.add_argument("--bf16",        action="store_true",
                        help="Enable bfloat16 activations (halves attention/hidden-state memory). "
                             "Equivalent to use_bf16: true in trainer config.")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.lora_rank:   cfg["lora"]["rank"] = args.lora_rank
    if args.lr:          cfg["trainer"]["learning_rate"] = args.lr
    if args.epochs:      cfg["trainer"]["num_epochs"] = args.epochs
    if args.max_steps:   cfg["trainer"]["max_steps"] = args.max_steps
    if args.bf16:        cfg["trainer"]["use_bf16"] = True

    print(f"\n{'='*60}")
    print(f"  MLX Audio Finetuning")
    print(f"  Model: {cfg['model']['model_id']}")
    print(f"  Config: {args.config}")
    print(f"  Device: {mx.default_device()}")
    print(f"{'='*60}")

    # Load model
    model = load_model(cfg)

    # Apply LoRA
    n_lora = apply_lora(model, cfg)

    model_type = cfg["model"]["model_type"]

    # Freeze parts that must not be trained.
    #
    # speech_tokenizer — always frozen: contains a gc_func (compiled decoder)
    #   that breaks model.update() inside nn.value_and_grad.
    #
    # speaker_encoder  — frozen for language-adaptation pipeline (not used).
    #   For the speaker-cloning pipeline (qwen3_tts_speaker) we keep it
    #   UNfrozen so it can run in the forward pass to extract speaker
    #   embeddings. Its weights are still NOT updated by the optimizer because
    #   get_trainable_params() returns only LoRA tensors.  mx.stop_gradient()
    #   in the loss function prevents gradients from flowing into it.
    always_freeze = ["speech_tokenizer"]
    if model_type != "qwen3_tts_speaker":
        always_freeze.append("speaker_encoder")

    for attr in always_freeze:
        sub = getattr(model, attr, None)
        if sub is not None:
            sub.freeze()
            print(f"[train] Froze model.{attr}")

    print_param_count(model)

    # Resume from checkpoint
    if args.resume:
        from train.lora import load_adapters
        adapter_path = str(Path(args.resume) / "adapters.safetensors")
        if os.path.exists(adapter_path):
            load_adapters(model, adapter_path)
            print(f"[train] Resumed from: {args.resume}")
        else:
            print(f"[train] Warning: no adapters found at {adapter_path}")

    # Build loss function
    loss_fn = build_loss_fn(cfg)

    # Smoke test mode
    if args.smoke_test:
        run_smoke_test(model, loss_fn, cfg)
        return

    # Build data loaders (pass model so processor reuses its speech_tokenizer)
    _, train_loader = build_dataset(cfg, "train", model=model, max_seq_len=args.max_seq_len)
    _, val_loader   = build_dataset(cfg, "val",   model=model, max_seq_len=args.max_seq_len)

    if train_loader is None:
        print("[train] ERROR: No training data found. Check your config.")
        sys.exit(1)

    # Build trainer
    from train.trainer import Trainer, TrainerConfig
    t = cfg["trainer"]
    trainer_config = TrainerConfig(
        output_dir          = t.get("output_dir",           "./checkpoints"),
        run_name            = t.get("run_name",             "run"),
        num_epochs          = t.get("num_epochs",           10),
        batch_size          = t.get("batch_size",           4),
        grad_accumulation   = t.get("grad_accumulation",    4),
        max_steps           = t.get("max_steps",            None),
        learning_rate       = t.get("learning_rate",        2e-4),
        weight_decay        = t.get("weight_decay",         0.01),
        grad_clip           = t.get("grad_clip",            1.0),
        warmup_steps        = t.get("warmup_steps",         100),
        lr_schedule         = t.get("lr_schedule",          "cosine"),
        save_every_n_steps  = t.get("save_every_n_steps",   200),
        save_every_n_epochs = t.get("save_every_n_epochs",  1),
        keep_last_n         = t.get("keep_last_n",          3),
        eval_every_n_steps  = t.get("eval_every_n_steps",   100),
        val_batches         = t.get("val_batches",          20),
        log_every_n_steps   = t.get("log_every_n_steps",    10),
        log_file            = t.get("log_file",             None),
        label_smoothing     = t.get("label_smoothing",      0.0),
        use_bf16            = t.get("use_bf16",             False),
        clear_cache_steps   = t.get("clear_cache_steps",    0),
        # Pass resume path so trainer can load optimizer state + counters.
        # Adapter weights are loaded above (before trainer is constructed),
        # optimizer state is loaded inside trainer.train() after AdamW is created.
        resume_from         = args.resume,
    )

    # Save the full config alongside checkpoints so demo.py can read custom_lang_ids etc.
    import shutil
    shutil.copy(args.config, Path(trainer_config.output_dir) / "model_config.yaml")

    trainer = Trainer(trainer_config)
    trainer.train(model, train_loader, loss_fn, val_loader)


if __name__ == "__main__":
    main()
