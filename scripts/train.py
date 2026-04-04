#!/usr/bin/env python3
"""
train.py — Universal MLX TTS finetuning entry point.

Supports: qwen3_tts | csm | kokoro | chatterbox | personaplex  (model_type in config)

Usage:
    # Qwen3-TTS Hindi LoRA
    python scripts/train.py --config configs/qwen3_tts_hindi.yaml

    # PersonaPlex Hindi LoRA
    python scripts/train.py --config configs/personaplex_hindi.yaml

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


def build_dataset(cfg: dict, split: str = "train", model=None):
    """Build dataset + loader for the configured model type."""
    from data.base_dataset import TTSDataset, DatasetConfig, BatchIterator

    model_type = cfg["model"]["model_type"]
    data_cfg   = cfg["data"]
    t_cfg      = cfg["trainer"]

    # ── PersonaPlex: manifest.json + tokens/*.npz (no JSONL, no raw audio) ──
    if model_type == "personaplex":
        from data.processors.personaplex import PersonaPlexDataset, collate_personaplex

        dir_key  = f"{split}_data_dir"
        data_dir = data_cfg.get(dir_key) or data_cfg.get("train_data_dir")
        if not data_dir or not Path(data_dir).exists():
            print(f"[train] No {split} PersonaPlex data dir at: {data_dir}")
            return None, None

        dataset = PersonaPlexDataset(
            data_dir             = data_dir,
            max_seq_len          = data_cfg.get("max_seq_len",          2048),
            audio_codebooks      = data_cfg.get("audio_codebooks",      16),
            assistant_codebooks  = data_cfg.get("assistant_codebooks",  8),
            audio_pad_token      = data_cfg.get("audio_pad_token",      2048),
            text_pad_token       = data_cfg.get("text_pad_token",       3),
            shuffle              = (split == "train"),
            max_samples          = data_cfg.get("max_samples",          None),
            # train_data_dir / val_data_dir are already separate prepared datasets
            split                = "all",
            val_fraction         = 0.0,
        )
        if (
            split == "val"
            and data_cfg.get("train_data_dir")
            and data_cfg.get("val_data_dir")
            and Path(data_cfg["train_data_dir"]).resolve() == Path(data_cfg["val_data_dir"]).resolve()
        ):
            print("[train] Warning: PersonaPlex train_data_dir and val_data_dir are the same.")
        sort_by_length = t_cfg.get("sort_by_length", True)
        length_key_fn  = (lambda meta: meta.get("num_frames", 0)) if sort_by_length else None
        loader = BatchIterator(
            dataset,
            batch_size     = t_cfg["batch_size"],
            drop_last      = (split == "train"),
            collate_fn     = collate_personaplex,
            sort_by_length = sort_by_length,
            prefetch       = t_cfg.get("prefetch", 2),
            length_key_fn  = length_key_fn,
        )
        return dataset, loader

    # ── All other models: JSONL + raw audio ─────────────────────────────────
    from data.processors.qwen3_tts import Qwen3TTSProcessor, Qwen3TTSProcessorConfig, collate_qwen3
    from data.processors.csm import CSMProcessor, CSMProcessorConfig, collate_csm

    proc_cfg   = cfg.get("processor", {})
    jsonl_key  = f"{split}_jsonl"
    jsonl_path = data_cfg.get(jsonl_key)
    if not jsonl_path or not Path(jsonl_path).exists():
        print(f"[train] No {split} data found at: {jsonl_path}")
        return None, None

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

    if model_type in ("qwen3_tts", "qwen3_tts_speaker"):
        speech_tok = getattr(model, "speech_tokenizer", None) if model is not None else None
        processor = Qwen3TTSProcessor(Qwen3TTSProcessorConfig(
            model_id          = cfg["model"]["model_id"],
            tokenizer_id      = cfg["model"]["tokenizer_id"],
            max_text_len      = proc_cfg.get("max_text_len",  256),
            max_codec_len     = proc_cfg.get("max_codec_len", 1500),
            speaker_name      = proc_cfg.get("speaker_name",  "speaker_0"),
            speech_tokenizer  = speech_tok,
            include_ref_mel   = proc_cfg.get("include_ref_mel", False),
            lang_code         = cfg["trainer"].get("lang_code", "auto"),
        ))
        collate_fn = collate_qwen3

    elif model_type == "csm":
        processor = CSMProcessor(CSMProcessorConfig(
            model_id    = cfg["model"]["model_id"],
            max_seq_len = proc_cfg.get("max_text_len",  2048),
        ))
        collate_fn = collate_csm

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: qwen3_tts, qwen3_tts_speaker, csm, personaplex")

    dataset = TTSDataset(ds_config, processor=processor)
    loader  = BatchIterator(
        dataset,
        batch_size     = t_cfg["batch_size"],
        drop_last      = (split == "train"),
        collate_fn     = collate_fn,
        sort_by_length = t_cfg.get("sort_by_length", False),
        prefetch       = t_cfg.get("prefetch", 2),
    )
    return dataset, loader


def load_model(cfg: dict):
    """Load model via mlx-audio or local models/ directory."""
    model_type = cfg["model"]["model_type"]

    if model_type == "personaplex":
        from models.personaplex import Lm
        from models.personaplex.persona_utils import (
            get_lm_config, get_or_download_model_file, load_lm_weights
        )
        hf_repo    = cfg["model"]["hf_repo"]
        quantized  = cfg["model"].get("quantized", False)
        model_file = cfg["model"].get("model_file", None)

        print(f"[train] Loading PersonaPlex from: {hf_repo}")
        lm_config = get_lm_config(cfg["model"].get("lm_config"), hf_repo)
        model     = Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        quantized = cfg["model"].get("quantized", None)
        resolved_file, _ = get_or_download_model_file(hf_repo, quantized, model_file)
        load_lm_weights(model, lm_config, resolved_file, quantized)
        mx.eval(model.parameters())
        print(f"[train] PersonaPlex loaded: d_model={lm_config.transformer.d_model}")
        return model

    model_id = cfg["model"]["model_id"]
    print(f"[train] Loading model: {model_id}")

    if model_type in ("qwen3_tts", "qwen3_tts_speaker"):
        from mlx_audio.tts.utils import load_model as mlx_load
        model = mlx_load(model_id)
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

    if model_type == "personaplex":
        from train.losses.personaplex_loss import personaplex_loss
        audio_loss_weight = cfg["trainer"].get("audio_loss_weight", 1.0)
        def loss_fn(model, batch):
            return personaplex_loss(model, batch, audio_loss_weight=audio_loss_weight)
        return loss_fn

    elif model_type == "qwen3_tts":
        from train.losses.codec_loss import qwen3_tts_loss
        label_smoothing = cfg["trainer"].get("label_smoothing", 0.0)
        lang_code       = cfg["trainer"].get("lang_code", "auto")
        def loss_fn(model, batch):
            return qwen3_tts_loss(model, batch, label_smoothing=label_smoothing, lang_code=lang_code)
        return loss_fn

    elif model_type == "qwen3_tts_speaker":
        from train.losses.codec_loss import qwen3_tts_speaker_loss
        label_smoothing = cfg["trainer"].get("label_smoothing", 0.0)
        lang_code       = cfg["trainer"].get("lang_code", "auto")
        def loss_fn(model, batch):
            return qwen3_tts_speaker_loss(model, batch, label_smoothing=label_smoothing, lang_code=lang_code)
        return loss_fn

    elif model_type == "csm":
        from train.losses.codec_loss import csm_loss
        def loss_fn(model, batch):
            return csm_loss(model, batch)
        return loss_fn

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def apply_lora(model, cfg: dict) -> int:
    """Apply LoRA + freeze. Returns number of LoRA layers patched."""
    model_type   = cfg["model"]["model_type"]
    lora_cfg_raw = cfg.get("lora", {})

    if model_type == "personaplex":
        from models.personaplex.training import (
            apply_lora_to_transformer, freeze_non_trainable, grad_checkpoint
        )
        rank    = lora_cfg_raw.get("rank",    16)
        alpha   = lora_cfg_raw.get("alpha",   16.0)
        dropout = lora_cfg_raw.get("dropout", 0.0)
        n = apply_lora_to_transformer(model, rank=rank, alpha=alpha, dropout=dropout)

        if cfg["model"].get("grad_checkpoint", False):
            from models.personaplex.modules.transformer import TransformerLayer
            grad_checkpoint(TransformerLayer)
            print("[train] Gradient checkpointing enabled for TransformerLayer")

        train_depformer    = lora_cfg_raw.get("train_depformer", False)
        freeze_text_linear = lora_cfg_raw.get("freeze_text_linear", False)
        num_trainable, num_frozen = freeze_non_trainable(
            model, train_depformer=train_depformer, freeze_text_linear=freeze_text_linear
        )
        print(f"[train] PersonaPlex freeze: {num_trainable:,} trainable / {num_frozen:,} frozen "
              f"(depformer={'trainable' if train_depformer else 'frozen'}, "
              f"text_linear={'frozen' if freeze_text_linear else 'trainable'})")
        return n

    from train.lora import apply_lora as _apply, LoRAConfig
    lora_config = LoRAConfig(
        rank           = lora_cfg_raw.get("rank",    8),
        alpha          = lora_cfg_raw.get("alpha",   16.0),
        dropout        = lora_cfg_raw.get("dropout", 0.05),
        target_modules = lora_cfg_raw.get("target_modules", None),
        model_type     = model_type,
    )
    return _apply(model, lora_config)


def print_param_count(model, model_type: str = ""):
    import mlx.utils as mxu
    if model_type == "personaplex":
        # After freeze_non_trainable(), model.trainable_parameters() returns ~100M params
        flat = dict(mxu.tree_flatten(model.trainable_parameters()))
        trainable = sum(v.size for v in flat.values())
        all_flat  = dict(mxu.tree_flatten(model.parameters()))
        total     = sum(v.size for v in all_flat.values())
    else:
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

    model_type = cfg["model"]["model_type"]
    print(f"\n[smoke-test] Running 5 steps with dummy data ({model_type})...")

    if model_type == "personaplex":
        # (B=2, 17 streams, T=64) — row 0 text, rows 1-16 audio
        B, num_streams, T = 2, 17, 64
        input_tokens  = np.random.randint(0, 2048, (B, num_streams, T), dtype=np.int32)
        target_tokens = np.roll(input_tokens, -1, axis=2)   # naive next-token shift
        batch = {
            "input_tokens":  mx.array(input_tokens),
            "target_tokens": mx.array(target_tokens),
        }
    else:
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
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  step {step+1}: loss={float(loss):.4f}  {metrics_str}  ✅")
        except Exception as e:
            print(f"  step {step+1}: ERROR — {e}  ❌")
            import traceback
            traceback.print_exc()
            return False

    print("[smoke-test] PASSED ✅\n")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Audio eval fn factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_audio_eval_fn(model, model_type: str, cfg: dict, val_dataset, eval_audio_cfg: dict):
    """Create and return the appropriate audio eval function for the model type.

    Returns None if audio eval is not supported or not configured.
    Called only when eval_audio.enabled=true AND tensorboard_dir is set.
    """
    from train.audio_logging import make_personaplex_audio_eval_fn, make_qwen3_tts_audio_eval_fn

    if model_type == "personaplex":
        if val_dataset is None:
            print("[train] audio eval: no val dataset — skipping")
            return None
        base_model = None
        # Get mimi weight path (download from HF if needed — cached after first run)
        try:
            from models.personaplex.persona_utils import (
                get_lm_config,
                get_or_download_mimi,
                get_or_download_model_file,
                load_lm_weights,
            )
            from models.personaplex import Lm
            mimi_weight = get_or_download_mimi(
                cfg["model"]["hf_repo"],
                cfg["model"].get("mimi_weight"),
            )
            if eval_audio_cfg.get("log_base_assistant_pred", True):
                lm_config = get_lm_config(cfg["model"].get("lm_config"), cfg["model"]["hf_repo"])
                model_file, _ = get_or_download_model_file(
                    cfg["model"]["hf_repo"],
                    cfg["model"].get("quantized", None),
                    cfg["model"].get("model_file"),
                )
                base_model = Lm(lm_config)
                base_model.set_dtype(mx.bfloat16)
                load_lm_weights(base_model, lm_config, model_file, cfg["model"].get("quantized", None))
        except Exception as e:
            print(f"[train] audio eval: could not resolve mimi weight: {e} — skipping")
            return None
        return make_personaplex_audio_eval_fn(
            val_dataset,
            eval_audio_cfg,
            mimi_weight,
            base_model=base_model,
        )

    elif model_type in ("qwen3_tts", "qwen3_tts_speaker"):
        return make_qwen3_tts_audio_eval_fn(model, eval_audio_cfg)

    else:
        print(f"[train] audio eval not implemented for model_type={model_type} — skipping")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLX Audio Finetuning")
    parser.add_argument("--config",     required=True,        help="Path to YAML config")
    parser.add_argument("--smoke-test", action="store_true",  help="Run 5 steps with dummy data")
    parser.add_argument("--resume",     default=None,         help="Path to checkpoint dir to resume from")
    parser.add_argument("--lora-rank",  type=int, default=None, help="Override LoRA rank from config")
    parser.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs",     type=int,   default=None, help="Override num_epochs")
    parser.add_argument("--max-steps",  type=int,   default=None, help="Max training steps")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    if args.lora_rank:  cfg["lora"]["rank"] = args.lora_rank
    if args.lr:         cfg["trainer"]["learning_rate"] = args.lr
    if args.epochs:     cfg["trainer"]["num_epochs"] = args.epochs
    if args.max_steps:  cfg["trainer"]["max_steps"] = args.max_steps

    print(f"\n{'='*60}")
    print(f"  MLX Audio Finetuning")
    model_label = cfg['model'].get('model_id') or cfg['model'].get('hf_repo', 'unknown')
    print(f"  Model: {model_label}")
    print(f"  Config: {args.config}")
    print(f"  Device: {mx.default_device()}")
    print(f"{'='*60}")

    # Load model
    model = load_model(cfg)

    # Apply LoRA (personaplex: also calls freeze_non_trainable internally)
    n_lora = apply_lora(model, cfg)

    model_type = cfg["model"]["model_type"]

    # Freeze model-specific submodules that must not be trained.
    # PersonaPlex: freeze_non_trainable() already handled this inside apply_lora.
    # Other models: freeze speech_tokenizer (gc_func breaks value_and_grad) + speaker_encoder.
    if model_type != "personaplex":
        always_freeze = ["speech_tokenizer"]
        if model_type != "qwen3_tts_speaker":
            always_freeze.append("speaker_encoder")
        for attr in always_freeze:
            sub = getattr(model, attr, None)
            if sub is not None:
                sub.freeze()
                print(f"[train] Froze model.{attr}")

    print_param_count(model, model_type=model_type)

    # PersonaPlex-specific trainer hooks
    trainable_params_fn = None
    save_fn             = None
    if model_type == "personaplex":
        from train.lora import get_personaplex_trainable_params, save_personaplex_adapters
        trainable_params_fn = get_personaplex_trainable_params
        save_fn             = save_personaplex_adapters

    # Resume from checkpoint
    if args.resume:
        if model_type == "personaplex":
            from train.lora import load_personaplex_adapters
            npz_path = str(Path(args.resume) / "adapters.npz")
            if os.path.exists(npz_path):
                load_personaplex_adapters(model, npz_path)
                print(f"[train] Resumed PersonaPlex from: {args.resume}")
            else:
                print(f"[train] Warning: no adapters.npz found at {npz_path}")
        else:
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
    val_dataset, val_loader   = build_dataset(cfg, "val",   model=model)
    _,           train_loader = build_dataset(cfg, "train", model=model)

    if train_loader is None:
        print("[train] ERROR: No training data found. Check your config.")
        sys.exit(1)

    # Build trainer config
    from train.trainer import Trainer, TrainerConfig
    t = cfg["trainer"]

    # Build audio eval fn (optional — generates sample audio to TensorBoard after each eval)
    audio_eval_fn = None
    eval_audio_cfg = cfg.get("eval_audio", {})
    if eval_audio_cfg.get("enabled", False) and t.get("tensorboard_dir"):
        audio_eval_fn = _build_audio_eval_fn(model, model_type, cfg, val_dataset, eval_audio_cfg)
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
        tensorboard_dir     = t.get("tensorboard_dir",      None),
    )

    # Save the full config alongside checkpoints so demo.py can read custom_lang_ids etc.
    import shutil
    Path(trainer_config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, Path(trainer_config.output_dir) / "model_config.yaml")

    trainer = Trainer(trainer_config,
                      trainable_params_fn=trainable_params_fn,
                      save_fn=save_fn,
                      audio_eval_fn=audio_eval_fn)
    trainer.train(model, train_loader, loss_fn, val_loader)


if __name__ == "__main__":
    main()
