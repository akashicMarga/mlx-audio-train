#!/usr/bin/env python3
"""
train.py — Universal MLX TTS finetuning entry point.

Supports: qwen3_tts | csm | kokoro | chatterbox | personaplex | lfm_audio  (model_type in config)

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

    elif model_type == "lfm_audio":
        from data.processors.lfm_audio import LFMAudioProcessor, LFMAudioProcessorConfig, collate_lfm_audio
        # Reuse pre-loaded LFM processor from model if available
        lfm_proc = getattr(model, "processor", None) if model is not None else None
        processor = LFMAudioProcessor(LFMAudioProcessorConfig(
            model_id         = cfg["model"]["model_id"],
            max_text_len     = proc_cfg.get("max_text_len",      256),
            max_audio_frames = proc_cfg.get("max_audio_frames",  512),
            training_mode    = proc_cfg.get("training_mode",     "tts"),
            system_prompt    = proc_cfg.get("system_prompt",     "You are a text-to-speech assistant."),
            processor        = lfm_proc,
        ))
        collate_fn = collate_lfm_audio

    elif model_type == "indic_parler_tts":
        from data.processors.indic_parler import (
            IndicParlerProcessor, IndicParlerProcessorConfig, IndicParlerDataset,
        )
        proc_cfg  = cfg.get("processor", {})
        processor = IndicParlerProcessor(IndicParlerProcessorConfig(
            hf_repo            = cfg["model"]["hf_repo"],
            max_desc_len       = data_cfg.get("max_desc_len",       128),
            max_prompt_len     = data_cfg.get("max_prompt_len",     256),
            max_frames         = data_cfg.get("max_frames",         860),
            num_codebooks      = data_cfg.get("num_codebooks",      9),
            fixed_description  = data_cfg.get("fixed_description",  None),
        ))
        dataset = IndicParlerDataset(
            jsonl_path  = jsonl_path,
            processor   = processor,
            shuffle     = (split == "train"),
            max_samples = data_cfg.get("max_samples", None),
        )
        loader = BatchIterator(
            dataset,
            batch_size = t_cfg["batch_size"],
            drop_last  = (split == "train"),
            collate_fn = processor.collate,
            prefetch   = t_cfg.get("prefetch", 2),
        )
        return dataset, loader

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: qwen3_tts, qwen3_tts_speaker, csm, lfm_audio, personaplex, indic_parler_tts")

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

    if model_type == "indic_parler_tts":
        from models.indic_parler_tts.generate import load_model as parler_load
        hf_repo = cfg["model"]["hf_repo"]
        model, tokenizers = parler_load(hf_repo)
        model._eval_tokenizers = tokenizers
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

    elif model_type == "lfm_audio":
        from mlx_audio.sts.models.lfm_audio import LFM2AudioModel
        print(f"[train] Loading LFM 2.5 Audio from: {model_id}")
        model = LFM2AudioModel.from_pretrained(model_id)
        mx.eval(model.parameters())
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_grpo_prompts(cfg: dict, model, jsonl_override: str = None,
                       max_samples_override: int = None) -> list:
    """Build the GRPO prompt list from the train jsonl.

    GRPO only needs prompt TEXT (codec labels are unused unless sft_lambda>0).
    Text is tokenised via the SAME Qwen3TTSProcessor.encode_text the SFT used,
    so rollouts run in the regime the SFT adapter was trained on.

    `jsonl_override` / `max_samples_override` let the caller build a separate
    fixed set (e.g. the held-out in-loop eval prompts) without touching cfg.
    """
    import mlx.core as mx
    from data.processors.qwen3_tts import Qwen3TTSProcessor, Qwen3TTSProcessorConfig

    data_cfg  = cfg.get("data", {})
    proc_cfg  = cfg.get("processor", {})
    jsonl_path = jsonl_override or data_cfg.get("train_jsonl")
    if not jsonl_path or not Path(jsonl_path).exists():
        raise FileNotFoundError(f"[grpo] jsonl not found: {jsonl_path}")

    lang_code = cfg["trainer"].get("lang_code", "auto")
    processor = Qwen3TTSProcessor(Qwen3TTSProcessorConfig(
        model_id     = cfg["model"]["model_id"],
        tokenizer_id = cfg["model"]["tokenizer_id"],
        max_text_len = proc_cfg.get("max_text_len", 256),
        lang_code    = lang_code,
    ))

    # Pipeline 2: when the speaker-similarity reward is on, each prompt also needs
    # a speaker embedding (for the rollout prefix) and a ref_mel (for the reward),
    # both derived from the record's ref_audio. The frozen speaker_encoder makes
    # spk_embeds adapter-independent, so compute it once here.
    want_speaker = cfg.get("grpo", {}).get("rewards", {}).get(
        "speaker_similarity", {}).get("weight", 0.0) > 0
    if want_speaker and getattr(model, "speaker_encoder", None) is None:
        raise RuntimeError("[grpo] speaker_similarity reward needs model.speaker_encoder")
    sr = data_cfg.get("target_sr", 24000)

    def _speaker_fields(rec):
        from data.audio_utils import load_audio, mel_spectrogram
        ref_audio = rec.get("ref_audio")
        if not ref_audio or not Path(ref_audio).exists():
            return None
        wav, _ = load_audio(ref_audio, target_sr=sr)
        ref_mel = mx.array(mel_spectrogram(wav, sr=sr))[None, ...]   # [1, T, 128]
        spk = mx.stop_gradient(model.speaker_encoder(ref_mel))        # [1, D]
        # ref_mel → reward; spk_embeds → concatenated-layout prefix; ref_audio →
        # interleaved-layout prefix (generate() re-extracts the speaker from it).
        return {"spk_embeds": spk, "ref_mel": ref_mel, "ref_audio": ref_audio}

    max_samples = max_samples_override if max_samples_override is not None \
        else data_cfg.get("max_samples", None)
    prompts, skipped = [], 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if not text:
                continue
            prompt = {
                "text_ids":  mx.array(processor.encode_text(text)),
                "text":      text,
                "lang_code": rec.get("lang_code", lang_code),
            }
            if want_speaker:
                spk = _speaker_fields(rec)
                if spk is None:
                    skipped += 1
                    continue
                prompt.update(spk)
            prompts.append(prompt)
            if max_samples and len(prompts) >= max_samples:
                break
    note = f" ({skipped} skipped: missing ref_audio)" if skipped else ""
    print(f"[grpo] built {len(prompts)} prompts from {jsonl_path}{note}")
    return prompts


def build_grpo_eval_prompts(cfg: dict, model, prompts: list) -> list:
    """Fixed prompt set for the in-loop held-out eval.

    Prefers a dedicated `grpo.eval.jsonl` (truly held out); otherwise falls back
    to a deterministic slice of the training prompts (legibility tracking only —
    these overlap training, so treat the curve as a learning signal, not a
    generalisation measure).
    """
    ev = cfg.get("grpo", {}).get("eval", {})
    n  = ev.get("num_prompts", 8)
    eval_jsonl = ev.get("jsonl")
    if eval_jsonl:
        return build_grpo_prompts(cfg, model, jsonl_override=eval_jsonl,
                                  max_samples_override=n)
    return prompts[:n]


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

    elif model_type == "indic_parler_tts":
        from train.losses.parler_loss import parler_tts_loss
        label_smoothing = cfg["trainer"].get("label_smoothing", 0.0)
        def loss_fn(model, batch):
            return parler_tts_loss(model, batch, label_smoothing=label_smoothing)
        return loss_fn

    elif model_type == "lfm_audio":
        from train.losses.lfm_audio_loss import lfm_audio_tts_loss, lfm_audio_asr_loss
        label_smoothing   = cfg["trainer"].get("label_smoothing",   0.0)
        text_loss_weight  = cfg["trainer"].get("text_loss_weight",  0.0)
        training_mode     = cfg.get("processor", {}).get("training_mode", "tts")
        if training_mode == "asr":
            def loss_fn(model, batch):
                return lfm_audio_asr_loss(model, batch, label_smoothing=label_smoothing)
        else:
            def loss_fn(model, batch):
                return lfm_audio_tts_loss(
                    model, batch,
                    label_smoothing=label_smoothing,
                    text_loss_weight=text_loss_weight,
                )
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
    elif model_type == "lfm_audio":
        training_mode = cfg.get("processor", {}).get("training_mode", "tts")
        if training_mode == "asr":
            # ASR: audio_features (mel) in, text tokens out
            batch = {
                "text_ids":       mx.array(np.random.randint(0, 1000, (2, 20),      dtype=np.int32)),
                "audio_features": mx.array(np.zeros((2, 100, 128),                  dtype=np.float32)),
                "text_lengths":   mx.array(np.array([20, 18],                        dtype=np.int32)),
                "text_mask":      mx.array(np.ones((2, 20),                          dtype=bool)),
            }
        else:
            # TTS: text conditioning in, audio codes [B, T, 8] out
            batch = {
                "text_ids":      mx.array(np.random.randint(0, 1000, (2, 20),     dtype=np.int32)),
                "audio_codes":   mx.array(np.random.randint(0, 2049, (2, 50, 8),  dtype=np.int32)),
                "text_lengths":  mx.array(np.array([20, 18], dtype=np.int32)),
                "audio_lengths": mx.array(np.array([50, 45], dtype=np.int32)),
                "text_mask":     mx.array(np.ones((2, 20), dtype=bool)),
                "audio_mask":    mx.array(np.ones((2, 50), dtype=bool)),
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

    elif model_type == "indic_parler_tts":
        from train.audio_logging import make_indic_parler_audio_eval_fn
        tokenizers = getattr(model, "_eval_tokenizers", None)
        if tokenizers is None:
            print("[train] audio eval: model has no _eval_tokenizers — skipping")
            return None
        return make_indic_parler_audio_eval_fn(model, tokenizers, eval_audio_cfg)

    else:
        print(f"[train] audio eval not implemented for model_type={model_type} — skipping")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_grpo(model, cfg: dict, args):
    """GRPO post-training entry point (Pipeline 3).

    Loads the SFT adapter (becomes initial policy AND frozen KL reference),
    builds the prompt list, and runs GRPOTrainer.train_grpo. Reuses the base
    Trainer machinery (checkpointing/LR/logging) via GRPOTrainer.
    """
    from train.grpo.trainer import GRPOTrainer
    from train.grpo.rewards import RewardConfig
    from train.lora import load_adapters
    from train.trainer import TrainerConfig

    g = cfg.get("grpo", {})
    t = cfg["trainer"]

    if getattr(model, "speech_tokenizer", None) is None:
        print("[grpo] ERROR: model.speech_tokenizer is required for rollout decode.")
        sys.exit(1)

    # Load the SFT adapter — the starting policy and (snapshotted) reference.
    init_adapters = g.get("init_adapters")
    if init_adapters and os.path.exists(init_adapters):
        load_adapters(model, init_adapters)
        print(f"[grpo] loaded SFT adapter (policy + reference): {init_adapters}")
    elif init_adapters:
        print(f"[grpo] WARNING: init_adapters not found ({init_adapters}); "
              f"starting from BASE — KL will anchor to base, not SFT.")
    else:
        print("[grpo] WARNING: no init_adapters set; starting from BASE adapter.")

    prompts = build_grpo_prompts(cfg, model)
    if not prompts:
        print("[grpo] ERROR: no prompts built. Check data.train_jsonl.")
        sys.exit(1)

    # Fixed prompt set for the in-loop held-out eval (legibility).
    eval_prompts = build_grpo_eval_prompts(cfg, model, prompts)

    # Reward config from the YAML rewards block.
    rw = g.get("rewards", {})
    intel = rw.get("intelligibility", {})
    length = rw.get("length_penalty", {})
    speaker = rw.get("speaker_similarity", {})
    reward_cfg = RewardConfig(
        w_intel   = intel.get("weight",   1.0),
        w_length  = length.get("weight",  0.5),
        w_speaker = speaker.get("weight", 0.0),
        asr_model = intel.get("asr_model", "mlx-community/whisper-large-v3-turbo"),
        language  = intel.get("language",  cfg["trainer"].get("lang_code", "auto")),
        metric    = intel.get("metric",    "cer"),
        no_eos_penalty = length.get("no_eos_penalty", 1.0),
        silence_penalty = length.get("silence_penalty", 0.5),
        speaking_rate_min_cps = length.get("speaking_rate_min_cps", 0.0),
        speaking_rate_penalty = length.get("speaking_rate_penalty", 0.5),
    )

    trainer_config = TrainerConfig(
        output_dir          = t.get("output_dir",          "./checkpoints/grpo"),
        run_name            = t.get("run_name",            "grpo"),
        num_epochs          = t.get("num_epochs",          100),
        grad_accumulation   = t.get("grad_accumulation",   g.get("prompts_per_step", 4)),
        max_steps           = t.get("max_steps",           300),
        learning_rate       = t.get("learning_rate",       5e-6),
        weight_decay        = t.get("weight_decay",        0.01),
        grad_clip           = t.get("grad_clip",           1.0),
        warmup_steps        = t.get("warmup_steps",        10),
        lr_schedule         = t.get("lr_schedule",         "constant"),
        save_every_n_steps  = t.get("save_every_n_steps",  50),
        save_every_n_epochs = t.get("save_every_n_epochs", 999999),
        keep_last_n         = t.get("keep_last_n",         3),
        log_every_n_steps   = t.get("log_every_n_steps",   1),
        eval_every_n_steps  = t.get("eval_every_n_steps",  25),
        log_file            = t.get("log_file",            None),
        tensorboard_dir     = t.get("tensorboard_dir",     None),
    )

    # SFT-mixin: build a real SFT batch loader only when the term is active
    # (it needs codec labels, i.e. a pre-tokenized train_codes.jsonl).
    sft_lambda = g.get("sft_lambda", 0.0)
    sft_loader = None
    if sft_lambda > 0:
        _, sft_loader = build_dataset(cfg, "train", model=model)
        if sft_loader is None:
            print("[grpo] WARNING: sft_lambda>0 but no SFT loader built; mixin disabled.")
            sft_lambda = 0.0

    import shutil
    Path(trainer_config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, Path(trainer_config.output_dir) / "model_config.yaml")

    trainer = GRPOTrainer(trainer_config)
    trainer.train_grpo(
        model, prompts,
        reward_cfg     = reward_cfg,
        group_size     = g.get("group_size",     4),
        max_new_tokens = g.get("max_new_tokens",  240),
        temperature    = g.get("temperature",     0.9),
        top_p          = g.get("top_p",            0.95),
        top_k          = g.get("top_k",            50),
        lang_code      = t.get("lang_code",       "auto"),
        layout         = g.get("layout",          "interleaved"),
        kl_beta        = g.get("kl_beta",          0.05),
        kl_clip        = g.get("kl_clip",          10.0),
        sub_pg_weight  = g.get("sub_pg_weight",    0.0),
        pg_norm        = g.get("pg_norm",          "token"),
        sft_lambda     = sft_lambda,
        sft_loader     = sft_loader,
        skip_zero_variance = g.get("skip_zero_variance", True),
        eval_prompts   = eval_prompts,
        eval_group_size = g.get("eval", {}).get("group_size", 2),
    )


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
    if model_type == "lfm_audio":
        # Freeze audio_encoder (ConformerEncoder) and audio_head (Depthformer).
        # LoRA is applied only to model.lfm (the LFM transformer backbone).
        for attr in ("audio_encoder", "audio_head", "audio_embedding", "depth_embeddings",
                     "depth_linear", "audio_adapter", "detokenizer"):
            sub = getattr(model, attr, None)
            if sub is None:
                continue
            if isinstance(sub, list):
                # depth_embeddings is a list of nn.Module items
                for item in sub:
                    if hasattr(item, "freeze"):
                        item.freeze()
            else:
                sub.freeze()
            print(f"[train] Froze model.{attr}")
    elif model_type != "personaplex":
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

    # ── GRPO post-training (Pipeline 3) ─────────────────────────────────────
    # Dispatched before the SFT path: it has its own loss, trainer, and data
    # (prompt text only). Requires model.speech_tokenizer (for rollout decode).
    if cfg.get("pipeline") == "grpo":
        run_grpo(model, cfg, args)
        return

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
