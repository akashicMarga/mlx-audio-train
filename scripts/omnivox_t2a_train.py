"""
OmniVox T2A Training — Hindi text → Mimi speech codes (pure MLX).

Trains Qwen2.5-0.5B Thinker + Talker to generate Hindi speech from Hindi text.
Audio projector (Whisper) stays frozen / random — this phase establishes
the model's ability to SPEAK Hindi before we add speech INPUT in A2A.

Trainable (~60.9M params):
  talker.*       ~59.4M   (all Talker layers, embeddings, heads)
  audio_proj     ~1.5M    (projector — kept in loop so A2A can warm-start from here)
  qwen.*         FROZEN

Loss (per step):
  text_loss   = cross-entropy on assistant text tokens (same as LM pre-training)
  audio_loss  = cross-entropy on next Mimi code, averaged across 8 codebooks
  total_loss  = text_loss + audio_loss

Sequence layout:
  input_ids  (B, T)    — text token ids (ChatML format)
  audio_ids  (B, 8, T) — Mimi code history aligned to text length T
                          (codes subsampled/padded to match T)

Monitoring:
  tensorboard --logdir <save_dir>/tensorboard
  tail -f <save_dir>/train_log.jsonl

Usage:
    # Prepare data first (one-time, ~5 min for 1900 samples):
    python scripts/omnivox_prepare_t2a.py \\
        --input    /Users/akashsingh/Documents/exps/hindi/train.jsonl \\
        --audio-dir /Users/akashsingh/Documents/exps/hindi \\
        --output   /Users/akashsingh/Documents/exps/omnivox_t2a/train.parquet

    # Then train:
    python scripts/omnivox_t2a_train.py \\
        --config  configs/omnivox/whisper_small_t2a.yaml \\
        --data    /Users/akashsingh/Documents/exps/omnivox_t2a/train.parquet \\
        --val-data /Users/akashsingh/Documents/exps/omnivox_t2a/val.parquet
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from models.omnivox.config import load_omnivox_config, OmniVoxConfig
from models.omnivox.model import OmniVox
from scripts.minimind_o_train_utils import (
    build_tokens_with_audio,
    TrainingLogger,
)

MAX_SEQ_LEN    = 512
AUDIO_PAD_CODE = 2049   # Mimi pad token (audio_pad_token in OmniVoxConfig)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_t2a_samples(parquet_path: str, split_name: str = "train") -> list[dict]:
    """
    Load T2A parquet → list of {convs, codes_np (8, T_frames)}.
    """
    import pyarrow.parquet as pq

    pf   = pq.ParquetFile(parquet_path, pre_buffer=False)
    tbl  = pf.read(columns=["conversations", "answer_audios"])
    rows = tbl.to_pydict()
    n    = len(rows["conversations"])

    samples = []
    for i in range(n):
        convs    = json.loads(rows["conversations"][i])
        raw      = rows["answer_audios"][i]
        flat     = np.frombuffer(raw, dtype=np.int32).copy()   # (n_frames * 8,)
        n_frames = len(flat) // 8
        codes    = flat.reshape(n_frames, 8).T                 # (8, n_frames)
        samples.append({"convs": convs, "codes": codes})

    print(f"[{time.strftime('%H:%M:%S')}] {split_name}: {n} samples loaded", flush=True)
    return samples


AUDIO_STOP_CODE = 2050   # AUDIO_STOP_TOKEN — appended to the end of each codebook


# ---------------------------------------------------------------------------
# Audio code placement — MiniMind-O staggered/diagonal layout (CORRECT)
# ---------------------------------------------------------------------------
#
# The Talker is architecturally designed for MTP (multi-token prediction) where
# codebook `li` is placed offset by `li` positions from the start:
#
#     position:    asst_start+1  +2   +3   +4   ...
#     codebook 0:  code[0,0]     [0,1] [0,2] [0,3]
#     codebook 1:                code[1,0] [1,1] [1,2]
#     codebook 2:                          code[2,0] [2,1]
#     ...
#
# At position p the Talker reads the embedding fused from codebooks 0..7 and is
# asked to predict the NEXT position's code for each codebook. Because of the
# staggered placement, codebook li only has real targets in the range
# (asst_start+li+1, asst_start+li+T_mimi+1). Outside this range, labels = -100.
#
# The previous align_codes_to_len approach (parallel subsampling) destroyed this
# alignment — it placed all 8 codebooks at the same positions and stretched the
# codes to match text length, which broke the architectural assumption.

def find_asst_start(labels: list[int]) -> int:
    """Find the position where the LAST assistant turn's content begins."""
    last_start = -1
    in_run = False
    for i, l in enumerate(labels):
        if l != -100 and not in_run:
            last_start = i
            in_run = True
        elif l == -100 and in_run:
            in_run = False
    return last_start


def place_codes_diagonal(
    codes: np.ndarray, T: int, asst_start: int,
    audio_pad: int = AUDIO_PAD_CODE, audio_stop: int = AUDIO_STOP_CODE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Place (8, T_mimi) Mimi codes diagonally in a (8, T) sequence.

    Returns:
        audio_ids:    (8, T) int32 — codes at diagonal positions, audio_pad elsewhere
                                     (this is what the Talker reads as input)
        audio_labels: (8, T) int32 — codes at diagonal positions, -100 elsewhere
                                     (this is what the loss masks against)

    Codes are appended with a stop token per codebook before placement.
    """
    n_codebooks, T_mimi = codes.shape
    # Append stop token to each codebook (MiniMind-O convention)
    codes_with_stop = np.concatenate(
        [codes, np.full((n_codebooks, 1), audio_stop, dtype=np.int32)], axis=1,
    )
    T_mimi_total = codes_with_stop.shape[1]

    audio_ids    = np.full((n_codebooks, T), audio_pad, dtype=np.int32)
    audio_labels = np.full((n_codebooks, T), -100,      dtype=np.int32)

    for li in range(n_codebooks):
        for i in range(T_mimi_total):
            pos = asst_start + li + 1 + i
            if pos >= T:
                break
            audio_ids[li, pos]    = codes_with_stop[li, i]
            audio_labels[li, pos] = codes_with_stop[li, i]

    return audio_ids, audio_labels


# ---------------------------------------------------------------------------
# Forward step — text loss + audio (Talker) loss
# ---------------------------------------------------------------------------

def forward_step(model, convs, codes_np, bos_ids, eos_ids, tokenizer,
                 audio_pad_str="<|audio_pad|>"):
    """
    T2A forward: text in → text logits (LM loss) + audio logits (Talker loss).

    codes_np: (8, T_mimi) Mimi codes for the audio output
    """
    token_ids, labels = build_tokens_with_audio(
        convs, 0, tokenizer, bos_ids, eos_ids, audio_pad_str=audio_pad_str,
    )
    T = len(token_ids)   # MAX_SEQ_LEN

    # Find where the assistant turn begins — anchor for diagonal code placement
    asst_start = find_asst_start(labels)
    if asst_start < 0:
        # Defensive: no assistant tokens found — skip this sample
        return mx.array(0.0)

    # Place codes diagonally (MTP layout) — the Talker is designed for this
    audio_ids_np, audio_labels_np = place_codes_diagonal(codes_np, T, asst_start)

    ids          = mx.array([token_ids], dtype=mx.int32)            # (1, T)
    labs         = mx.array([labels],    dtype=mx.int32)            # (1, T)
    audio_ids    = mx.array(audio_ids_np[None],    dtype=mx.int32)  # (1, 8, T)
    audio_labels = mx.array(audio_labels_np[None], dtype=mx.int32)  # (1, 8, T)

    # No audio input (T2A: text only) — no mlx_audio_feats
    out          = model(ids, audio_ids=audio_ids, use_cache=False)
    text_logits  = out["logits"]        # (1, T, vocab_size)
    audio_logits = out["audio_logits"]  # 8 × (1, T, 2112)

    # ── Text loss (assistant turn prediction) ─────────────────────────────────
    tgt_text    = labs[:, 1:].reshape(-1)                    # (T-1,)
    valid_text  = (tgt_text != -100).astype(mx.float32)
    n_valid     = valid_text.sum()
    tgt_safe    = mx.where(tgt_text == -100, mx.zeros_like(tgt_text), tgt_text)
    flat_text   = text_logits[:, :-1, :].reshape(-1, text_logits.shape[-1])
    text_loss   = mx.where(
        n_valid > 0,
        (nn.losses.cross_entropy(flat_text, tgt_safe, reduction="none") * valid_text).sum()
            / (n_valid + 1e-9),
        mx.array(0.0),
    )

    # ── Audio loss (8 codebooks, masked by -100 from diagonal layout) ─────────
    audio_loss = mx.array(0.0)
    n_codebooks = len(audio_logits)
    for k in range(n_codebooks):
        logits_k = audio_logits[k][:, :-1, :]                # (1, T-1, 2112)
        target_k = audio_labels[:, k, 1:].reshape(-1)        # (T-1,)
        valid_k  = (target_k != -100).astype(mx.float32)
        n_k      = valid_k.sum()
        tgt_safe_k = mx.where(target_k == -100, mx.zeros_like(target_k), target_k)
        flat_k   = logits_k.reshape(-1, logits_k.shape[-1])
        ce_k     = nn.losses.cross_entropy(flat_k, tgt_safe_k, reduction="none")
        audio_loss = audio_loss + mx.where(
            n_k > 0,
            (ce_k * valid_k).sum() / (n_k + 1e-9),
            mx.array(0.0),
        )
    audio_loss = audio_loss / n_codebooks

    return text_loss + audio_loss


# ---------------------------------------------------------------------------
# Val loss
# ---------------------------------------------------------------------------

def evaluate_t2a(model, val_samples, tokenizer, bos_ids, eos_ids,
                 max_samples=100, audio_pad_str="<|audio_pad|>"):
    total, n_valid = 0.0, 0
    for sample in val_samples[:max_samples]:
        loss = forward_step(
            model, sample["convs"], sample["codes"], bos_ids, eos_ids, tokenizer,
            audio_pad_str=audio_pad_str,
        )
        v = float(loss)
        if v > 0:
            total  += v
            n_valid += 1
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    omni_cfg, train_cfg = load_omnivox_config(args.config)
    if args.data:
        train_cfg.data_path = args.data
    if args.epochs > 0:
        train_cfg.epochs = args.epochs
    if args.save_dir:
        train_cfg.save_dir = args.save_dir

    log(f"OmniVox T2A  backbone={omni_cfg.backbone_path.split('/')[-1]}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log(f"Loading tokenizer from {omni_cfg.backbone_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(omni_cfg.backbone_path, trust_remote_code=True)

    audio_pad_str = omni_cfg.audio_pad_token_str
    if audio_pad_str.startswith("<|") and audio_pad_str not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [audio_pad_str]})
    omni_cfg.audio_pad_id = tokenizer.convert_tokens_to_ids(audio_pad_str)
    assert omni_cfg.audio_pad_id < omni_cfg.vocab_size
    log(f"  audio_pad: {audio_pad_str!r} → id={omni_cfg.audio_pad_id}")

    tokenizer.bos_token = omni_cfg.bos_token
    tokenizer.eos_token = omni_cfg.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bos_ids = tokenizer(f"{omni_cfg.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_ids = tokenizer(f"{omni_cfg.eos_token}\n",          add_special_tokens=False).input_ids
    log(f"  bos_ids={bos_ids}  eos_ids={eos_ids}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    samples = load_t2a_samples(train_cfg.data_path, "train")
    if not samples:
        log("ERROR: no training samples found")
        sys.exit(1)

    val_path = args.val_data or train_cfg.val_data_path
    if not val_path:
        val_path = train_cfg.data_path.replace("train.parquet", "val.parquet")
    val_samples = []
    if val_path and os.path.exists(val_path):
        val_samples = load_t2a_samples(val_path, "val")
    else:
        log("No val set found — skipping eval")

    # ── Model ──────────────────────────────────────────────────────────────────
    log(f"Loading OmniVox backbone: {omni_cfg.backbone_path} ...")
    model = OmniVox(omni_cfg)
    model.load_backbone()

    # Warm-start Talker from checkpoint if provided (.safetensors preferred, .npz legacy)
    talker_ckpt = args.talker_ckpt
    if talker_ckpt and os.path.exists(talker_ckpt):
        if talker_ckpt.endswith(".safetensors"):
            weights = mx.load(talker_ckpt)              # dict[str, mx.array]
            talker_w = list(weights.items())
        else:
            talker_w = [(k, mx.array(v))
                        for k, v in np.load(talker_ckpt, allow_pickle=False).items()]
        model.talker.load_weights(talker_w, strict=False)
        log(f"  Loaded Talker checkpoint from {talker_ckpt}")

    # Freeze Qwen, train Talker + audio_proj
    model.freeze()
    model.talker.unfreeze()
    model.audio_proj.unfreeze()

    n_train = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Trainable: {n_train/1e6:.2f}M (talker + audio_proj)"
        f"  |  Total: {n_total/1e6:.2f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(learning_rate=train_cfg.learning_rate, weight_decay=0.01)
    loss_fn   = nn.value_and_grad(
        model,
        lambda m, convs, codes, bids, eids, tok:
            forward_step(m, convs, codes, bids, eids, tok,
                         audio_pad_str=audio_pad_str),
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    logger = TrainingLogger(train_cfg.save_dir)

    global_step = 0
    log(f"\nT2A: {train_cfg.epochs} epochs × {len(samples)} samples"
        f"  LR={train_cfg.learning_rate}\n")

    for epoch in range(train_cfg.epochs):
        np.random.shuffle(samples)
        epoch_loss, n_valid = 0.0, 0
        t_epoch = time.time()

        for step, sample in enumerate(samples):
            loss, grads = loss_fn(
                model,
                sample["convs"], sample["codes"],
                bos_ids, eos_ids, tokenizer,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            v            = float(loss)
            epoch_loss  += v
            if v > 0:
                n_valid += 1
            global_step += 1

            logger.log_step(global_step, v, epoch + 1, train_cfg.learning_rate)

            if global_step % train_cfg.log_interval == 0 or step == 0:
                log(f"  ep{epoch+1}  step{step+1}/{len(samples)}"
                    f"  loss={v:.4f}  global={global_step}")

        avg_train = epoch_loss / max(n_valid, 1)
        elapsed   = time.time() - t_epoch
        log(f"Epoch {epoch+1} done — train_loss={avg_train:.4f}  ({elapsed:.0f}s)")

        avg_val = None
        if val_samples:
            log("  Evaluating val set ...")
            avg_val = evaluate_t2a(model, val_samples, tokenizer, bos_ids, eos_ids,
                                    audio_pad_str=audio_pad_str)
            log(f"  val_loss={avg_val:.4f}")

        logger.log_epoch(global_step, epoch + 1, avg_train, avg_val)

        # Checkpoint — Talker is ~250M params, use safetensors (native MLX, no numpy
        # round-trip). Previously np.savez silently failed on the Talker; only
        # audio_proj got saved across 25 epochs. Verify the file exists after save.
        ckpt_dir = os.path.join(train_cfg.save_dir, f"epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)

        talker_path = os.path.join(ckpt_dir, "talker.safetensors")
        talker_flat = dict(tree_flatten(model.talker.parameters()))
        mx.save_safetensors(talker_path, talker_flat)
        if not (os.path.exists(talker_path) and os.path.getsize(talker_path) > 0):
            raise RuntimeError(f"Talker checkpoint failed to save: {talker_path}")

        proj_path = os.path.join(ckpt_dir, "audio_proj.safetensors")
        proj_flat = dict(tree_flatten(model.audio_proj.parameters()))
        mx.save_safetensors(proj_path, proj_flat)

        size_mb = os.path.getsize(talker_path) / (1024 * 1024)
        log(f"  Checkpoint → {ckpt_dir}/  (talker.safetensors {size_mb:.1f}MB + audio_proj.safetensors)")

    logger.close()
    log("\nT2A complete.")
    log("Next: A2A (add Whisper speech input) using the trained Talker checkpoint.")
    log(f"  python scripts/omnivox_phase1a_train.py \\")
    log(f"      --config configs/omnivox/whisper_small_phase1a.yaml \\")
    log(f"      --data   /Users/akashsingh/Documents/exps/fleurs_hindi/train.parquet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OmniVox T2A — Hindi text → Mimi codes")
    p.add_argument("--config",    required=True)
    p.add_argument("--data",      default="", help="T2A train parquet")
    p.add_argument("--val-data",  default="", help="T2A val parquet")
    p.add_argument("--epochs",      default=0, type=int)
    p.add_argument("--save-dir",    default="", dest="save_dir")
    p.add_argument("--talker-ckpt", default="", dest="talker_ckpt",
                   help="Talker .npz checkpoint to warm-start from")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
