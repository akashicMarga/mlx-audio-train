"""
MiniMind-O training script — MLX port of minimind-o/trainer/train_sft_omni.py.

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    # Stage 1: Text → Audio alignment (T2A)
    python scripts/minimind_o_train.py --data_path dataset/train_t2a.parquet --mode all --epochs 15

    # Stage 2: Audio → Audio (A2A, adds speech input)
    python scripts/minimind_o_train.py --data_path dataset/train_a2a.parquet --from_weight sft_omni --mode all

    # Stage 3: vision projection only (I2T)
    python scripts/minimind_o_train.py --data_path dataset/train_i2t.parquet --mode vision_proj

    # Freeze backbone, train audio layers only
    python scripts/minimind_o_train.py --data_path dataset/train_t2a.parquet --freeze_backbone all
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from models.minimind_o import MiniMindOmni, OmniConfig
from models.minimind_o.dataset import OmniDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, base_lr: float) -> float:
    return base_lr * (0.1 + 0.45 * (1 + np.cos(np.pi * step / max(total_steps, 1))))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collate_fn(batch):
    """Collate OmniDataset samples into batched tensors."""
    import torch

    input_ids, text_labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_embs = zip(*batch)
    input_ids_b = torch.stack(input_ids)
    text_labels_b = torch.stack(text_labels)
    audio_labels_b = torch.stack(audio_labels)
    audio_lens_b = torch.tensor(audio_lens, dtype=torch.long)

    valid = [a for a in audio_inputs if a is not None]
    if valid:
        max_t = max(a.size(1) for a in valid)
        padded = [
            a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1)))
            for a in valid
        ]
        audio_inputs_b = torch.cat(padded, dim=0)
    else:
        audio_inputs_b = None

    valid_imgs = [p for p in pixel_values if p is not None]
    if valid_imgs and hasattr(valid_imgs[0], "keys"):
        keys = set.intersection(*[set(d.keys()) for d in valid_imgs])
        pixel_values_b = {k: torch.cat([d[k] for d in valid_imgs], dim=0) for k in keys}
    elif valid_imgs:
        pixel_values_b = torch.cat(valid_imgs, dim=0)
    else:
        pixel_values_b = None

    spk_emb_b = torch.stack(spk_embs)
    return (
        input_ids_b, text_labels_b, audio_labels_b,
        audio_inputs_b, audio_lens_b, pixel_values_b, spk_emb_b,
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(model: MiniMindOmni, batch, config: OmniConfig):
    import torch

    (input_ids_t, text_labels_t, audio_labels_t,
     audio_inputs, audio_lens, pixel_values, spk_emb_t) = batch

    input_ids = mx.array(input_ids_t.numpy())
    text_labels = mx.array(text_labels_t.numpy())
    audio_labels = mx.array(audio_labels_t.numpy())
    spk_emb = mx.array(spk_emb_t.float().numpy())

    kwargs = {}
    if audio_inputs is not None:
        kwargs["audio_inputs"] = audio_inputs
        kwargs["audio_lens"] = audio_lens
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values

    out = model(input_ids, use_cache=False, spk_emb=spk_emb, **kwargs)

    text_logits = out["logits"]
    audio_logits = out["audio_logits"]
    aux_loss = out["aux_loss"]

    text_log_probs = nn.losses.cross_entropy(
        text_logits[:, :-1, :].reshape(-1, text_logits.shape[-1]),
        text_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    valid_text = (text_labels[:, 1:].reshape(-1) != -100).astype(mx.float32)
    text_loss = (text_log_probs * valid_text).sum() / (valid_text.sum() + 1e-9)

    audio_loss = mx.array(0.0)
    for i, al in enumerate(audio_logits):
        targets = audio_labels[:, i, 1:].reshape(-1)
        al_flat = al[:, :-1, :].reshape(-1, al.shape[-1])
        layer_loss = nn.losses.cross_entropy(al_flat, targets, ignore_index=-100)
        valid = (targets != -100).astype(mx.float32)
        stop = (targets == config.audio_stop_token).astype(mx.float32)
        weighted = layer_loss * valid * (1.0 + stop * 9.0)
        denom = valid.sum() + 1e-9
        audio_loss = audio_loss + weighted.sum() / denom
    audio_loss = audio_loss / 8

    total = text_loss + audio_loss + aux_loss
    return total, text_loss, audio_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    log(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    log("Building model ...")
    model = MiniMindOmni(
        config,
        audio_encoder_path=args.audio_encoder_dir,
        vision_model_path=args.vision_dir or None,
    )

    if args.from_weight and args.from_weight != "none":
        pth = os.path.join(args.save_dir, f"{args.from_weight}_{config.hidden_size}.pth")
        if os.path.exists(pth):
            model.load_weights_from_pt(pth, strict=False)
            log(f"Loaded weights from {pth}")
        else:
            log(f"[warn] Weight file not found: {pth}")

    if args.freeze_backbone == "all":
        model.model.freeze()
        log("Frozen: entire thinker backbone")
    elif args.freeze_backbone == "last1":
        model.model.freeze()
        model.model.layers[-1].unfreeze()
        model.model.norm.unfreeze()
        log("Frozen: all thinker layers except last 1")

    if args.mode == "audio_proj":
        model.freeze()
        model.audio_proj.unfreeze()
        log("Training: audio_proj only")
    elif args.mode == "vision_proj":
        model.freeze()
        model.vision_proj.unfreeze()
        log("Training: vision_proj only")

    n_params = sum(
        p.size for name, p in tree_flatten(model.trainable_parameters())
        if "audio_encoder" not in name and "vision_encoder" not in name
    )
    log(f"Trainable params: {n_params/1e6:.2f}M")

    log(f"Loading dataset from {args.data_path} ...")
    processor = model.audio_processor if model.audio_processor is not None else None

    dataset = OmniDataset(
        args.data_path, tokenizer,
        audio_processor=processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=config.image_token_len,
    )
    log(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False,
    )

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, batch: compute_loss(m, batch, config)[0])

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    total_steps = args.epochs * len(loader)

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            lr = get_lr(global_step, total_steps, args.learning_rate)
            optimizer.learning_rate = lr

            loss_val, grads = loss_and_grad_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val)

            if global_step % args.log_interval == 0:
                _, text_l, audio_l = compute_loss(model, batch, config)
                mx.eval(text_l, audio_l)
                log(
                    f"Epoch [{epoch+1}/{args.epochs}] step {step}/{len(loader)} | "
                    f"loss={float(loss_val):.4f} text={float(text_l):.4f} audio={float(audio_l):.4f} "
                    f"lr={lr:.6f}"
                )

            if global_step % args.save_interval == 0 or step == len(loader):
                _save(model, config, args)

            global_step += 1

    _save(model, config, args)
    log("Training complete.")


def _save(model: MiniMindOmni, config: OmniConfig, args) -> None:
    import torch

    moe_sfx = "_moe" if config.use_moe else ""
    out_path = os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}{moe_sfx}.pth")
    weights = {}
    for name, val in tree_flatten(model.parameters()):
        if "audio_encoder" in name or "vision_encoder" in name:
            continue
        weights[name] = torch.tensor(np.array(val).astype(np.float16))
    torch.save(weights, out_path)
    log(f"Saved checkpoint: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MiniMind-O training (MLX)")
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument("--save_weight", default="sft_omni", type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", default="dataset/train_t2a_mini.parquet", type=str)
    parser.add_argument("--audio_encoder_dir", default="./model/SenseVoiceSmall", type=str)
    parser.add_argument("--vision_dir", default=None, type=str)
    parser.add_argument("--tokenizer_path", default="./model", type=str)
    parser.add_argument("--from_weight", default="llm", type=str)
    parser.add_argument(
        "--freeze_backbone", default="none", type=str, choices=["none", "all", "last1"]
    )
    parser.add_argument(
        "--mode", default="all", type=str, choices=["all", "audio_proj", "vision_proj"]
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
