"""
training.py — PersonaPlex-specific LoRA and parameter-management utilities.

Knows about PersonaPlex architecture:
  - LoRA targets: transformer attention in_proj / out_proj only
  - Fully trainable: depformer, audio_embs, text_linear, out_norm
  - Frozen: everything else (main transformer weights ~4B params)

This lives here (not in train/lora.py) because it encodes structural knowledge
of PersonaPlex's model layout.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .lm import Lm


# ─────────────────────────────────────────────────────────────────────────────
# LoRALinear
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Linear layer with low-rank adaptation.

    Wraps an existing nn.Linear (or nn.QuantizedLinear for QLoRA), freezing the
    original weight and adding trainable low-rank matrices A and B such that:
        output = x @ (W + scale * B @ A)^T

    Output is cast back to input dtype (important for bfloat16 training).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features  = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        if isinstance(base_linear, nn.QuantizedLinear):
            in_features = in_features * 32 // base_linear.bits

        self.linear = base_linear

        # Kaiming uniform init for lora_a, zeros for lora_b
        limit       = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(-limit, limit, (rank, in_features))
        self.lora_b = mx.zeros((out_features, rank))
        self.scale  = alpha / rank

        self.dropout: nn.Dropout | None = nn.Dropout(p=dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        z = self.dropout(x) if self.dropout is not None else x
        lora_out = (z @ self.lora_a.T @ self.lora_b.T) * self.scale
        return (y + lora_out).astype(x.dtype)

    def fuse(self) -> nn.Linear:
        """Merge LoRA delta into base weight and return a plain nn.Linear."""
        weight  = self.linear.weight
        delta   = (self.lora_b @ self.lora_a) * self.scale
        merged  = weight + delta
        fused   = nn.Linear(*reversed(merged.shape), bias="bias" in self.linear)
        fused.weight = merged
        if "bias" in self.linear:
            fused.bias = self.linear.bias
        return fused

    @classmethod
    def from_base(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        in_features  = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        if isinstance(linear, nn.QuantizedLinear):
            in_features = in_features * 32 // linear.bits
        lora        = cls.__new__(cls)
        nn.Module.__init__(lora)
        lora.linear = linear
        limit       = 1.0 / math.sqrt(in_features)
        lora.lora_a = mx.random.uniform(-limit, limit, (rank, in_features))
        lora.lora_b = mx.zeros((out_features, rank))
        lora.scale  = alpha / rank
        lora.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        return lora


# ─────────────────────────────────────────────────────────────────────────────
# Gradient checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def grad_checkpoint(layer_class: type) -> None:
    """Enable gradient checkpointing on a transformer layer class.

    Monkeypatches the class's __call__ to use mx.checkpoint, which recomputes
    activations during the backward pass instead of storing them. Reduces peak
    activation memory by ~30-40% at the cost of ~30% more compute.
    """
    original_call = layer_class.__call__

    def checkpointed_call(self, *args, **kwargs):
        def inner(params, *args, **kwargs):
            self.update(params)
            return original_call(self, *args, **kwargs)
        return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

    layer_class.__call__ = checkpointed_call


# ─────────────────────────────────────────────────────────────────────────────
# Apply LoRA to PersonaPlex transformer
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora_to_transformer(
    model: Lm,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> None:
    """Apply LoRA adapters to the main transformer's attention layers and depformer bridge.

    Modifies the model in-place:
      - Main transformer: replaces self_attn in_proj / out_proj with LoRA-wrapped versions
      - Depformer: replaces each slice's linear_in with a LoRA-wrapped version

    The depformer linear_in is the only layer that reads main transformer output.
    When LoRA shifts the transformer_out distribution, a frozen linear_in produces
    OOD inputs to the depformer → garbled audio. LoRA on linear_in lets the bridge
    adapt without touching any depformer internal parameters.
    """
    for layer in model.transformer.layers:
        attn = layer.self_attn

        if hasattr(attn, "in_proj") and isinstance(attn.in_proj, (nn.Linear, nn.QuantizedLinear)):
            attn.in_proj = LoRALinear.from_base(attn.in_proj, rank=rank, alpha=alpha, dropout=dropout)

        if hasattr(attn, "out_proj") and isinstance(attn.out_proj, (nn.Linear, nn.QuantizedLinear)):
            attn.out_proj = LoRALinear.from_base(attn.out_proj, rank=rank, alpha=alpha, dropout=dropout)

    # LoRA on depformer bridge layers (linear_in reads transformer_out)
    for slice_mod in model.depformer.slices:
        if hasattr(slice_mod, "linear_in") and isinstance(slice_mod.linear_in, nn.Linear):
            slice_mod.linear_in = LoRALinear.from_base(
                slice_mod.linear_in, rank=rank, alpha=alpha, dropout=dropout
            )


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def freeze_non_trainable(model: Lm, train_depformer: bool = True) -> tuple[int, int]:
    """Freeze all parameters except the trainable ones.

    Trainable:
      - LoRA adapters (lora_a, lora_b) in main transformer attention
      - LoRA adapters (lora_a, lora_b) in depformer.slices[i].linear_in (always)
      - Full depformer (all parameters) — only when train_depformer=True
      - Audio embeddings (audio_embs)
      - Text linear output head (text_linear)
      - Output norm (out_norm)

    Frozen:
      - Main transformer weights (except LoRA adapters above)
      - Text embedding (text_emb)
      - Depformer internal weights (when train_depformer=False) — only linear_in LoRA adapters train

    The depformer linear_in LoRA adapters are ALWAYS trainable (even when train_depformer=False)
    because they bridge transformer_out → depformer. Without them, LoRA-shifted transformer_out
    causes the frozen depformer to produce garbled audio during autoregressive inference.

    Returns (num_trainable_params, num_frozen_params).
    """
    def _named_params(module):
        def _flatten(d, prefix):
            items = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, mx.array):
                    items.append((key, v))
                elif isinstance(v, dict):
                    items.extend(_flatten(v, key))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        ikey = f"{key}.{i}"
                        if isinstance(item, mx.array):
                            items.append((ikey, item))
                        elif isinstance(item, dict):
                            items.extend(_flatten(item, ikey))
            return items
        return _flatten(module.parameters(), "")

    num_trainable = 0
    num_frozen    = 0
    for name, param in _named_params(model):
        # lora_a/lora_b catches both transformer attention LoRA and depformer linear_in LoRA
        is_trainable = (
            "lora_a" in name or "lora_b" in name
            or (train_depformer and name.startswith("depformer."))
            or name.startswith("audio_embs.")
            or name.startswith("text_linear.")
            or name.startswith("out_norm.")
        )
        size = param.size
        if is_trainable:
            num_trainable += size
        else:
            num_frozen += size

    # Apply freeze/unfreeze
    model.freeze()
    if train_depformer:
        model.depformer.unfreeze()
    for emb in model.audio_embs:
        emb.unfreeze()
    model.text_linear.unfreeze()
    model.out_norm.unfreeze()
    # For LoRALinear layers in main transformer attention: unfreeze the whole module,
    # then re-freeze the base linear weight so only lora_a / lora_b remain trainable.
    for layer in model.transformer.layers:
        attn = layer.self_attn
        if hasattr(attn, "in_proj") and hasattr(attn.in_proj, "lora_a"):
            attn.in_proj.unfreeze()
            attn.in_proj.linear.freeze()   # keep base transformer weight frozen
        if hasattr(attn, "out_proj") and hasattr(attn.out_proj, "lora_a"):
            attn.out_proj.unfreeze()
            attn.out_proj.linear.freeze()  # keep base transformer weight frozen
    # Always unfreeze depformer linear_in LoRA adapters (bridge from transformer_out).
    # These are trainable even when train_depformer=False — they fix the distribution
    # gap caused by LoRA-shifted transformer_out without touching depformer internals.
    for slice_mod in model.depformer.slices:
        if hasattr(slice_mod, "linear_in") and hasattr(slice_mod.linear_in, "lora_a"):
            slice_mod.linear_in.unfreeze()
            slice_mod.linear_in.linear.freeze()  # keep base bridge weight frozen

    return num_trainable, num_frozen
