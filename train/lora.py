"""
lora.py — Universal LoRA / QLoRA for any MLX nn.Module.

Supports both:
  - nn.Linear       → LoRALinear       (full-precision base)
  - nn.QuantizedLinear → QLoRALinear   (quantized base, trainable A/B in bf16)

The patching is done recursively in-place on the model tree.
A `scope` prefix can be used to limit patching to a submodule
(e.g. scope="talker" to avoid patching the speech tokenizer).

Usage:
    apply_lora(model, config, scope="talker")    # patch only talker sub-tree
    trainable = get_trainable_params(model)       # LoRA A/B matrices only
    save_adapters(model, "adapters.safetensors")
    load_adapters(model, "adapters.safetensors")
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mxu


# ──────────────────────────────────────────────────────────────────────────────
# LoRALinear — wraps a full-precision nn.Linear
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adapter on top of a frozen nn.Linear."""

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float, dropout: float) -> "LoRALinear":
        out_f, in_f = linear.weight.shape
        lora = cls(in_f, out_f, rank, alpha, dropout, bias="bias" in linear)
        lora.weight = linear.weight
        if "bias" in linear:
            lora.bias = linear.bias
        return lora

    def __init__(self, in_f: int, out_f: int, rank: int, alpha: float, dropout: float, bias: bool):
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.weight = mx.zeros((out_f, in_f))
        if bias:
            self.bias = mx.zeros((out_f,))

        limit = 1.0 / math.sqrt(in_f)
        self.lora_a = mx.random.uniform(-limit, limit, (rank, in_f))
        self.lora_b = mx.zeros((out_f, rank))

    def __call__(self, x: mx.array) -> mx.array:
        x_d = self.dropout(x) if self.dropout else x
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        return y + (x_d @ self.lora_a.T @ self.lora_b.T) * self.scale

    def fuse(self) -> nn.Linear:
        merged = self.weight + (self.lora_b @ self.lora_a) * self.scale
        lin = nn.Linear(*reversed(merged.shape), bias="bias" in self)
        lin.weight = merged
        if "bias" in self:
            lin.bias = self.bias
        return lin


# ──────────────────────────────────────────────────────────────────────────────
# QLoRALinear — wraps a frozen nn.QuantizedLinear (QLoRA)
# ──────────────────────────────────────────────────────────────────────────────

class QLoRALinear(nn.Module):
    """
    Low-rank adapter on top of a frozen QuantizedLinear (QLoRA).

    Keeps the original QuantizedLinear intact (frozen) and adds trainable LoRA delta:
        y = QuantizedLinear(x) + x @ A^T @ B^T * scale
    """

    @classmethod
    def from_quantized(cls, qlinear: nn.QuantizedLinear, rank: int, alpha: float, dropout: float) -> "QLoRALinear":
        # Derive input/output dims from the scales tensor
        # scales shape: [out_features, in_features // group_size]
        out_f = qlinear.scales.shape[0]
        in_f  = qlinear.weight.shape[-1] * (32 // qlinear.bits)

        ql = cls(qlinear, in_f, out_f, rank, alpha, dropout)
        return ql

    def __init__(
        self,
        base:     nn.QuantizedLinear,
        in_f:     int,
        out_f:    int,
        rank:     int,
        alpha:    float,
        dropout:  float,
    ):
        super().__init__()
        self.rank    = rank
        self.scale   = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        # Freeze base so its internal gc_func attrs are excluded from
        # trainable_parameters() traversal — gradients to lora_a/b flow
        # through the input x directly, not through base weights.
        base.freeze()
        self.base    = base

        limit = 1.0 / math.sqrt(in_f)
        self.lora_a = mx.random.uniform(-limit, limit, (rank, in_f))
        self.lora_b = mx.zeros((out_f, rank))

    def __call__(self, x: mx.array) -> mx.array:
        # Frozen quantized forward (handles affine/symmetric correctly)
        y = self.base(x)
        # Trainable LoRA delta
        x_d = self.dropout(x) if self.dropout else x
        return y + (x_d @ self.lora_a.T @ self.lora_b.T) * self.scale


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGETS = {
    "qwen3_tts":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "csm":        ["q_proj", "k_proj", "v_proj", "o_proj"],
    "kokoro":     ["to_q", "to_k", "to_v", "to_out"],
    "chatterbox": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "default":    ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Submodule scope per model type: limit LoRA to this submodule only
DEFAULT_SCOPE = {
    "qwen3_tts":  "talker",          # avoid patching speech_tokenizer
    "csm":        "model",
    "kokoro":     None,              # patch all
    "chatterbox": None,
    "default":    None,
}


@dataclass
class LoRAConfig:
    rank:           int        = 8
    alpha:          float      = 16.0
    dropout:        float      = 0.05
    target_modules: List[str]  = None
    model_type:     str        = "default"
    scope:          str        = None    # override scope (submodule name)
    freeze_base:    bool       = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = DEFAULT_TARGETS.get(self.model_type, DEFAULT_TARGETS["default"])
        if self.scope is None:
            self.scope = DEFAULT_SCOPE.get(self.model_type)


# ──────────────────────────────────────────────────────────────────────────────
# Apply LoRA — recursive in-place patching
# ──────────────────────────────────────────────────────────────────────────────

def apply_lora(model: nn.Module, config: LoRAConfig) -> int:
    """
    Patch matching Linear / QuantizedLinear layers with LoRA adapters.
    Returns number of layers patched.
    """
    targets = set(config.target_modules)

    # Optionally scope to a sub-module
    root = model
    if config.scope:
        if hasattr(model, config.scope):
            root = getattr(model, config.scope)
            print(f"[lora] Scoping to model.{config.scope}")
        else:
            print(f"[lora] Warning: scope '{config.scope}' not found, patching full model")

    patched = _recursive_patch(root, targets, config.rank, config.alpha, config.dropout)

    print(f"[lora] Patched {patched} layers  |  targets: {sorted(targets)}")
    # Note: we do NOT freeze base weights with stop_gradient.
    # Gradients must flow through base weights to reach lora_a.
    # Freezing is enforced by only passing lora_a/lora_b to the optimizer.
    return patched


def _recursive_patch(
    module:   nn.Module,
    targets:  Set[str],
    rank:     int,
    alpha:    float,
    dropout:  float,
) -> int:
    """
    Walk the MLX module tree via module.children() and replace matching
    Linear / QuantizedLinear layers with LoRA wrappers in-place.
    """
    patched = 0

    # MLX stores children via module.children() which returns a nested dict
    children = module.children()
    if not isinstance(children, dict):
        return 0

    for key, val in children.items():
        # Direct match: replace this child with LoRA wrapper
        if key in targets:
            if isinstance(val, nn.QuantizedLinear):
                try:
                    setattr(module, key, QLoRALinear.from_quantized(val, rank, alpha, dropout))
                    patched += 1
                except Exception as e:
                    print(f"[lora] Warning: could not patch QuantizedLinear '{key}': {e}")
            elif isinstance(val, nn.Linear):
                setattr(module, key, LoRALinear.from_linear(val, rank, alpha, dropout))
                patched += 1

        # Recurse into sub-modules
        elif isinstance(val, nn.Module):
            patched += _recursive_patch(val, targets, rank, alpha, dropout)

        # Recurse into lists of modules
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, nn.Module):
                    patched += _recursive_patch(item, targets, rank, alpha, dropout)

    return patched


def _freeze_non_lora(model: nn.Module):
    """
    Freeze all non-LoRA weights using mx.stop_gradient.
    LoRA A/B matrices remain trainable.
    """
    def _walk(module):
        children = module.children() if hasattr(module, 'children') else {}
        if not isinstance(children, dict):
            return
        for key, val in children.items():
            if isinstance(val, (LoRALinear, QLoRALinear)):
                # Freeze base weights; lora_a / lora_b stay trainable
                val.weight = mx.stop_gradient(val.weight)
                if hasattr(val, 'scales') and val.scales is not None:
                    val.scales = mx.stop_gradient(val.scales)
                if hasattr(val, 'biases') and val.biases is not None:
                    val.biases = mx.stop_gradient(val.biases)
            elif isinstance(val, nn.QuantizedLinear):
                val.weight = mx.stop_gradient(val.weight)
                if hasattr(val, 'scales'):
                    val.scales = mx.stop_gradient(val.scales)
            elif isinstance(val, nn.Linear):
                val.weight = mx.stop_gradient(val.weight)
                if "bias" in val:
                    val.bias = mx.stop_gradient(val.bias)
            elif isinstance(val, nn.Module):
                _walk(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, nn.Module):
                        _walk(item)
    _walk(model)


# ──────────────────────────────────────────────────────────────────────────────
# Get trainable parameters (LoRA A/B only)
# ──────────────────────────────────────────────────────────────────────────────

def get_trainable_params(model: nn.Module) -> Dict[str, mx.array]:
    """Return flat dict of only LoRA adapter matrices {path: tensor}."""
    result = {}

    def _walk(module, prefix=""):
        children = module.children() if hasattr(module, 'children') else {}
        if not isinstance(children, dict):
            return
        for key, val in children.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, (LoRALinear, QLoRALinear)):
                result[f"{path}.lora_a"] = val.lora_a
                result[f"{path}.lora_b"] = val.lora_b
            elif isinstance(val, nn.Module):
                _walk(val, path)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, nn.Module):
                        _walk(item, f"{path}.{i}")

    _walk(model)
    return result


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    flat = mxu.tree_flatten(model.trainable_parameters())
    total = sum(v.size for _, v in flat)

    trainable = sum(v.size for v in get_trainable_params(model).values())
    return trainable, total


# ──────────────────────────────────────────────────────────────────────────────
# Save / load adapters
# ──────────────────────────────────────────────────────────────────────────────

def save_adapters(model: nn.Module, path: str) -> None:
    params = get_trainable_params(model)
    if not params:
        print("[lora] Warning: no LoRA params found to save")
        return
    mx.save_safetensors(path, params)
    print(f"[lora] Saved {len(params)} adapter tensors → {path}")


def load_adapters(model: nn.Module, path: str) -> None:
    adapters = mx.load(path)
    trainable = get_trainable_params(model)
    loaded = 0
    for key, val in adapters.items():
        if key in trainable:
            # Walk path to set value
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
            setattr(obj, parts[-1], val)
            loaded += 1
    print(f"[lora] Loaded {loaded}/{len(adapters)} adapter tensors from {path}")
