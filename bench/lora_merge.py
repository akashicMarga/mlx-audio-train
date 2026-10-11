"""
Merge QLoRA adapters into their quantized base (Part 4, §5.9).

train/lora.py's QLoRALinear has no fuse(): at inference every wrapped projection
runs the quantized base matmul plus two skinny LoRA matmuls, a scale and an add.
These helpers replace each QLoRALinear with a single layer:

  bf16  W' = dequant(W_q) + scale · B·A, kept as bf16 nn.Linear (no extra
        kernels, ~2× the weight bytes of q8 for the merged layers)
  q8    same W', requantized with the base's own group size and bits (same
        bytes and kernels as stock q8; quantization error now lands on W')

The merge is done in float32. `check_error` reports, per merged layer, the
relative error of the merged layer's output vs the unmerged QLoRA output on a
random input — the numerical cost before any CER is measured.
"""

import mlx.core as mx
import mlx.nn as nn

from train.lora import QLoRALinear


def _merged_weight(q: QLoRALinear) -> mx.array:
    base = q.base
    w = mx.dequantize(base.weight, base.scales, base.biases,
                      group_size=base.group_size, bits=base.bits).astype(mx.float32)
    delta = (q.lora_b.astype(mx.float32) @ q.lora_a.astype(mx.float32)) * q.scale
    return w + delta


def _merged_layer(q: QLoRALinear, mode: str) -> nn.Module:
    base = q.base
    w = _merged_weight(q)
    out_f, in_f = w.shape
    has_bias = "bias" in base
    dtype = base.scales.dtype
    lin = nn.Linear(in_f, out_f, bias=has_bias)
    lin.weight = w.astype(dtype)
    if has_bias:
        lin.bias = base.bias
    if mode == "bf16":
        lin.weight = w.astype(mx.bfloat16)
        return lin
    if mode == "q8":
        return nn.QuantizedLinear.from_linear(lin, group_size=base.group_size, bits=base.bits)
    raise ValueError(mode)


def _qlora_sites(root: nn.Module):
    """(parent, key, QLoRALinear) for every wrapped projection under root."""
    sites = []
    for _, mod in root.named_modules():
        for key, child in mod.children().items():
            if isinstance(child, QLoRALinear):
                sites.append((mod, key, child))
    return sites


def merge(model, mode: str, check_error: bool = True) -> dict:
    sites = _qlora_sites(model.talker)
    if not sites:
        raise SystemExit("no QLoRALinear layers found — run with --adapter to merge")
    errs = []
    for parent, key, q in sites:
        new = _merged_layer(q, mode)
        if check_error:
            in_f = q.lora_a.shape[1]
            x = mx.random.normal((4, in_f), key=mx.random.key(0)).astype(q.base.scales.dtype)
            q_enabled = q.enabled
            q.enabled = True
            ref = q(x).astype(mx.float32)
            q.enabled = q_enabled
            got = new(x).astype(mx.float32)
            errs.append(float(mx.linalg.norm(got - ref) / (mx.linalg.norm(ref) + 1e-12)))
        setattr(parent, key, new)
    mx.eval(model.talker.parameters())
    info = {"merged_layers": len(sites), "mode": mode}
    if errs:
        errs.sort()
        info.update(rel_err_median=errs[len(errs) // 2], rel_err_max=errs[-1])
    print(f"[lora-merge] {mode}: merged {len(sites)} layers"
          + (f"; output rel. error vs unmerged: median {info['rel_err_median']:.2e}, "
             f"max {info['rel_err_max']:.2e}" if errs else ""))
    model._bench_merge_info = info
    return info
