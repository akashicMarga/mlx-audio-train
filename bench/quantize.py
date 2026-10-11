"""
Requantize part of the model to a different bit width (Part 4, §5.8).

`requantize(root, bits, group_size)` replaces every nn.QuantizedLinear under
`root` with one quantized at `bits`: dequantize → quantize. Used for mixed
precision (e.g. the code predictor — 72% of the bytes per frame — at 4-bit
while the Talker stays at 8-bit). Apply after weight-changing levers such as
lora-merge-q8, so the merged weights are what get requantized; the q8→q4 path
rounds twice, adding the q8 error (~0.4%) on top of the q4 error.

Selective variants (`include=`) follow FluidInference's PocketTTS recipe and ML
Drift's 8/4/4 split: lower precision on the MLP body only, keeping attention and
the output heads (which decide the token, like PocketTTS's EOS head) at 8-bit.

Reports each replaced layer's output relative error vs the layer it replaced,
on a random input, before any CER is measured.
"""

import mlx.core as mx
import mlx.nn as nn


def _sites(root, include=None):
    """(parent, key, index, layer) for QuantizedLinear children; `include(path)` filters
    by dotted path relative to root (e.g. "model.layers.3.mlp.down_proj")."""
    out = []
    for name, mod in root.named_modules():
        for key, child in mod.children().items():
            if isinstance(child, nn.QuantizedLinear):
                path = f"{name}.{key}" if name else key
                if include is None or include(path):
                    out.append((mod, key, None, child))
            elif isinstance(child, list):
                # children() returns copies of list children: keep (module, key, index)
                # and write through getattr(module, key) so the replacement lands in the model
                for i, c in enumerate(child):
                    path = f"{name}.{key}.{i}" if name else f"{key}.{i}"
                    if isinstance(c, nn.QuantizedLinear) and (include is None or include(path)):
                        out.append((mod, key, i, c))
    return out


def requantize(root, bits: int, group_size: int = 64, label: str = "", include=None) -> dict:
    sites = _sites(root, include)
    if not sites:
        raise SystemExit(f"no QuantizedLinear layers under {label or root}")
    errs, n_params = [], 0
    for parent, key, idx, q in sites:
        w = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
        out_f, in_f = w.shape
        lin = nn.Linear(in_f, out_f, bias="bias" in q)
        lin.weight = w
        if "bias" in q:
            lin.bias = q.bias
        new = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
        x = mx.random.normal((4, in_f), key=mx.random.key(0)).astype(q.scales.dtype)
        ref = q(x).astype(mx.float32)
        got = new(x).astype(mx.float32)
        errs.append(float(mx.linalg.norm(got - ref) / (mx.linalg.norm(ref) + 1e-12)))
        n_params += out_f * in_f
        if idx is None:
            setattr(parent, key, new)
        else:
            getattr(parent, key)[idx] = new
    mx.eval(root.parameters())
    errs.sort()
    info = {"layers": len(sites), "bits": bits, "group_size": group_size, "params": n_params,
            "rel_err_median": errs[len(errs) // 2], "rel_err_max": errs[-1]}
    print(f"[requantize] {label}: {len(sites)} layers ({n_params / 1e6:.1f}M params) → q{bits} g{group_size}; "
          f"output rel. error vs previous: median {info['rel_err_median']:.2e}, max {info['rel_err_max']:.2e}")
    return info
