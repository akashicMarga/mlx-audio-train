"""
Fused projections (Part 4, §5 — idea from ML Drift's QKV layout fusion).

Each attention layer runs q_proj, k_proj and v_proj as three matmuls over the
same input; each MLP runs gate_proj and up_proj as two. At batch 1 these are
launch-bound matvecs, so concatenating the weights row-wise turns 7 matmul
launches per layer into 4 (qkv, o, gate_up, down). Over a frame that is
28 Talker + 75 code-predictor layer calls.

Fusion is a pure weight-layout change: each output row of a quantized matvec is
computed independently from its own packed weights, scales and biases, so the
fused output should equal the concatenation of the separate outputs.

Requirements: the projections to fuse must be plain nn.QuantizedLinear (or
nn.Linear) with identical input size, group size, bits and mode — i.e. apply
after lora-merge-q8 and any requantization levers. Unmerged QLoRA layers are
refused.

The attention / MLP classes get a patched __call__ that takes the fused path only
for instances carrying `qkv_proj` / `gate_up_proj`; every other instance runs the
original forward.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.qwen3_tts import talker as T

_ORIG = {}


def _fuse(layers):
    first = layers[0]
    if all(isinstance(l, nn.QuantizedLinear) for l in layers):
        keys = ("group_size", "bits", "mode")
        if any(getattr(l, k) != getattr(first, k) for l in layers for k in keys):
            raise ValueError("cannot fuse quantized layers with different group_size/bits/mode")
        in_dims = first.scales.shape[1] * first.group_size
        out_dims = sum(l.scales.shape[0] for l in layers)
        has_bias = "bias" in first
        fused = nn.QuantizedLinear(in_dims, out_dims, bias=has_bias,
                                   group_size=first.group_size, bits=first.bits, mode=first.mode)
        fused.weight = mx.concatenate([l.weight for l in layers], axis=0)
        fused.scales = mx.concatenate([l.scales for l in layers], axis=0)
        if first.biases is not None:
            fused.biases = mx.concatenate([l.biases for l in layers], axis=0)
        if has_bias:
            fused.bias = mx.concatenate([l.bias for l in layers], axis=0)
        return fused
    if all(type(l) is nn.Linear for l in layers):
        out_f = sum(l.weight.shape[0] for l in layers)
        fused = nn.Linear(first.weight.shape[1], out_f, bias="bias" in first)
        fused.weight = mx.concatenate([l.weight for l in layers], axis=0)
        if "bias" in first:
            fused.bias = mx.concatenate([l.bias for l in layers], axis=0)
        return fused
    names = sorted({type(l).__name__ for l in layers})
    raise SystemExit(f"fuse-proj needs merged layers (got {names}); apply lora-merge-q8 first")


def _attn_call(rope):
    def __call__(self, x, position_embeddings, mask=None, cache=None):
        if "qkv_proj" not in self:
            return _ORIG[type(self)](self, x, position_embeddings, mask, cache)
        batch, seq_len, _ = x.shape
        nq = self.num_heads * self.head_dim
        nkv = self.num_kv_heads * self.head_dim
        q, k, v = mx.split(self.qkv_proj(x), [nq, nq + nkv], axis=-1)
        q = self.q_norm(q.reshape(batch, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        cos, sin = position_embeddings
        q, k = rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_len, -1)
        return self.o_proj(out)
    return __call__


def _mlp_call(self, x):
    if "gate_up_proj" not in self:
        return _ORIG[type(self)](self, x)
    gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
    return self.down_proj(T.swiglu(gate, up))


def _patch_classes():
    if _ORIG:
        return
    for cls, fn in ((T.TalkerAttention, _attn_call(T.apply_multimodal_rotary_pos_emb)),
                    (T.CodePredictorAttention, _attn_call(T.apply_rotary_pos_emb)),
                    (T.TalkerMLP, _mlp_call), (T.CodePredictorMLP, _mlp_call)):
        _ORIG[cls] = cls.__call__
        cls.__call__ = fn


def unpatch_classes():
    for cls, fn in _ORIG.items():
        cls.__call__ = fn
    _ORIG.clear()


def fuse(model, check_error: bool = True) -> dict:
    _patch_classes()
    n_attn = n_mlp = 0
    errs = []
    for _, mod in model.talker.named_modules():
        if isinstance(mod, (T.TalkerAttention, T.CodePredictorAttention)) and "qkv_proj" not in mod:
            if check_error:
                x = mx.random.normal((1, 1, mod.hidden_size), key=mx.random.key(0)).astype(mx.bfloat16)
                ref = mx.concatenate([mod.q_proj(x), mod.k_proj(x), mod.v_proj(x)], axis=-1)
            mod.qkv_proj = _fuse([mod.q_proj, mod.k_proj, mod.v_proj])
            if check_error:
                errs.append(float(mx.abs(mod.qkv_proj(x) - ref).max()))
            del mod.q_proj, mod.k_proj, mod.v_proj
            n_attn += 1
        elif isinstance(mod, (T.TalkerMLP, T.CodePredictorMLP)) and "gate_up_proj" not in mod:
            if check_error:
                x = mx.random.normal((1, 1, mod.gate_proj.scales.shape[1] * mod.gate_proj.group_size
                                      if isinstance(mod.gate_proj, nn.QuantizedLinear)
                                      else mod.gate_proj.weight.shape[1]),
                                     key=mx.random.key(1)).astype(mx.bfloat16)
                ref = mx.concatenate([mod.gate_proj(x), mod.up_proj(x)], axis=-1)
            mod.gate_up_proj = _fuse([mod.gate_proj, mod.up_proj])
            if check_error:
                errs.append(float(mx.abs(mod.gate_up_proj(x) - ref).max()))
            del mod.gate_proj, mod.up_proj
            n_mlp += 1
    mx.eval(model.talker.parameters())
    info = {"attention_fused": n_attn, "mlp_fused": n_mlp,
            "max_abs_diff_vs_separate": max(errs) if errs else None}
    print(f"[fuse-proj] fused QKV in {n_attn} attention layers, gate/up in {n_mlp} MLPs; "
          f"max |fused − separate| = {info['max_abs_diff_vs_separate']}")
    return info
