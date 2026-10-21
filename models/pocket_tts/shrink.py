"""
Seed a smaller FlowLM from a bigger one (numpy port of pocket-tts
training/scripts/shrink_checkpoint.py).

Depth: the student keeps the teacher's bottom and top layers ("ends"); every
tensor outside the backbone is copied unchanged. Kyutai's ablation: top-heavy
seeds reach teacher parity ~20k steps sooner than bottom-heavy ones.

Width: fewer / grouped attention heads (largest q/k/v norm), a narrower residual
stream (channels with the largest LayerNorm gains), a narrower FFN (largest
||linear1 row|| * ||linear2 col||) and a low-rank text table (truncated SVD).
The SliceGPT-style PCA cut (pca_channels upstream) is not ported.

All functions work on flat {name: np.ndarray} dicts without the "flow_lm." prefix.
"""

from __future__ import annotations

import re

import numpy as np

Tensors = dict[str, np.ndarray]

_LAYER = re.compile(r"^transformer\.layers\.(\d+)\.(.*)$")


def select_layers(n_teacher: int, n_student: int) -> list[int]:
    assert 0 < n_student <= n_teacher, f"cannot shrink {n_teacher} -> {n_student}"
    head = (n_student + 1) // 2
    tail = n_student - head
    return list(range(head)) + list(range(n_teacher - tail, n_teacher))


def shrink_depth(state: Tensors, n_student: int) -> tuple[Tensors, list[int]]:
    depth = max((int(m.group(1)) for m in map(_LAYER.match, state) if m), default=-1) + 1
    assert depth > 0, "no transformer layers in state"
    keep = select_layers(depth, n_student)
    remap = {old: new for new, old in enumerate(keep)}
    out: Tensors = {}
    for k, v in state.items():
        m = _LAYER.match(k)
        if not m:
            out[k] = v
        elif int(m.group(1)) in remap:
            out[f"transformer.layers.{remap[int(m.group(1))]}.{m.group(2)}"] = v
    return out, keep


def layer_prefixes(state: Tensors) -> list[str]:
    n = 0
    while f"transformer.layers.{n}.norm1.weight" in state:
        n += 1
    return [f"transformer.layers.{i}" for i in range(n)]


def narrow_attention(src: Tensors, dst: Tensors, head_dim: int) -> Tensors:
    """Fewer query heads and/or grouped kv heads at the source's d_model."""
    out: Tensors = {}
    for layer in layer_prefixes(dst):
        k_in, k_out = f"{layer}.self_attn.in_proj.weight", f"{layer}.self_attn.out_proj.weight"
        if k_in not in src or src[k_in].shape[0] == dst[k_in].shape[0]:
            continue
        w, d = src[k_in], src[k_in].shape[1]
        heads_t, heads_s = src[k_out].shape[1] // head_dim, dst[k_out].shape[1] // head_dim
        kv_t = (w.shape[0] - heads_t * head_dim) // (2 * head_dim)
        kv_s = (dst[k_in].shape[0] - heads_s * head_dim) // (2 * head_dim)
        if min(kv_t, kv_s) < 1 or heads_s > heads_t or heads_t % kv_t or heads_s % kv_s:
            raise ValueError(f"{layer}: cannot seed {heads_s}/{kv_s} heads from {heads_t}/{kv_t}")
        q = w[: heads_t * head_dim].reshape(heads_t, head_dim, d)
        kv = w[heads_t * head_dim:].reshape(2, kv_t, head_dim, d)  # [k|v] interleaved per kv head
        # Torch view(b, t, 2, kv, d) == rows ordered (2, kv, d): k = kv[0], v = kv[1].
        k, v = kv[0], kv[1]
        group = np.arange(heads_t) // (heads_t // kv_t)
        score = (q**2).sum((1, 2)) + (k[group] ** 2).sum((1, 2)) + (v[group] ** 2).sum((1, 2))
        keep = np.sort(np.argsort(-score, kind="stable")[:heads_s])
        k_new = k[group[keep]].reshape(kv_s, heads_s // kv_s, head_dim, d).mean(1)
        v_new = v[group[keep]].reshape(kv_s, heads_s // kv_s, head_dim, d).mean(1)
        out[k_in] = np.concatenate([q[keep].reshape(-1, d), k_new.reshape(-1, d), v_new.reshape(-1, d)])
        o = src[k_out].reshape(src[k_out].shape[0], heads_t, head_dim)[:, keep]
        out[k_out] = o.reshape(o.shape[0], heads_s * head_dim)
    return out


def residual_axes(state: Tensors) -> dict[str, int]:
    """Every tensor with a d_model (residual stream) axis -> that axis."""
    axes = {
        "out_norm.weight": 0, "out_norm.bias": 0, "bos_before_voice": 2,
        "speaker_proj_weight": 0, "input_linear.weight": 0, "conditioner.proj.weight": 0,
        **({} if "conditioner.proj.weight" in state else {"conditioner.embed.weight": 1}),
        **({"z_proj.weight": 1} if "z_proj.weight" in state
           else {"out_eos.weight": 1, "flow_net.cond_embed.weight": 1}),
    }
    for layer in layer_prefixes(state):
        axes.update({
            f"{layer}.norm1.weight": 0, f"{layer}.norm1.bias": 0,
            f"{layer}.norm2.weight": 0, f"{layer}.norm2.bias": 0,
            f"{layer}.self_attn.in_proj.weight": 1, f"{layer}.self_attn.out_proj.weight": 0,
            f"{layer}.linear1.weight": 1, f"{layer}.linear2.weight": 0,
        })
    return {k: a for k, a in axes.items() if k in state}


def slice_channels(src: Tensors, d_student: int) -> Tensors:
    score = np.abs(src["out_norm.weight"]).copy()
    for layer in layer_prefixes(src):
        score += np.abs(src[f"{layer}.norm1.weight"]) + np.abs(src[f"{layer}.norm2.weight"])
    keep = np.sort(np.argsort(-score, kind="stable")[:d_student])
    return {k: np.take(src[k], keep, axis=a) for k, a in residual_axes(src).items()}


def prune_ffn(src: Tensors, dst: Tensors) -> Tensors:
    out: Tensors = {}
    for layer in layer_prefixes(dst):
        k1, k2 = f"{layer}.linear1.weight", f"{layer}.linear2.weight"
        hidden = dst[k1].shape[0]
        if k1 not in src or src[k1].shape[0] <= hidden:
            continue
        l1, l2 = src[k1], src[k2]
        score = np.linalg.norm(l1, axis=1) * np.linalg.norm(l2, axis=0)
        keep = np.sort(np.argsort(-score, kind="stable")[:hidden])
        out[k1], out[k2] = l1[keep], l2[:, keep]
    return out


def low_rank_text_table(src: Tensors, dst: Tensors) -> Tensors:
    if "conditioner.proj.weight" not in dst:
        return {}
    full = src["conditioner.embed.weight"].astype(np.float32)
    rank = dst["conditioner.embed.weight"].shape[1]
    if "conditioner.proj.weight" in src:
        if rank == full.shape[1]:
            return {}
        full = full @ src["conditioner.proj.weight"].astype(np.float32).T
    u, s, vh = np.linalg.svd(full, full_matrices=False)
    return {"conditioner.embed.weight": u[:, :rank] * s[:rank],
            "conditioner.proj.weight": np.ascontiguousarray(vh[:rank].T)}


def shrink_width(src: Tensors, dst: Tensors, head_dim: int) -> Tensors:
    """Tensors of `src` (same depth as `dst`) cut to `dst`'s shapes. Only dst's shapes are read."""
    state = {**src, **narrow_attention(src, dst, head_dim)}
    d_dst = dst["out_norm.weight"].shape[0]
    if state["out_norm.weight"].shape[0] != d_dst:
        state.update(slice_channels(state, d_dst))
    state.update(prune_ffn(state, dst))
    state.update(low_rank_text_table(state, dst))
    return {k: v for k, v in state.items() if k in dst and dst[k].shape == v.shape}


def seed_student(teacher: Tensors, student_shapes: Tensors, head_dim: int) -> tuple[Tensors, list[int]]:
    """Teacher tensors cut to the student's depth and width, plus the kept teacher layers."""
    seeded, kept = shrink_depth(teacher, len(layer_prefixes(student_shapes)))
    return shrink_width(seeded, student_shapes, head_dim), kept
