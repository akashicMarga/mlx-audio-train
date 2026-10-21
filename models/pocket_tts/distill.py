"""
Pocket TTS depth/width distillation in MLX (port of the official recipe,
kyutai-labs/pocket-tts training/configs/depth_distill.yaml + PRs #364/#367).

    teacher (frozen, e.g. 24 layers)            student (e.g. 6 layers, maybe narrower)
    z_c  = teacher.backbone_z(full cond)        z_s = student.backbone_z(full cond)
    z_0  = teacher.backbone_z(no voice/text)
    target = z_0 + cfg * (z_c - z_0)            loss = masked_mse(z_s, target)

* Only the backbone (transformer, conditioning, input/output projections) learns.
  The sampler head (flow_net) and EOS head are copied from the teacher and frozen,
  so the student must produce z in the teacher's space -> the z width must match.
* cfg > 1 bakes classifier-free guidance into the student: it generates at
  cfg 1 (one backbone pass per frame) with the quality of guided sampling.
* The student is seeded from the teacher (bottom+top layers, strongest channels),
  which converges far faster than a random init.
* The loss covers valid frames plus the first padded one (the EOS frame).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from models.pocket_tts.config import flowlm_args_from_raw, load_raw_config
from models.pocket_tts.model import FlowLM
from models.pocket_tts.shrink import seed_student, shrink_width
from models.pocket_tts.weights import build_flow_lm, load_flow_lm_state, model_state

logger = logging.getLogger(__name__)


def to_numpy(state: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {k: np.array(v.astype(mx.float32)) for k, v in state.items()}


def shapes_of(model: FlowLM) -> dict[str, np.ndarray]:
    return {k: np.empty(v.shape, dtype=np.float32) for k, v in model_state(model).items()}


def build_student(teacher: FlowLM, student_config: str, student_overrides: dict | None = None,
                  init_from: str | None = None) -> tuple[FlowLM, dict, list[int]]:
    """Student FlowLM seeded from the teacher, heads frozen. Returns (student, raw config, kept layers)."""
    raw = load_raw_config(student_config, student_overrides)
    args = flowlm_args_from_raw(raw)
    student = FlowLM(args)
    if "speaker_proj_weight" not in teacher:
        student.pop("speaker_proj_weight")

    if student.args.z_width != teacher.args.z_width:
        raise ValueError(
            "distillation needs the student's heads to read the teacher's width: give both the "
            f"same z_dim (student {student.args.z_width}, teacher {teacher.args.z_width}). "
            "A teacher without z_dim reads d_model, so the student must keep that d_model "
            "(or fold a bottleneck into the teacher first: fold_z_bottleneck)."
        )
    if student.args.head_dim != teacher.args.head_dim:
        raise ValueError(
            f"attention head width must match the teacher's ({teacher.args.head_dim}): "
            "narrow attention with fewer heads (num_heads, attn_dim) and/or num_kv_heads"
        )
    if student.args.ldim != teacher.args.ldim or student.args.n_bins != teacher.args.n_bins:
        raise ValueError("student and teacher must share the latent space and tokenizer")

    dst = shapes_of(student)
    seeded, kept = seed_student(to_numpy(model_state(teacher)), dst, args.head_dim)
    if init_from:
        src = {k: v for k, v in to_numpy(load_flow_lm_state(init_from)).items()
               if not k.startswith(FlowLM.HEAD_PREFIXES)}
        seeded.update(shrink_width(src, dst, args.head_dim))
        logger.info("student backbone seeded from %s", init_from)
    missing = sorted(set(dst) - set(seeded))
    student.load_weights([(k, mx.array(v)) for k, v in seeded.items()], strict=False)
    # The heads must be the teacher's exactly; anything else unseeded keeps its init.
    head_missing = [k for k in missing if k.startswith(FlowLM.HEAD_PREFIXES)]
    if head_missing:
        raise ValueError(f"teacher heads do not fit the student: {head_missing}")
    if missing:
        logger.warning("student tensors left at random init: %s", missing)

    student.freeze_buffers()
    student.flow_net.freeze()
    student.out_eos.freeze()
    return student, raw, kept


def build_distillation(cfg: dict):
    """(teacher, student, student raw config, kept layers) from the YAML sections."""
    t, s = cfg["teacher"], cfg["student"]
    dtype = getattr(mx, t.get("dtype", "float32"))
    teacher, _ = build_flow_lm(t["config"], t.get("overrides"), t["weights"], dtype=dtype)
    teacher.freeze()
    student, raw, kept = build_student(teacher, s["config"], s.get("overrides"), s.get("init_from"))
    return teacher, student, raw, kept


def teacher_target(teacher: FlowLM, batch: dict, cfg_coef: float) -> mx.array:
    """Guidance-combined teacher z for the batch (always fully conditioned, no dropout)."""
    B = batch["latents"].shape[0]
    ones, zeros = np.ones(B, dtype=bool), np.zeros(B, dtype=bool)
    args = (batch["latents"], batch["text_tokens"], batch["voice"], batch["voice_lens"])
    z = teacher.backbone_z(*args, keep_voice=ones, keep_text=ones)
    if cfg_coef != 1.0:
        z_null = teacher.backbone_z(*args, keep_voice=zeros, keep_text=zeros)
        z = z_null + cfg_coef * (z - z_null)
    return mx.stop_gradient(z.astype(mx.float32))


def shifted_mask(mask: mx.array) -> mx.array:
    """Valid frames plus the first invalid one (where EOS fires)."""
    return mx.concatenate([mask[:, :1], mask[:, :-1]], axis=1).astype(mx.float32)


def distill_loss(student: FlowLM, batch: dict, target: mx.array) -> mx.array:
    B = batch["latents"].shape[0]
    ones = np.ones(B, dtype=bool)
    z = student.backbone_z(batch["latents"], batch["text_tokens"], batch["voice"],
                           batch["voice_lens"], keep_voice=ones, keep_text=ones)
    m = shifted_mask(batch["mask"])
    err = mx.mean((z.astype(mx.float32) - target) ** 2, axis=-1)
    return (err * m).sum() / mx.maximum(m.sum(), 1.0)


def trainable_count(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


# ── Experimental: give a bottleneck-free teacher a z_dim ──────────────────────

def fold_z_bottleneck(teacher: FlowLM, z_samples: mx.array, z_dim: int,
                      eos_weight: float | None = None) -> tuple[dict[str, mx.array], float]:
    """Fold a rank-`z_dim` bottleneck into a teacher whose heads read d_model.

    The heads only see z through W = [flow_net.cond_embed.weight; out_eos.weight].
    Reduced-rank regression gives the best rank-r map from z to the head inputs
    y = W z under the data's second moment S = E[z z^T]:
        W S^1/2 = A diag(s) B^T,   y ~= A_r (A_r^T W z)
    so z_proj = A_r^T W (r x d), cond_embed' = A_r[:C], out_eos' = A_r[C:] (biases kept).

    Returns the new tensors (to load into a FlowLM built with z_dim=r) and the
    relative error of the head inputs on `z_samples` [N, d_model]. This changes
    the teacher, so measure its WER/quality before distilling from it.
    """
    W_c = np.array(teacher.flow_net.cond_embed.weight.astype(mx.float32))   # [C, d]
    W_e = np.array(teacher.out_eos.weight.astype(mx.float32))               # [1, d]
    # Weight the EOS row so its single logit is not drowned by C conditioning dims.
    eos_weight = eos_weight if eos_weight is not None else float(np.sqrt(W_c.shape[0]))
    W = np.concatenate([W_c, eos_weight * W_e], axis=0)
    Z = np.array(z_samples.astype(mx.float32)).astype(np.float64)
    S = Z.T @ Z / Z.shape[0]
    evals, evecs = np.linalg.eigh(S)
    S_half = (evecs * np.sqrt(np.clip(evals, 0, None))) @ evecs.T
    A, _, _ = np.linalg.svd(W @ S_half, full_matrices=False)
    A_r = A[:, :z_dim]
    z_proj = A_r.T @ W                                                      # [r, d]
    Y = Z @ W.T
    Y_hat = (Z @ z_proj.T) @ A_r.T
    rel_err = float(np.linalg.norm(Y - Y_hat) / max(np.linalg.norm(Y), 1e-12))
    new = {
        "z_proj.weight": mx.array(z_proj.astype(np.float32)),
        "flow_net.cond_embed.weight": mx.array(A_r[: W_c.shape[0]].astype(np.float32)),
        "out_eos.weight": mx.array((A_r[W_c.shape[0]:] / eos_weight).astype(np.float32)),
    }
    return new, rel_err
