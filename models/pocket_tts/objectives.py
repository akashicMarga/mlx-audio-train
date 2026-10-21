"""
Pocket TTS training objective in MLX (port of pocket-tts training/modules/samplers.py
LSD + training/modules/model.py TrainableTTS.forward).

Used to fine-tune a FlowLM (new voices / domain, or a new language with a fresh
text table) — the step before distilling a smaller student from it.

    z      = backbone_z(...)  with CFG dropout of the voice and text prefixes
    loss   = LSD(head(z), noise, latents)  +  eos_loss_weight * EOS BCE

LSD — Lagrangian Self Distillation (arXiv:2505.18825), the objective of the
released models; its head takes two times (s, t) and decodes in one step:

    diag:     t ~ sigmoid(N(0.4, 1)),  x_t = t x + (1-t) e
              || v(t, t, x_t) - (x - e) ||^2
    distill:  s < t,  x_s = s x + (1-s) e,  (v, dv/dt) = jvp(v(s, t, x_s), dt=1)
              x_t' = x_s + (t-s) v,  || v + (t-s) dv/dt - sg_params(v(t, t, x_t')) ||^2
    both terms weighted by a learned per-(s, t) log-variance w_s_t (uncertainty weighting);
    loss = p_equal * diag + (1 - p_equal) / distill_prob * distill  (distill term on a
    random distill_prob fraction of steps, unbiased).
"""

from __future__ import annotations

import random
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map


class MLP(nn.Module):
    """ReLU MLP over the concatenated inputs (training/modules/utils.py MLP), zero-initialised."""

    def __init__(self, in_channels: int, hidden: list[int]):
        super().__init__()
        dims = [in_channels] + hidden
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        for layer in self.layers:
            layer.weight = mx.zeros_like(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, *inputs):
        x = mx.concatenate(inputs, axis=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


def call_without_param_grad(module: nn.Module, fn, *args):
    """fn(*args) with `module`'s parameters held constant (gradients reach the inputs only)."""
    params = module.parameters()
    module.update(tree_map(mx.stop_gradient, params))
    try:
        return fn(*args)
    finally:
        module.update(params)


class LSD(nn.Module):
    num_time_conds = 2

    def __init__(self, p_equal: float = 0.75, lognorm_mean: float = 0.4, lognorm_std: float = 1.0,
                 w_t_dims: int = 32, w_t_depth: int = 3, normalize: bool = True,
                 stopgrad_type: str = "minimal", distill_prob: float = 0.25, seed: int = 0):
        super().__init__()
        assert 0.0 < distill_prob <= 1.0 and stopgrad_type in ("minimal", "classic")
        self.p_equal = p_equal
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.normalize = normalize
        self.stopgrad_type = stopgrad_type
        self.distill_prob = distill_prob
        self._skip_rng = random.Random(seed)
        if normalize:
            self.w_s_t = MLP(2, [w_t_dims] * w_t_depth + [1])

    def sample_t(self, n: int) -> mx.array:
        return mx.sigmoid(mx.random.normal((n, 1)) * self.lognorm_std + self.lognorm_mean)

    def sample_s_t(self, n: int) -> tuple[mx.array, mx.array]:
        logits = mx.random.normal((n, 2)) * self.lognorm_std + self.lognorm_mean
        return mx.sigmoid(logits.min(-1, keepdims=True)), mx.sigmoid(logits.max(-1, keepdims=True))

    def loss(self, head: nn.Module, z: mx.array, e: mx.array, x: mx.array, *, training: bool = True,
             t_diag: mx.array | None = None, s_t: tuple[mx.array, mx.array] | None = None,
             skip: bool | None = None) -> tuple[mx.array, dict]:
        """Per-row loss [N] for head(z, s, t, x_t). e = noise, x = target latents (normalized).

        t_diag / s_t / skip override the random draws (tests)."""
        v_t = partial(head, z)
        D = e.shape[-1]
        metrics = {}

        t = self.sample_t(x.shape[0]) if t_diag is None else t_diag
        x_t = t * x + (1 - t) * e
        flow_diag = mx.sum((v_t(t, t, x_t) - (x - e)) ** 2, axis=-1)
        metrics["flow_diag"] = flow_diag.mean()
        if self.normalize:
            logvar = self.w_s_t(t, t)[..., 0]
            flow_diag = flow_diag * mx.exp(logvar) / D - logvar

        if skip is None:
            skip = training and self.distill_prob < 1.0 and self._skip_rng.random() >= self.distill_prob
        if skip:
            return self.p_equal * flow_diag, metrics

        s, t = self.sample_s_t(x.shape[0]) if s_t is None else s_t
        x_s = s * x + (1 - s) * e
        (vt,), (dvdt,) = mx.jvp(lambda s_, t_, x_: v_t(s_, t_, x_), [s, t, x_s],
                                [mx.zeros_like(s), mx.ones_like(t), mx.zeros_like(x_s)])
        x_t = x_s + (t - s) * vt
        dxdt = vt + (t - s) * dvdt
        if self.stopgrad_type == "minimal":
            # Gradient reaches x_t only: not the head's parameters, not z (f_grad_x_only).
            z_c = mx.stop_gradient(z)
            u_tgt = call_without_param_grad(head, lambda y: head(z_c, t, t, y), x_t)
        else:
            u_tgt = mx.stop_gradient(v_t(t, t, x_t))
        flow_distill = mx.sum((dxdt - u_tgt) ** 2, axis=-1)
        metrics["flow_distill"] = flow_distill.mean()
        if self.normalize:
            logvar = self.w_s_t(s, t)[..., 0]
            flow_distill = flow_distill * mx.exp(logvar) / D - logvar
        w = (1 - self.p_equal) / (self.distill_prob if training else 1.0)
        return self.p_equal * flow_diag + w * flow_distill, metrics


def eos_loss(model, z: mx.array, mask: mx.array) -> mx.array:
    """BCE: 1 on each row's first invalid frame, 0 on valid ones; padding beyond is ignored."""
    is_eos = mx.logical_not(mask)
    is_eos = mx.concatenate([mx.zeros_like(is_eos[:, :1]), is_eos[:, 1:]], axis=1)
    logits = model.out_eos(z)[..., 0].astype(mx.float32)
    per = mx.where(is_eos, nn.softplus(-logits), nn.softplus(logits))
    sm = mx.concatenate([mask[:, :1], mask[:, :-1]], axis=1).astype(mx.float32)
    return (per * sm).sum() / mx.maximum(sm.sum(), 1.0)


def flow_lm_loss(model, lsd: LSD, batch: dict, *, text_dropout: float = 0.2,
                 voice_dropout: float = 0.2, eos_loss_weight: float = 0.1,
                 flow_batch_multiplier: int = 4, training: bool = True,
                 rng: np.random.Generator | None = None) -> tuple[mx.array, dict]:
    """Teacher-forced fine-tuning loss. batch["latents"] must already be normalized."""
    rng = rng or np.random.default_rng()
    lat = batch["latents"]
    B, T, C = lat.shape
    if training:
        keep_voice = rng.random(B) < 1 - voice_dropout
        keep_text = rng.random(B) < 1 - text_dropout
    else:
        keep_voice = keep_text = np.ones(B, dtype=bool)
    z = model.backbone_z(lat, batch["text_tokens"], batch["voice"], batch["voice_lens"],
                         keep_voice, keep_text)
    mask = batch["mask"]
    l_eos = eos_loss(model, z, mask)

    idx = mx.array(np.flatnonzero(np.array(mask).reshape(-1)), dtype=mx.int32)
    sel_z = z.reshape(-1, z.shape[-1])[idx].astype(mx.float32)
    sel_x = lat.reshape(-1, C)[idx].astype(mx.float32)
    if flow_batch_multiplier > 1:
        sel_z = mx.concatenate([sel_z] * flow_batch_multiplier, axis=0)
        sel_x = mx.concatenate([sel_x] * flow_batch_multiplier, axis=0)
    noise = mx.random.normal(sel_x.shape)
    per_row, metrics = lsd.loss(model.flow_net, sel_z, noise, sel_x, training=training)
    l_flow = per_row.mean()
    loss = l_flow + eos_loss_weight * l_eos
    metrics = {k: v for k, v in metrics.items()}
    metrics.update(flow_loss=l_flow, eos_loss=l_eos)
    return loss, metrics


def update_latent_stats(model, latents: mx.array, mask: mx.array, decay: float = 0.999):
    """EMA of the latent normalisation stats (first stats_ema_steps of a fine-tune)."""
    sel = np.array(latents)[np.array(mask)]
    if len(sel) < 2:
        return
    model.emb_mean = decay * model.emb_mean + (1 - decay) * mx.array(sel.mean(0))
    model.emb_std = decay * model.emb_std + (1 - decay) * mx.array(sel.std(0, ddof=1))
