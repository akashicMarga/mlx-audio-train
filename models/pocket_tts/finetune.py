"""
Build a Pocket TTS FlowLM for fine-tuning (or training from scratch) with the LSD objective.

Modes, mirroring the official configs:
  * same language (training/configs/finetune.yaml): load everything, lr ~2e-5.
  * new language (finetune_language.yaml): override lookup_table.tokenizer_path with
    a tokenizer fitted on the new transcripts and reset the text table
    (reset_text_embedding) — the backbone and heads keep their speech knowledge.
    On Czech this reached 12% WER by 10k steps where scratch was still at 46%.
  * from scratch (scratch.yaml): weights: null — xlformers/DiT init.

freeze_head keeps the sampler head fixed (the official robustness fine-tunes do this
so the latent geometry a distilled student was trained against is preserved).
"""

from __future__ import annotations

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from models.pocket_tts.config import flowlm_args_from_raw, load_raw_config
from models.pocket_tts.model import FlowLM
from models.pocket_tts.objectives import LSD
from models.pocket_tts.weights import load_flow_lm_state, model_state, weights_from

logger = logging.getLogger(__name__)


class Trainable(nn.Module):
    """FlowLM + the objective's own parameters (LSD's w_s_t), optimised together."""

    def __init__(self, flow_lm: FlowLM, objective: LSD):
        super().__init__()
        self.flow_lm = flow_lm
        self.objective = objective


def _trunc_normal(shape, std):
    return mx.random.truncated_normal(-3.0, 3.0, shape) * std


def gaussian_init(module: nn.Module):
    """Backbone init: truncated normal, std 1/sqrt(fan_in); embeddings std 1/sqrt(dim)."""
    for _, m in module.named_modules():
        if isinstance(m, nn.Linear):
            m.weight = _trunc_normal(m.weight.shape, 1 / math.sqrt(m.weight.shape[1]))
            if "bias" in m:
                m.bias = mx.zeros_like(m.bias)
        elif isinstance(m, nn.Embedding):
            m.weight = _trunc_normal(m.weight.shape, 1 / math.sqrt(m.weight.shape[1]))


def dit_init(head: nn.Module):
    """MAR/DiT head init: xavier everywhere, zero adaLN modulations and output."""
    for _, m in head.named_modules():
        if isinstance(m, nn.Linear):
            fan_out, fan_in = m.weight.shape
            a = math.sqrt(6 / (fan_in + fan_out))
            m.weight = mx.random.uniform(-a, a, m.weight.shape)
            m.bias = mx.zeros_like(m.bias)
    for emb in head.time_embed:
        for i in (0, 2):
            emb.mlp[i].weight = mx.random.normal(emb.mlp[i].weight.shape) * 0.02
    zero = [b.adaLN_modulation[1] for b in head.res_blocks]
    zero += [head.final_layer.adaLN_modulation[1], head.final_layer.linear]
    for lin in zero:
        lin.weight = mx.zeros_like(lin.weight)
        lin.bias = mx.zeros_like(lin.bias)


def build_finetune(cfg: dict) -> tuple[Trainable, dict]:
    m = cfg["model"]
    raw = load_raw_config(m["config"], m.get("overrides"))
    args = flowlm_args_from_raw(raw)
    if args.flow_type != "lsd":
        raise ValueError(f"only the LSD objective is ported (config head type {args.flow_type!r})")
    model = FlowLM(args)
    m["weights"] = weights_from(m.get("weights"), raw)  # also the Mimi source of exports
    if m["weights"]:
        state = load_flow_lm_state(m["weights"])
        dropped = []
        if m.get("reset_text_embedding"):
            dropped = [k for k in state if k.startswith("conditioner.")]
            state = {k: v for k, v in state.items() if k not in dropped}
            logger.info("text table starts from scratch: %s", dropped)
        if "speaker_proj_weight" not in state:
            model.pop("speaker_proj_weight")
        shapes = model_state(model)
        bad = [k for k, v in state.items() if k in shapes and shapes[k].shape != v.shape]
        if bad:
            raise ValueError(f"weights do not fit the config: {bad[:5]}")
        model.load_weights(list(state.items()), strict=False)
        missing = sorted(set(shapes) - set(state) - {"speaker_proj_weight"})
        missing = [k for k in missing if not k.endswith("freqs")]
        if any(not k.startswith("conditioner.") for k in missing) or (missing and not dropped):
            raise ValueError(f"pretrained weights miss tensors: {missing[:5]}")
        if dropped:
            gaussian_init(model.conditioner)
    else:
        logger.info("training from scratch")
        gaussian_init(model.transformer)
        gaussian_init(model.input_linear)
        gaussian_init(model.conditioner)
        if "z_proj" in model:
            gaussian_init(model.z_proj)
        model.speaker_proj_weight = _trunc_normal(model.speaker_proj_weight.shape,
                                                  1 / math.sqrt(args.voice_dim))
        dit_init(model.flow_net)
    model.freeze_buffers()
    if m.get("freeze_head"):
        model.flow_net.freeze()
    lsd = LSD(**cfg.get("flow", {}).get("kwargs", {}))
    return Trainable(model, lsd), raw
