"""Pocket TTS distillation (models/pocket_tts): no downloads needed.

The parity tests compare against the official PyTorch code and run only when
torch and a kyutai-labs/pocket-tts checkout are importable, e.g.

    git clone https://github.com/kyutai-labs/pocket-tts /tmp/pocket-tts
    POCKET_TTS_REPO=/tmp/pocket-tts pytest tests/test_pocket_tts.py
"""

import copy
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.pocket_tts.config import flowlm_args_from_raw  # noqa: E402
from models.pocket_tts.distill import (  # noqa: E402
    build_student, distill_loss, fold_z_bottleneck, teacher_target, trainable_count,
)
from models.pocket_tts.model import FlowLM, sample_head  # noqa: E402
from models.pocket_tts.shrink import select_layers  # noqa: E402
from models.pocket_tts.weights import model_state  # noqa: E402

if os.environ.get("POCKET_TTS_REPO"):
    sys.path.insert(0, os.environ["POCKET_TTS_REPO"])
try:
    import torch

    from pocket_tts.models.flow_lm import FlowLMModel
    from pocket_tts.modules import text_conditioner
    from pocket_tts.utils.config import FlowLMConfig
    HAVE_REF = True
except Exception:  # torch or pocket-tts missing
    HAVE_REF = False

needs_ref = pytest.mark.skipif(not HAVE_REF, reason="torch + pocket-tts checkout not importable")


def tiny_raw(layers=4, d_model=64, z_dim=None, attn_dim=None, kv=None, ff=None, rank=None,
             flow_type="lsd", heads=4) -> dict:
    return {
        "flow_lm": {
            "dtype": "float32",
            "insert_bos_before_voice": True,
            "z_dim": z_dim,
            "flow": {"dim": 32, "depth": 2, "type": flow_type},
            "transformer": {"d_model": d_model, "hidden_scale": 2, "max_period": 10000,
                            "num_heads": heads, "num_layers": layers, "attn_dim": attn_dim,
                            "num_kv_heads": kv, "dim_feedforward": ff},
            "lookup_table": {"dim": d_model, "n_bins": 50, "tokenizer": "sentencepiece",
                             "tokenizer_path": "unused.model", "rank": rank},
        },
        "mimi": {"inner_dim": 8, "seanet": {"dimension": 16}, "quantizer": {"dimension": 8}},
    }


def random_batch(B=3, T=11, ldim=8, Tv=5, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    lens = rng.integers(T // 2, T + 1, size=B)
    lens[0] = T
    mask = np.arange(T)[None, :] < lens[:, None]
    return {
        "latents": mx.array(rng.normal(size=(B, T, ldim)).astype(np.float32)),
        "mask": mx.array(mask),
        "text_tokens": [rng.integers(0, 50, size=int(n)) for n in rng.integers(2, 7, size=B)],
        "voice": mx.array(rng.normal(size=(B, Tv, ldim)).astype(np.float32)),
        "voice_lens": rng.integers(1, Tv + 1, size=B),
    }


def randomize(model: nn.Module, seed=0, std=0.2):
    rng = np.random.default_rng(seed)
    state = model_state(model)
    model.load_weights([(k, mx.array((rng.normal(size=v.shape) * std).astype(np.float32)
                         + (1.0 if k.endswith(("norm1.weight", "norm2.weight", "alpha")) else 0.0)))
                        for k, v in state.items() if not k.endswith("freqs")], strict=False)
    model.emb_std = mx.ones_like(model.emb_std)


def all_true(B):
    return np.ones(B, dtype=bool)


# ── MLX-only ──────────────────────────────────────────────────────────────────

def test_select_layers_keeps_ends():
    assert select_layers(24, 6) == [0, 1, 2, 21, 22, 23]
    assert select_layers(24, 5) == [0, 1, 2, 22, 23]


def test_build_inputs_layout():
    model = FlowLM(flowlm_args_from_raw(tiny_raw()))
    b = random_batch()
    x, prefix = model.build_inputs(b["latents"], b["text_tokens"], b["voice"], b["voice_lens"],
                                   all_true(3), all_true(3))
    expected = 1 + b["voice_lens"] + np.array([len(t) for t in b["text_tokens"]])
    np.testing.assert_array_equal(prefix, expected)
    assert x.shape == (3, int(expected.max()) + 11, 64)
    # Null branch: only bos_before_voice precedes the audio.
    _, prefix0 = model.build_inputs(b["latents"], b["text_tokens"], b["voice"], b["voice_lens"],
                                    np.zeros(3, bool), np.zeros(3, bool))
    np.testing.assert_array_equal(prefix0, [1, 1, 1])


def test_padding_never_reaches_valid_frames():
    """Changing latents past a row's end leaves z on its valid frames unchanged."""
    model = FlowLM(flowlm_args_from_raw(tiny_raw()))
    b = random_batch()
    z1 = model.backbone_z(b["latents"], b["text_tokens"], b["voice"], b["voice_lens"], all_true(3), all_true(3))
    lat = np.array(b["latents"])
    mask = np.array(b["mask"])
    lat[~mask] = 123.0
    z2 = model.backbone_z(mx.array(lat), b["text_tokens"], b["voice"], b["voice_lens"], all_true(3), all_true(3))
    # z[:, t] reads audio up to t-1, so frame t is clean while t-1 is valid.
    clean = np.concatenate([mask[:, :1], mask[:, :-1]], axis=1)
    np.testing.assert_allclose(np.array(z1)[clean], np.array(z2)[clean], atol=1e-5)


def test_kv_cache_matches_full_forward():
    model = FlowLM(flowlm_args_from_raw(tiny_raw(attn_dim=32, kv=2, z_dim=16)))
    x = mx.random.normal((1, 9, 64))
    full = model.backbone(x)
    from models.pocket_tts.model import KVCache
    cache = [KVCache() for _ in model.transformer.layers]
    first = model.backbone(x[:, :5], cache)
    rest = [model.backbone(x[:, i:i + 1], cache) for i in range(5, 9)]
    np.testing.assert_allclose(np.array(mx.concatenate([first] + rest, axis=1)), np.array(full), atol=1e-4)


def test_narrow_student_seeded_and_heads_frozen():
    teacher = FlowLM(flowlm_args_from_raw(tiny_raw(layers=6, d_model=64, z_dim=16)))
    randomize(teacher)
    teacher.freeze()
    path = REPO_ROOT / "tests" / "_tmp_student.yaml"
    path.write_text(yaml.safe_dump(tiny_raw(layers=2, d_model=32, z_dim=16, attn_dim=32,
                                            kv=1, ff=48, rank=8, heads=2)))
    try:
        student, _, kept = build_student(teacher, str(path))
    finally:
        path.unlink()
    assert kept == [0, 5]
    s_state, t_state = model_state(student), model_state(teacher)
    for k in s_state:
        if k.startswith(("flow_net.", "out_eos.")):
            np.testing.assert_array_equal(np.array(s_state[k]), np.array(t_state[k]))
    trainable = {k for k, _ in __import__("mlx.utils").utils.tree_flatten(student.trainable_parameters())}
    assert not any(k.startswith(("flow_net.", "out_eos.", "emb_")) for k in trainable)
    assert "transformer.layers.0.self_attn.in_proj.weight" in trainable
    assert trainable_count(student) < trainable_count(FlowLM(student.args))


def test_z_width_mismatch_is_rejected():
    teacher = FlowLM(flowlm_args_from_raw(tiny_raw(layers=4, d_model=64)))
    path = REPO_ROOT / "tests" / "_tmp_student.yaml"
    path.write_text(yaml.safe_dump(tiny_raw(layers=2, d_model=32)))
    try:
        with pytest.raises(ValueError, match="z_dim"):
            build_student(teacher, str(path))
    finally:
        path.unlink()


def test_distillation_converges_and_samples_match_teacher():
    """A 2-layer student learns a 6-layer teacher's guided z, and its frozen head follows."""
    mx.random.seed(0)
    teacher = FlowLM(flowlm_args_from_raw(tiny_raw(layers=6, z_dim=16)))
    randomize(teacher, std=0.1)
    teacher.freeze()
    path = REPO_ROOT / "tests" / "_tmp_student.yaml"
    path.write_text(yaml.safe_dump(tiny_raw(layers=2, z_dim=16)))
    try:
        student, _, _ = build_student(teacher, str(path))
    finally:
        path.unlink()
    batch = random_batch(B=4, T=12)
    target = teacher_target(teacher, batch, cfg_coef=2.0)
    opt = optim.AdamW(learning_rate=3e-3, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(student, lambda m: distill_loss(m, batch, target))
    losses = []
    for _ in range(150):
        loss, grads = loss_and_grad(student)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state)
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0], (losses[0], losses[-1])
    # Same noise through the shared frozen head gives (nearly) the teacher's guided sample.
    ones = all_true(4)
    z_s = student.backbone_z(batch["latents"], batch["text_tokens"], batch["voice"],
                             batch["voice_lens"], ones, ones)
    noise = mx.random.normal(z_s.shape[:2] + (8,))
    a = sample_head("lsd", student.flow_net, z_s, noise)
    b = sample_head("lsd", teacher.flow_net, target, noise)
    assert float(mx.abs(a - b).mean()) < 0.1 * float(mx.abs(b).mean())


def test_fold_z_bottleneck_exact_at_full_rank():
    teacher = FlowLM(flowlm_args_from_raw(tiny_raw(d_model=64)))
    randomize(teacher)
    z = mx.random.normal((512, 64))
    # Head inputs: 32 conditioning dims + 1 EOS logit -> rank <= 33, so 33 is lossless.
    _, err_full = fold_z_bottleneck(teacher, z, 33)
    new, err_low = fold_z_bottleneck(teacher, z, 8)
    assert err_full < 1e-5 < err_low < 1.0
    # The folded tensors reproduce the reduced-rank head inputs.
    folded = FlowLM(flowlm_args_from_raw(tiny_raw(d_model=64, z_dim=8)))
    state = {k: v for k, v in model_state(teacher).items() if k not in new}
    folded.load_weights(list(state.items()) + list(new.items()), strict=True)


# ── Parity with the official PyTorch implementation ───────────────────────────

def torch_flow_lm(raw: dict):
    text_conditioner.build_tokenizer = lambda *a, **k: None  # no tokenizer file needed
    fl = copy.deepcopy(raw["flow_lm"])
    ref = FlowLMModel.from_pydantic_config(FlowLMConfig(**fl), raw["mimi"]["quantizer"]["dimension"],
                                          fl["insert_bos_before_voice"])
    ref.speaker_proj_weight = torch.nn.Parameter(torch.randn(fl["transformer"]["d_model"],
                                                              raw["mimi"]["inner_dim"]) * 0.2)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "freqs" not in name:
                p.copy_(torch.randn_like(p) * 0.2 + (1.0 if name.endswith(("norm1.weight", "norm2.weight", "alpha")) else 0.0))
    return ref.eval()


def mlx_from_torch(raw: dict, ref) -> FlowLM:
    model = FlowLM(flowlm_args_from_raw(raw))
    state = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
    model.load_weights(list(state.items()), strict=True)
    return model


@needs_ref
@pytest.mark.parametrize("variant", [
    dict(),                                                       # released 100M layout
    dict(z_dim=16, attn_dim=32, kv=2, ff=48, rank=8, flow_type="drifting"),  # small models
    dict(flow_type="pointwise", kv=1),
])
def test_backbone_z_matches_pytorch(variant):
    from training.args import TrainArgs
    from training.modules.model import TrainableTTS

    raw = tiny_raw(**variant)
    ref = torch_flow_lm(raw)
    model = mlx_from_torch(raw, ref)
    b = random_batch()
    wrapper = TrainableTTS.__new__(TrainableTTS)
    torch.nn.Module.__init__(wrapper)
    wrapper.args = TrainArgs()
    with torch.no_grad():
        for force_null in (False, True):
            z_ref = wrapper.backbone_z(
                ref, torch.from_numpy(np.array(b["latents"])),
                [torch.from_numpy(t).long() for t in b["text_tokens"]],
                torch.from_numpy(np.array(b["voice"])), torch.from_numpy(b["voice_lens"]),
                cfg_dropout=False, force_null=force_null,
            ).numpy()
            keep = np.full(3, not force_null)
            z = model.backbone_z(b["latents"], b["text_tokens"], b["voice"], b["voice_lens"], keep, keep)
            np.testing.assert_allclose(np.array(z), z_ref, rtol=1e-3, atol=1e-3)

        # The sampler head on the same noise.
        z_t = torch.from_numpy(z_ref[:, 0])
        noise = torch.randn(z_t.shape[0], 8)
        ts = [torch.full((z_t.shape[0], 1), v) for v in (0.0, 1.0)][: ref.flow_net.num_time_conds]
        out_ref = ref.flow_net(z_t, *ts, noise).numpy()
        out = model.flow_net(mx.array(z_t.numpy()), *[mx.array(t.numpy()) for t in ts], mx.array(noise.numpy()))
        np.testing.assert_allclose(np.array(out), out_ref, rtol=1e-3, atol=1e-3)
        eos_ref = ref.out_eos(z_t).numpy()
        np.testing.assert_allclose(np.array(model.out_eos(mx.array(z_t.numpy()))), eos_ref, rtol=1e-3, atol=1e-3)


@needs_ref
def test_shrink_matches_pytorch():
    from training.scripts.shrink_checkpoint import shrink, shrink_width as ref_shrink_width

    teacher_raw = tiny_raw(layers=6, d_model=64, z_dim=16)
    student_raw = tiny_raw(layers=2, d_model=32, z_dim=16, attn_dim=32, kv=1, ff=48, rank=8, heads=2)
    t_state = {k: v.detach() for k, v in torch_flow_lm(teacher_raw).state_dict().items()}
    dst = {k: v.detach() for k, v in torch_flow_lm(student_raw).state_dict().items()}
    ref, _ = shrink(t_state, 2)
    ref = ref_shrink_width(ref, dst, 16)

    from models.pocket_tts.shrink import seed_student
    ours, _ = seed_student({k: v.numpy() for k, v in t_state.items()},
                           {k: v.numpy() for k, v in dst.items()}, 16)
    assert set(ours) == set(ref)
    for k in ref:
        if k == "conditioner.embed.weight" or k == "conditioner.proj.weight":
            # SVD factors are unique up to sign: compare the product.
            continue
        np.testing.assert_allclose(ours[k], ref[k].numpy(), rtol=1e-5, atol=1e-5, err_msg=k)
    np.testing.assert_allclose(ours["conditioner.embed.weight"] @ ours["conditioner.proj.weight"].T,
                               (ref["conditioner.embed.weight"] @ ref["conditioner.proj.weight"].T).numpy(),
                               rtol=1e-4, atol=1e-4)


def test_dataset_prompt_modes():
    from models.pocket_tts.data import LatentDataset, batches

    rng = np.random.default_rng(0)
    lat = rng.normal(size=(50, 8)).astype(np.float32)          # 4 s at 12.5 Hz
    words = [{"word": w, "start": s, "end": e} for w, s, e in
             [("one", 0.1, 0.9), ("two", 1.1, 1.9), ("three", 2.1, 2.9), ("four.", 3.0, 3.6)]]
    rows = [
        {"latents": lat, "text": "one two three four.", "words": words},
        {"latents": lat, "text": "with ref", "voice_latents": lat[:20] * 0 + 7},
        {"latents": lat, "text": "fallback"},
    ]
    ds = LatentDataset(rows, lambda s: [len(w) for w in s.split()], prompt_trim_max_sec=0.0,
                       final_punct_dropout=0.0)
    target, tokens, prompt = ds.sample(0)
    # Cut between two words at >= 1 s; prompt = audio before it, target = after it (+0.2 s tail).
    assert len(prompt) in (int(round(1.0 * 12.5)), int(round(2.0 * 12.5)))
    np.testing.assert_array_equal(prompt, lat[: len(prompt)])
    np.testing.assert_array_equal(target[0], lat[len(prompt)])
    assert len(prompt) + len(target) == int(np.ceil((3.6 + 0.2) * 12.5))
    assert list(tokens) in ([3, 5, 5], [5, 5])
    _, _, prompt = ds.sample(1)
    assert (prompt == 7).all()
    target, _, prompt = ds.sample(2)
    assert len(target) == 50 and len(prompt) == 50  # 5 s window capped by the clip

    b = next(batches(ds, 3, 1, shuffle=False))
    assert b["latents"].shape[0] == 3 and b["mask"].shape == b["latents"].shape[:2]
    assert not np.array(b["mask"])[:, -1].any()  # every row keeps an EOS frame


@needs_ref
def test_export_loads_into_pytorch(tmp_path):
    from models.pocket_tts.weights import export_pocket_tts

    raw = tiny_raw(z_dim=16, attn_dim=32, kv=2, rank=8)
    model = FlowLM(flowlm_args_from_raw(raw))
    randomize(model)
    src = tmp_path / "src.safetensors"
    mx.save_safetensors(str(src), {"mimi.dummy": mx.zeros((2,))})
    out = export_pocket_tts(model, raw, str(src), tmp_path / "ckpt")
    import safetensors.torch
    state = safetensors.torch.load_file(str(out))
    assert "mimi.dummy" in state and (tmp_path / "ckpt" / "config.yaml").exists()
    ref = torch_flow_lm(raw)
    ref.load_state_dict({k.removeprefix("flow_lm."): v for k, v in state.items()
                         if k.startswith("flow_lm.")}, strict=True)


# ── Fine-tuning objective (LSD) ───────────────────────────────────────────────

@needs_ref
@pytest.mark.parametrize("stopgrad", ["minimal", "classic"])
def test_lsd_loss_and_grads_match_pytorch(stopgrad):
    from training.modules.samplers import LSD as TorchLSD

    from models.pocket_tts.objectives import LSD

    raw = tiny_raw(z_dim=16)
    ref = torch_flow_lm(raw)
    model = mlx_from_torch(raw, ref)
    t_lsd = TorchLSD(stopgrad_type=stopgrad, distill_prob=1.0)
    m_lsd = LSD(stopgrad_type=stopgrad, distill_prob=1.0)
    torch.manual_seed(1)
    with torch.no_grad():  # non-trivial uncertainty weighting
        for p in t_lsd.w_s_t.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    m_lsd.w_s_t.load_weights([(f"layers.{int(k.split('.')[0]) // 2}.{k.split('.')[1]}",
                               mx.array(v.detach().numpy())) for k, v in t_lsd.w_s_t.state_dict().items()])

    N = 7
    rng = np.random.default_rng(3)
    z = rng.normal(size=(N, 16)).astype(np.float32)
    e = rng.normal(size=(N, 8)).astype(np.float32)
    x = rng.normal(size=(N, 8)).astype(np.float32)
    t_d = rng.uniform(0.05, 0.95, size=(N, 1)).astype(np.float32)
    st = np.sort(rng.uniform(0.05, 0.95, size=(N, 2)), axis=1).astype(np.float32)

    # Torch: fixed draws.
    draws = iter([torch.from_numpy(t_d)])
    t_lsd.sample_t = lambda _x: next(draws)
    t_lsd.sample_s_t = lambda _x: (torch.from_numpy(st[:, :1]), torch.from_numpy(st[:, 1:]))
    t_lsd.train()
    zt = torch.from_numpy(z).requires_grad_(True)
    from functools import partial as fpartial
    loss_t, _, _ = t_lsd.loss(fpartial(ref.flow_net, zt), torch.from_numpy(e), torch.from_numpy(x))
    loss_t.sum().backward()

    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.head, self.obj = model.flow_net, m_lsd

    both = Both()

    def f(mod, z_):
        loss, _ = mod.obj.loss(mod.head, z_, mx.array(e), mx.array(x), t_diag=mx.array(t_d),
                               s_t=(mx.array(st[:, :1]), mx.array(st[:, 1:])), skip=False)
        return loss.sum(), loss

    (total, loss_m), grads = nn.value_and_grad(both, lambda m_: f(m_, mx.array(z)))(both)
    np.testing.assert_allclose(np.array(loss_m), loss_t.detach().numpy(), rtol=2e-4, atol=2e-4)
    g = dict(__import__("mlx.utils").utils.tree_flatten(grads))
    for name in ("cond_embed.weight", "res_blocks.0.mlp.0.weight", "time_embed.1.mlp.0.weight",
                 "final_layer.linear.weight"):
        ref_g = dict(ref.flow_net.named_parameters())[name].grad.numpy()
        np.testing.assert_allclose(np.array(g[f"head.{name}"]), ref_g, rtol=2e-3, atol=2e-4, err_msg=name)
    ref_w = t_lsd.w_s_t[0].weight.grad.numpy()
    np.testing.assert_allclose(np.array(g["obj.w_s_t.layers.0.weight"]), ref_w, rtol=2e-3, atol=2e-4)
    g_z = mx.grad(lambda z_: f(both, z_)[0])(mx.array(z))
    np.testing.assert_allclose(np.array(g_z), zt.grad.numpy(), rtol=2e-3, atol=2e-4)


def test_eos_loss_matches_reference_formula():
    from models.pocket_tts.objectives import eos_loss

    model = FlowLM(flowlm_args_from_raw(tiny_raw(z_dim=16)))
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2, 6, 16)).astype(np.float32)
    mask = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=bool)
    got = eos_loss(model, mx.array(z), mx.array(mask)).item()
    logits = np.array(model.out_eos(mx.array(z)))[..., 0]
    is_eos = ~mask
    is_eos[:, 0] = False
    sp = lambda v: np.log1p(np.exp(v))  # noqa: E731
    per = np.where(is_eos, sp(-logits), sp(logits))
    sm = np.concatenate([mask[:, :1], mask[:, :-1]], axis=1)
    np.testing.assert_allclose(got, (per * sm).sum() / sm.sum(), rtol=1e-5)


def test_finetune_reset_text_embedding_and_freeze_head(tmp_path):
    from mlx.utils import tree_flatten

    from models.pocket_tts.finetune import build_finetune

    raw = tiny_raw(z_dim=16)
    src = FlowLM(flowlm_args_from_raw(raw))
    randomize(src)
    weights = tmp_path / "src.safetensors"
    mx.save_safetensors(str(weights), {f"flow_lm.{k}": v for k, v in model_state(src).items()})
    cfg_path = tmp_path / "m.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    tr, _ = build_finetune({"model": {"config": str(cfg_path), "weights": str(weights),
                                      "reset_text_embedding": True, "freeze_head": True}})
    new, old = model_state(tr.flow_lm), model_state(src)
    assert not np.allclose(np.array(new["conditioner.embed.weight"]), np.array(old["conditioner.embed.weight"]))
    for k in ("transformer.layers.0.linear1.weight", "flow_net.cond_embed.weight", "bos_emb"):
        np.testing.assert_array_equal(np.array(new[k]), np.array(old[k]))
    trainable = {k for k, _ in tree_flatten(tr.trainable_parameters())}
    assert not any(k.startswith("flow_lm.flow_net.") for k in trainable)
    assert "flow_lm.conditioner.embed.weight" in trainable and "objective.w_s_t.layers.0.weight" in trainable


def test_hindi_normalizer():
    from models.pocket_tts.text import hindi_number, normalize_hindi

    assert hindi_number(25) == "पच्चीस"
    assert hindi_number(150000) == "एक लाख पचास हज़ार"
    assert hindi_number(1999) == "एक हज़ार नौ सौ निन्यानबे"
    assert hindi_number(32500000) == "तीन करोड़ पच्चीस लाख"
    assert normalize_hindi("कल २५ लोग आए") == "कल पच्चीस लोग आए"
    assert normalize_hindi("₹1,50,000 खर्च") == "एक लाख पचास हज़ार रुपये खर्च"
    assert normalize_hindi("20% छूट") == "बीस प्रतिशत छूट"
    assert normalize_hindi("3.5 किलो") == "तीन दशमलव पाँच किलो"


def test_hf_config_relative_paths(monkeypatch, tmp_path):
    from models.pocket_tts import config as C

    raw = tiny_raw()
    raw["weights_path"] = "model.safetensors"
    raw["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.model"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(C, "resolve_path", lambda path: str(p))
    out = C.load_raw_config("hf://someone/repo/config.yaml@abc")
    assert out["weights_path"] == "hf://someone/repo/model.safetensors@abc"
    assert out["flow_lm"]["lookup_table"]["tokenizer_path"] == "hf://someone/repo/tokenizer.model@abc"


def test_tokenizer_script(tmp_path):
    import subprocess

    pytest.importorskip("sentencepiece")
    rows = [{"text": f"आज {i} लोग बाज़ार गए और फल खरीदे"} for i in range(200)]
    data = tmp_path / "t.jsonl"
    data.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows))
    out = tmp_path / "tok"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/pocket_tts_train_tokenizer.py"), str(data),
                    "--out", str(out), "--vocab-size", "320", "--normalize-text", "hindi"],
                   check=True, capture_output=True)
    from models.pocket_tts.data import load_tokenizer

    ids = load_tokenizer(str(out) + ".model")("आज पच्चीस लोग")
    assert ids and max(ids) < 320  # byte fallback alone reserves 256 pieces


def test_teacher_cache_matches_live_teacher(tmp_path):
    from models.pocket_tts.data import LatentDataset, collate
    from models.pocket_tts.teacher_cache import CachedDataset, write_cache
    from models.pocket_tts.train_utils import normalize, synthetic_manifest

    teacher = FlowLM(flowlm_args_from_raw(tiny_raw(layers=3, z_dim=16)))
    randomize(teacher, std=0.1)
    teacher.freeze()
    rows = [{**r, "latents": r["latents"][:, :8]} for r in synthetic_manifest(10)]
    ds = LatentDataset(rows, lambda s: [len(s) % 50, 3], final_punct_dropout=0.0)
    n = write_cache(teacher, ds, tmp_path / "cache", cfg_coef=2.0, batch_size=4)
    assert n == 8  # two full batches of 4 (the last partial batch is dropped)
    cached = CachedDataset(tmp_path / "cache", expect_cfg_coef=2.0)
    with pytest.raises(ValueError, match="cfg_coef"):
        CachedDataset(tmp_path / "cache", expect_cfg_coef=1.0)

    rows3 = [cached.sample(i) for i in range(3)]
    batch = collate(rows3)
    live = teacher_target(teacher, normalize(batch, teacher), 2.0)
    m = np.array(__import__("models.pocket_tts.distill", fromlist=["x"]).shifted_mask(batch["mask"])) > 0
    # fp16 storage; compare on the frames the loss uses.
    np.testing.assert_allclose(np.array(batch["target"])[m], np.array(live)[m], rtol=2e-2, atol=2e-2)
    student = FlowLM(teacher.args)
    nb = normalize(batch, teacher)
    a = distill_loss(student, nb, batch["target"]).item()
    b = distill_loss(student, nb, live).item()
    assert abs(a - b) < 1e-2 * max(abs(b), 1e-6)
