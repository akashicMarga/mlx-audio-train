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
