"""Tests for the benchmark harness that need no model download."""

import csv
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench import ledger, levers  # noqa: E402
from bench.instrument import instrument  # noqa: E402
from bench.quality import GateConfig, audio_hash, bit_exact, propose, wilcoxon_p  # noqa: E402

BLOG_LEDGER = Path(os.path.expanduser(
    "~/Documents/akashicMarga.github.io/_data/qwen3tts_mlx_ledger.csv"))


def _row(**kw):
    base = dict(device="M5 Pro", variant="hindi-grpo", config_id="c1", lever="baseline",
                decision="baseline", rtf=0.123456789)
    base.update(kw)
    return base


# ── ledger ────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not BLOG_LEDGER.exists(), reason="blog repo not checked out here")
def test_columns_match_blog_ledger():
    assert ledger.read_header(BLOG_LEDGER) == ledger.COLUMNS


def test_append_writes_header_once_and_formats(tmp_path):
    p = tmp_path / "l.csv"
    ledger.append_row(_row(config_id="a"), p)
    ledger.append_row(_row(config_id="b", rtf=None), p)
    rows = list(csv.DictReader(p.open()))
    assert [r["config_id"] for r in rows] == ["a", "b"]
    assert rows[0]["rtf"] == "0.123457" and rows[1]["rtf"] == ""
    assert ledger.config_ids(p) == {"a", "b"}


def test_append_refuses_mismatched_header(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("date,device\n")
    with pytest.raises(ValueError, match="different header"):
        ledger.append_row(_row(), p)


def test_validate_rejects_unknown_column_and_bad_decision():
    with pytest.raises(ValueError, match="unknown"):
        ledger.validate_row(_row(bogus=1))
    with pytest.raises(ValueError, match="decision"):
        ledger.validate_row(_row(decision="maybe"))


# ── quality / gate ────────────────────────────────────────────────────────────
def _items(cers, seed=0, sha=None):
    return [{"seed": seed, "prompt_idx": i, "cer_capped": c,
             "audio_sha1": sha[i] if sha else f"h{i}"} for i, c in enumerate(cers)]


def test_audio_hash_detects_any_change():
    a = np.linspace(-1, 1, 100, dtype=np.float32)
    b = a.copy()
    b[3] += 1e-7
    assert audio_hash(a) == audio_hash(a.copy()) != audio_hash(b)


def test_wilcoxon_pairs_by_seed_and_prompt():
    parent = _items([0.10] * 10)
    worse = _items([0.30] * 10)
    assert wilcoxon_p(worse, parent) < 0.01
    assert wilcoxon_p(parent, parent) == 1.0
    assert wilcoxon_p(_items([0.3] * 10, seed=1), parent) is None   # no overlapping keys


def test_gate_proposals():
    g = GateConfig(min_speedup=1.03)
    parent_items = _items([0.1] * 10)
    parent = {"rtf": 0.10}
    # bit-identical and 20% faster -> accepted
    d, _ = propose({"rtf": 0.08}, parent, _items([0.1] * 10), parent_items, g)
    assert d == "accepted"
    # identical output, 1% faster -> no-benefit
    d, _ = propose({"rtf": 0.099}, parent, _items([0.1] * 10), parent_items, g)
    assert d == "no-benefit"
    # much faster but CER significantly worse -> rejected
    worse = _items([0.4] * 10, sha=[f"x{i}" for i in range(10)])
    d, why = propose({"rtf": 0.05}, parent, worse, parent_items, g)
    assert d == "rejected" and "CER worse" in why
    # output differs, no CER measured -> no decision
    unscored = [{"seed": 0, "prompt_idx": i, "audio_sha1": f"y{i}"} for i in range(10)]
    assert propose({"rtf": 0.05}, parent, unscored, parent_items, g)[0] == ""
    # 20% faster but runs are 30% noisy -> no-benefit
    d, why = propose({"rtf": 0.08, "rtf_spread": 0.30}, parent, _items([0.1] * 10), parent_items, g)
    assert d == "no-benefit" and "noise" in why
    # DNSMOS drop beyond threshold but on noisy, tiny samples -> not a rejection
    g2 = GateConfig(min_speedup=1.03, max_dnsmos_drop=0.05)
    # mean drop 0.06, but from one outlier clip: not a consistent shift
    spread = [2.8, 3.8] * 5
    par = [dict(r, dnsmos_ovrl=v) for r, v in zip(_items([0.1] * 10), spread)]
    cur = [dict(r, dnsmos_ovrl=v - (0.6 if i == 0 else 0.0), audio_sha1=f"z{i}") for i, (r, v) in
           enumerate(zip(_items([0.1] * 10), spread))]
    d, why = propose({"rtf": 0.08}, {"rtf": 0.1}, cur, par, g2)
    assert d == "accepted" and "not significant" in why
    # clear, consistent DNSMOS drop -> rejected
    cur2 = [dict(r, dnsmos_ovrl=2.5 + 0.01 * i, audio_sha1=f"w{i}") for i, r in enumerate(_items([0.1] * 10))]
    par2 = [dict(r, dnsmos_ovrl=3.3 + 0.01 * i) for i, r in enumerate(_items([0.1] * 10))]
    assert propose({"rtf": 0.08}, {"rtf": 0.1}, cur2, par2, g2)[0] == "rejected"
    # no parent -> baseline
    assert propose({"rtf": 0.1}, None, [], [], g)[0] == "baseline"


def test_bit_exact():
    a = _items([0.1, 0.2])
    assert bit_exact(a, a) is True
    assert bit_exact(_items([0.1, 0.2], sha=["h0", "zz"]), a) is False


# ── levers ────────────────────────────────────────────────────────────────────
def test_planned_levers_fail_loudly():
    with pytest.raises(SystemExit, match="not implemented"):
        levers.resolve(["cp-prealloc"])
    with pytest.raises(SystemExit, match="unknown lever"):
        levers.resolve(["nope"])
    assert levers.resolve([]) == []


# ── instrumentation (fake model, same call shapes as mlx-audio) ──────────────
class _Talker:
    def __call__(self, inputs_embeds, cache=None):
        return inputs_embeds * 2, inputs_embeds


class _CodePred:
    def __call__(self, inputs_embeds, cache=None, generation_step=0):
        return inputs_embeds + 1, cache, None


class _Tok:
    def decode(self, codes):
        return codes * 3, codes.shape[-1]


class _Model:
    def __init__(self):
        self.talker = _Talker()
        self.talker.code_predictor = _CodePred()
        self.speech_tokenizer = _Tok()


def test_instrument_records_and_restores():
    m = _Model()
    orig_t, orig_c = _Talker.__call__, _CodePred.__call__
    with instrument(m) as times:
        m.talker(mx.ones((1, 7, 4)))                 # prefill
        for _ in range(3):                           # 3 decode frames
            m.talker(mx.ones((1, 1, 4)))
            for _ in range(15):
                m.talker.code_predictor(mx.ones((1, 1, 4)))
        m.speech_tokenizer.decode(mx.ones((1, 3, 16)))
    assert len(times.talker_prefill_s) == 1 and len(times.talker_step_s) == 3
    assert len(times.code_pred_s) == 45 and len(times.decode_s) == 1
    s = times.summary(audio_s=0.24, wall_s=1.0)
    assert s["frames"] == 4 and s["code_pred_calls"] == 45   # 3 steps + 1 prefill
    assert _Talker.__call__ is orig_t and _CodePred.__call__ is orig_c
    assert "decode" not in vars(m.speech_tokenizer)


# ── sweep argv ────────────────────────────────────────────────────────────────
def test_sweep_to_argv():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from bench_sweep import to_argv
    assert to_argv({"config_id": "x", "skip_ttfp": True, "no_dnsmos": False, "parent": None,
                    "num_prompts": 5}) == ["--config-id", "x", "--skip-ttfp", "--num-prompts", "5"]


# ── LoRA merge (tiny quantized layer, non-zero adapter) ──────────────────────
def _tiny_qlora_model():
    import mlx.nn as nn
    from train.lora import QLoRALinear

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            lin = nn.Linear(128, 64, bias=False)
            q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
            self.proj = QLoRALinear.from_quantized(q, rank=4, alpha=8.0, dropout=0.0)
            self.proj.lora_b = mx.random.normal((64, 4), key=mx.random.key(1)) * 0.05

    class Talker(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

    class M:
        talker = Talker()
    return M()


@pytest.mark.parametrize("mode,tol", [("bf16", 1e-2), ("q8", 3e-2)])
def test_lora_merge_replaces_and_matches(mode, tol):
    import mlx.nn as nn
    from bench.lora_merge import merge
    m = _tiny_qlora_model()
    x = mx.random.normal((3, 128), key=mx.random.key(2))
    ref = m.talker.block.proj(x)
    info = merge(m, mode)
    layer = m.talker.block.proj
    assert info["merged_layers"] == 1
    assert isinstance(layer, nn.QuantizedLinear if mode == "q8" else nn.Linear)
    rel = float(mx.linalg.norm(layer(x).astype(mx.float32) - ref) / mx.linalg.norm(ref))
    assert rel < tol


def test_requantize_lists_and_attrs():
    import mlx.nn as nn
    from bench.quantize import requantize

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.QuantizedLinear.from_linear(nn.Linear(128, 64), group_size=64, bits=8)
            self.heads = [nn.QuantizedLinear.from_linear(nn.Linear(128, 32, bias=False), group_size=64, bits=8)
                          for _ in range(3)]
    r = Root()
    info = requantize(r, bits=4, group_size=64)
    assert info["layers"] == 4
    assert r.proj.bits == 4 and all(h.bits == 4 for h in r.heads)
    assert info["rel_err_max"] < 0.2


# ── A: fused projections give identical outputs (real mlx-audio classes) ─────
def _tiny_cp_layer():
    import mlx.nn as nn
    from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerCodePredictorConfig
    from mlx_audio.tts.models.qwen3_tts import talker as T
    cfg = Qwen3TTSTalkerCodePredictorConfig(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=32, num_hidden_layers=1)
    layer = T.CodePredictorDecoderLayer(cfg, 0)
    nn.quantize(layer, group_size=64, bits=8)
    rope = T.RotaryEmbedding(32)
    return T, layer, rope


def test_fuse_proj_matches_separate():
    import mlx.nn as nn
    from bench.fuse_proj import fuse, unpatch_classes
    T, layer, rope = _tiny_cp_layer()
    x = mx.random.normal((1, 3, 128), key=mx.random.key(3)).astype(mx.bfloat16)
    pe = rope(x, mx.arange(3)[None, :])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(3).astype(x.dtype)
    ref = layer(x, pe, mask, None)

    class Talker(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [layer]

    class M:
        talker = Talker()
    try:
        info = fuse(M())
        assert info["attention_fused"] == 1 and info["mlp_fused"] == 1
        assert "qkv_proj" in layer.self_attn and "q_proj" not in layer.self_attn
        out = layer(x, pe, mask, None)
        assert mx.array_equal(out, ref), float(mx.abs(out - ref).max())
    finally:
        unpatch_classes()


def test_fuse_proj_refuses_unmerged_lora():
    import mlx.nn as nn
    from bench.fuse_proj import _fuse
    from train.lora import QLoRALinear
    q = nn.QuantizedLinear.from_linear(nn.Linear(128, 64, bias=False), group_size=64, bits=8)
    with pytest.raises(SystemExit, match="merged"):
        _fuse([QLoRALinear.from_quantized(q, 4, 8.0, 0.0)] * 3)


# ── B: fast code-predictor sampler draws the same tokens as stock ────────────
def test_fast_cp_sampler_matches_stock():
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
    from bench.fast_generate import _sample_topk_topp
    mismatches = 0
    for seed in range(60):
        logits = (mx.random.normal((2, 1, 2048), key=mx.random.key(100 + seed)) * 3).astype(mx.bfloat16)
        mx.random.seed(seed)
        a = Model._sample_token_batch(None, logits, temperature=0.9, top_k=50, top_p=0.95)
        mx.random.seed(seed)
        b = _sample_topk_topp(logits, temperature=0.9, top_k=50, top_p=0.95)
        mismatches += int((a != b).sum())
    assert mismatches == 0, f"{mismatches} differing tokens out of 120"


# ── C′: selective requantization touches only MLP layers ─────────────────────
def test_requantize_mlp_only():
    from bench.levers import _is_mlp
    from bench.quantize import requantize
    T, layer, _ = _tiny_cp_layer()
    info = requantize(layer, bits=4, group_size=64, include=_is_mlp)
    assert info["layers"] == 3
    assert all(getattr(layer.mlp, n).bits == 4 for n in ("gate_proj", "up_proj", "down_proj"))
    assert all(getattr(layer.self_attn, n).bits == 8 for n in ("q_proj", "k_proj", "v_proj", "o_proj"))


def test_telemetry_unavailable_is_graceful():
    import bench.telemetry as tm
    old = tm.CANDIDATES
    tm.CANDIDATES = ["/nonexistent/macmon"]
    try:
        with tm.Telemetry() as t:
            t.mark_start(); t.mark_end()
        s = t.summary()
        assert s["available"] is False and "not found" in s["reason"]
        assert "unavailable" in tm.note(s)
    finally:
        tm.CANDIDATES = old
