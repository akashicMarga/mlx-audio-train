"""
Lever registry: each optimisation is a named, self-contained patch applied on
top of the stock mlx-audio baseline, so a run is "baseline + these levers" and
every ledger row can name exactly what changed.

A lever's `apply(model, gen_kwargs)` mutates the loaded model and/or the kwargs
passed to `model.generate()`. Levers must be independent and composable; a run
applies them in the order given on the command line.

Configuration-only changes don't need a lever:
  - precision      -> pick a different --model-id (bf16 / 8bit / 6bit / 4bit)
  - streaming      -> --stream-interval (first-chunk size, seconds)
  - batch size     -> scripts/bench_sweep.py with the batch session (§7)
Set --lever / --change on the run so the ledger still records them.

Planned levers are registered with `implemented=False` so `--list-levers` shows
the whole ladder; selecting one fails loudly until it is written.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class Lever:
    name: str
    post_section: str
    description: str
    expect_bit_exact: bool          # True if output must match the parent exactly
    apply: Optional[Callable] = None

    @property
    def implemented(self) -> bool:
        return self.apply is not None


LEVERS: Dict[str, Lever] = {}


def register(name, post_section, description, expect_bit_exact):
    def deco(fn):
        LEVERS[name] = Lever(name, post_section, description, expect_bit_exact, fn)
        return fn
    return deco


def plan(name, post_section, description, expect_bit_exact):
    LEVERS[name] = Lever(name, post_section, description, expect_bit_exact, None)


# ── Loop levers: route generate() through bench/fast_generate.py ────────────
def _loop_opts(model):
    from bench.fast_generate import LoopOptions
    return getattr(model, "_bench_loop", None) or LoopOptions()


def _install(model, **changes):
    from dataclasses import replace
    from bench.fast_generate import install
    install(model, replace(_loop_opts(model), **changes))


@register("loop-copy", "5", "stock decode loop copied into bench/fast_generate.py, no switches "
          "(validates the copy: must be bit-identical to stock)", True)
def _loop_copy(model, gen_kwargs):
    _install(model)


@register("device-rep-penalty", "5.2", "repetition penalty from a fixed-shape [vocab] mask on the "
          "device instead of a Python set per frame", True)
def _device_rep_penalty(model, gen_kwargs):
    _install(model, device_rep_penalty=True)


@register("async-eval", "5.1", "mx.async_eval frame t while building t+1; EOS checked one frame "
          "late (needs device-rep-penalty first)", True)
def _async_eval(model, gen_kwargs):
    if not _loop_opts(model).device_rep_penalty:
        raise SystemExit("async-eval needs device-rep-penalty earlier in --levers")
    _install(model, async_eval=True)


# ── Planned ladder (Part 4, §5) — implement one at a time ────────────────────


@register("compile-code-pred", "5.2", "whole 15-pass code-predictor frame (layers, heads, sampling, "
          "embedding sum) as one mx.compile'd function; apply after weight-changing levers", False)
def _compile_code_pred(model, gen_kwargs):
    _install(model, compile_code_pred=True)


@register("compile-talker", "5.2", "Talker decode step as one mx.compile'd call on fixed-capacity K/V "
          "slot buffers (one trace per utterance); needs device-rep-penalty", False)
def _compile_talker(model, gen_kwargs):
    if not _loop_opts(model).device_rep_penalty:
        raise SystemExit("compile-talker needs device-rep-penalty earlier in --levers")
    _install(model, compile_talker=True)


@register("decoder-compile", "5.3", "codec decoder mx.compile'd per 16-frame length bucket "
          "(causal decoder: right-pad codes, decode, trim)", False)
def _decoder_compile(model, gen_kwargs):
    from bench.decoder_compile import install
    install(model, bucket=16)
plan("cp-prealloc", "5.6", "fixed 16-slot K/V buffers for the code predictor; reset offset only", True)
plan("ref-cache", "5.7", "cache reference mel / speaker embedding / ref codes per clip (cloning path)", True)


@register("lora-merge-bf16", "5.9", "dequantize base, add scale·B·A (fp32), keep merged layers bf16", False)
def _lora_merge_bf16(model, gen_kwargs):
    from bench.lora_merge import merge
    merge(model, "bf16")


@register("lora-merge-q8", "5.9", "dequantize base, add scale·B·A (fp32), requantize with the base's "
          "group size and bits", False)
def _lora_merge_q8(model, gen_kwargs):
    from bench.lora_merge import merge
    merge(model, "q8")


def resolve(names):
    out = []
    for n in names:
        if n not in LEVERS:
            raise SystemExit(f"unknown lever '{n}'. Known: {', '.join(sorted(LEVERS))}")
        lv = LEVERS[n]
        if not lv.implemented:
            raise SystemExit(f"lever '{n}' (§{lv.post_section}) is planned but not implemented yet")
        out.append(lv)
    return out


def describe() -> str:
    rows = [f"{'lever':<20} {'§':<5} {'impl':<5} {'bit-exact':<9} description"]
    for lv in LEVERS.values():
        rows.append(f"{lv.name:<20} {lv.post_section:<5} {'yes' if lv.implemented else 'no':<5} "
                    f"{'yes' if lv.expect_bit_exact else 'no':<9} {lv.description}")
    return "\n".join(rows)


# ── Mixed precision (§5.8) ────────────────────────────────────────────────────
@register("cp-q4", "5.8", "requantize the code predictor (layers + 15 heads) to 4-bit, group 64; "
          "Talker unchanged. Apply after lora-merge-q8", False)
def _cp_q4(model, gen_kwargs):
    from bench.quantize import requantize
    requantize(model.talker.code_predictor, bits=4, group_size=64, label="code_predictor")


@register("cp-q4-g32", "5.8", "as cp-q4 but group size 32 (finer scales, ~0.5 bit/weight more)", False)
def _cp_q4_g32(model, gen_kwargs):
    from bench.quantize import requantize
    requantize(model.talker.code_predictor, bits=4, group_size=32, label="code_predictor")


def _is_mlp(path):
    return ".mlp." in f".{path}."


@register("cp-mlp-q4", "5.8", "code predictor MLP (gate/up/down) → 4-bit g64; attention and the 15 heads "
          "stay q8 (selective, after FluidInference PocketTTS / ML Drift 8/4/4). Apply after lora-merge-q8", False)
def _cp_mlp_q4(model, gen_kwargs):
    from bench.quantize import requantize
    requantize(model.talker.code_predictor, bits=4, group_size=64, label="code_predictor.mlp",
               include=_is_mlp)


@register("talker-mlp-q4", "5.8", "Talker MLP (gate/up/down) → 4-bit g64; attention and codec head stay q8", False)
def _talker_mlp_q4(model, gen_kwargs):
    from bench.quantize import requantize
    requantize(model.talker.model, bits=4, group_size=64, label="talker.mlp", include=_is_mlp)


# ── Kernel-launch levers (§5) ────────────────────────────────────────────────
@register("fuse-proj", "5", "fused QKV and gate/up projections (7 → 4 matmul launches per layer, "
          "ML Drift-style layout fusion). Apply after lora-merge-q8 and any requantization", True)
def _fuse_proj(model, gen_kwargs):
    from bench.fuse_proj import fuse
    fuse(model)


@register("fast-cp-sampler", "5", "code-predictor sampling does top-p over the top-k survivors only "
          "(inside the compiled frame; same random draw shape as stock). Needs compile-code-pred", True)
def _fast_cp_sampler(model, gen_kwargs):
    if not _loop_opts(model).compile_code_pred:
        raise SystemExit("fast-cp-sampler needs compile-code-pred earlier in --levers")
    _install(model, cp_sampler="fast")
