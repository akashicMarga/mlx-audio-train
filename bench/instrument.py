"""
Component timing for the instrumented pass.

MLX is lazy, so timing a component means forcing evaluation at its boundary:
evaluate its inputs before starting the clock and its outputs before stopping
it. Those extra syncs change the schedule, so numbers from this pass are only
used for the per-component breakdown (talker / code predictor / codec), never
for RTF or TTFP — those come from the uninstrumented pass.

The Talker and code predictor are invoked through `__call__`, which Python looks
up on the class, so they are patched at class level for the duration of the
`with` block and restored afterwards.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import mlx.core as mx


def _arrays(obj):
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _arrays(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _arrays(o)


def _force(obj):
    arrs = list(_arrays(obj))
    if arrs:
        mx.eval(*arrs)


@dataclass
class ComponentTimes:
    talker_prefill_s: List[float] = field(default_factory=list)
    talker_step_s: List[float] = field(default_factory=list)
    code_pred_s: List[float] = field(default_factory=list)
    decode_s: List[float] = field(default_factory=list)

    def summary(self, audio_s: float, wall_s: float) -> dict:
        steps = len(self.talker_step_s)
        # The prefill call also emits frame 0's codebook-0 token, and the code
        # predictor runs on it, so frames = decode steps + prefills.
        frames = steps + len(self.talker_prefill_s)
        talker = sum(self.talker_step_s)
        cp = sum(self.code_pred_s)
        dec = sum(self.decode_s)
        prefill = sum(self.talker_prefill_s)
        other = wall_s - talker - cp - dec - prefill
        per = (lambda x: 1e3 * x / frames) if frames else (lambda x: None)
        return {
            "frames": frames,
            "code_pred_calls": len(self.code_pred_s),
            "talker_ms_step": 1e3 * talker / steps if steps else None,   # decode steps only
            "mtp_ms_step": per(cp),                      # all passes for one frame
            "codec_ms_per_audio_s": 1e3 * dec / audio_s if audio_s else None,
            "prefill_ms_total": 1e3 * prefill,
            "other_ms_per_frame": per(other),            # sampling, embedding sum, Python, sync
            "instrumented_wall_s": wall_s,
        }


def _timed(fn, sink_for):
    def wrapper(*args, **kwargs):
        _force((args, kwargs))
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _force(out)
        sink_for(args, kwargs).append(time.perf_counter() - t0)
        return out
    return wrapper


@contextmanager
def instrument(model):
    """Patch talker / code predictor / codec decode with eval-bracketed timers."""
    times = ComponentTimes()
    talker_cls = type(model.talker)
    cp_cls = type(model.talker.code_predictor)
    tok = model.speech_tokenizer

    orig_talker = talker_cls.__call__
    orig_cp = cp_cls.__call__
    own_talker = "__call__" in vars(talker_cls)
    own_cp = "__call__" in vars(cp_cls)
    had_inst_decode = "decode" in vars(tok)
    orig_decode = tok.decode

    def talker_sink(args, kwargs):
        x = args[1] if len(args) > 1 else kwargs.get("inputs_embeds", kwargs.get("input_embeds"))
        seq = x.shape[1] if isinstance(x, mx.array) and x.ndim >= 2 else 1
        return times.talker_prefill_s if seq > 1 else times.talker_step_s

    # With the compiled code-predictor frame, its modules run inside a compile
    # trace where forcing evaluation is illegal: time the compiled frame instead.
    loop = getattr(model, "_bench_loop", None)
    cp_compiled = bool(loop and getattr(loop, "compile_code_pred", False))

    talker_compiled = bool(loop and getattr(loop, "compile_talker", False))
    if talker_compiled:
        model._bench_talker_timer = (times.talker_prefill_s, times.talker_step_s)
    else:
        talker_cls.__call__ = _timed(orig_talker, talker_sink)
    if cp_compiled:
        model._bench_cp_timer = times.code_pred_s
    else:
        cp_cls.__call__ = _timed(orig_cp, lambda a, k: times.code_pred_s)
    tok.decode = _timed(orig_decode, lambda a, k: times.decode_s)
    try:
        yield times
    finally:
        if cp_compiled:
            del model._bench_cp_timer
        if talker_compiled:
            del model._bench_talker_timer
        patched = (([] if talker_compiled else [(talker_cls, orig_talker, own_talker)])
                   + ([] if cp_compiled else [(cp_cls, orig_cp, own_cp)]))
        for cls, orig, own in patched:
            if own:
                cls.__call__ = orig
            else:
                del cls.__call__
        if had_inst_decode:
            tok.decode = orig_decode
        else:
            del tok.decode
