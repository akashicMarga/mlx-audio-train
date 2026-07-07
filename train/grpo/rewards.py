"""
grpo/rewards.py — Phase A scoring for GRPO.

Programmatic, model-free rewards per the axis we care about, plus a degeneracy
guard, combined and group-normalised into advantages. Nothing here touches
autograd — rewards are scalars computed on decoded audio.

Pipeline 1 (language adaptation):  intelligibility (1 − CER) via mlx-whisper.
Pipeline 2 (speaker cloning):       + speaker-similarity (frozen speaker_encoder).
Both:                               + length/degeneracy guard.

Reward → advantage:
    A_i = (r_i − mean_group(r)) / (std_group(r) + eps)
group = the `group_size` rollouts of one prompt (DeepSeek-style, critic-free).
"""

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import mlx.core as mx

from data.audio_utils import _resample


WHISPER_SR = 16000


# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation (for CER)
# ──────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[।॥.,!?;:\"'`´’‘“”()\[\]{}<>/\\|@#%^&*_+=~–—-]")


def normalize_text(s: str) -> str:
    """NFC-normalise, strip punctuation (incl. Devanagari danda), collapse spaces,
    lowercase. Applied to both hypothesis and reference before CER so the metric
    reflects phonetic content, not punctuation the TTS never voices."""
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT_RE.sub(" ", s)
    s = s.lower()
    return " ".join(s.split())


def char_error_rate(hyp: str, ref: str) -> float:
    """CER in [0, ∞). Uses jiwer if present, else a Levenshtein fallback."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if len(ref_n) == 0:
        return 0.0 if len(hyp_n) == 0 else 1.0
    try:
        import jiwer
        return float(jiwer.cer(ref_n, hyp_n))
    except ImportError:
        return _levenshtein(list(hyp_n), list(ref_n)) / len(ref_n)


def _levenshtein(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Reward registry
# ──────────────────────────────────────────────────────────────────────────────
#
# Each reward is a named component: a function `(ctx, **params) -> {"reward": [...],
# ...extra info...}` registered under the same key the YAML `rewards:` block uses.
# `combine_rewards` iterates the config's `rewards` dict, so adding / toggling /
# reweighting a reward is a YAML edit — no change to combine_rewards or train.py.
# To add a new reward: write the function, decorate it with @register_reward("name",
# default_weight=…), and reference "name" in a config's `rewards:` block. `params`
# is the reward's YAML block minus `weight` (splatted in), so a config typo raises
# TypeError rather than silently no-op'ing.

@dataclass
class _Reward:
    fn: Callable
    default_weight: float


_REWARD_REGISTRY: Dict[str, _Reward] = {}


def register_reward(name: str, *, default_weight: float = 0.0):
    """Decorator: register `fn` as the reward named `name`. `default_weight` is used
    when a config lists the reward but omits `weight`."""
    def deco(fn: Callable) -> Callable:
        if name in _REWARD_REGISTRY:
            raise ValueError(f"reward '{name}' already registered")
        _REWARD_REGISTRY[name] = _Reward(fn=fn, default_weight=default_weight)
        return fn
    return deco


def registered_rewards() -> List[str]:
    return sorted(_REWARD_REGISTRY)


@dataclass
class RewardContext:
    """Per-call inputs shared by every reward. Rewards read only the fields they
    need (e.g. intelligibility uses audios/texts; length also uses gen_lengths/
    max_new_tokens; speaker uses model/ref_mel)."""
    audios:         List[mx.array]
    texts:          List[str]
    sample_rate:    int
    gen_lengths:    Optional[Sequence[int]] = None
    max_new_tokens: Optional[int]           = None
    model:          object                  = None
    ref_mel:        Optional[mx.array]       = None


def _invoke(name: str, ctx: RewardContext, spec: dict) -> Dict[str, List[float]]:
    """Call a registered reward with its params (the spec block minus `weight`)."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"unknown reward '{name}'; registered: {registered_rewards()}")
    params = {k: v for k, v in spec.items() if k != "weight"}
    return _REWARD_REGISTRY[name].fn(ctx, **params)


def score(name: str, ctx: RewardContext, cfg: "RewardConfig") -> Dict[str, List[float]]:
    """Run one registered reward with this cfg's params and return its RAW info dict
    (unweighted). For eval/logging that wants a metric (cer/mos/spk), not the
    weighted policy-gradient reward."""
    return _invoke(name, ctx, cfg.spec(name))


@dataclass
class RewardConfig:
    """Config for the reward stack. `rewards` maps reward-name → its YAML block
    (`{weight, ...params}`); everything else is a global read by the trainer."""
    rewards:  Dict[str, dict] = field(default_factory=dict)

    # advantage normalisation (Dr. GRPO argument, Liu et al. "Understanding
    # R1-Zero-Like Training"): "std" = group-relative + /std (DeepSeek default);
    # "none" drops the /std, which up-weights low-variance (low-info) groups.
    # TRAP: "none" shrinks advantages ~10× (see scripts/grpo_reward_ablation.py).
    adv_norm: str = "std"           # "std" or "none"
    eps:      float = 1e-4          # advantage std floor

    def spec(self, name: str) -> dict:
        return self.rewards.get(name, {})

    def weight(self, name: str) -> float:
        """Effective weight: the block's `weight`, else the reward's registered
        default; 0 if the reward isn't in the config at all (→ inactive)."""
        if name not in self.rewards:
            return 0.0
        default = _REWARD_REGISTRY[name].default_weight if name in _REWARD_REGISTRY else 0.0
        return float(self.rewards[name].get("weight", default))

    def param(self, name: str, key: str, default=None):
        return self.rewards.get(name, {}).get(key, default)

    @classmethod
    def from_config(cls, grpo_block: dict, *, default_language: str = "auto") -> "RewardConfig":
        """Build from a config's `grpo` block. The `rewards:` sub-block drives the
        stack directly. Two conveniences: the intelligibility `language` defaults to
        the trainer's lang_code, and a top-level `grpo.reward_shape`/`reward_k`
        (written by the ablation driver) is promoted onto the intelligibility reward
        for back-compat."""
        rewards = copy.deepcopy(dict(grpo_block.get("rewards", {}) or {}))
        intel = rewards.setdefault("intelligibility", {})
        intel.setdefault("language", default_language)
        for k in ("reward_shape", "reward_k"):
            if k in grpo_block and k not in intel:
                intel[k] = grpo_block[k]
        return cls(
            rewards=rewards,
            adv_norm=grpo_block.get("adv_norm", "std"),
            eps=float(grpo_block.get("eps", 1e-4)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual rewards
# ──────────────────────────────────────────────────────────────────────────────

def _to_whisper_audio(audio: mx.array, src_sr: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if src_sr != WHISPER_SR:
        wav = _resample(wav, src_sr, WHISPER_SR)
    return wav


def _shape_intel_reward(err: float, reward_shape: str, reward_k: float) -> float:
    """Map an error rate (CER/WER, ∈[0, ∞)) to a reward ∈(0, 1].

    linear: 1 − min(1, err) — flat sensitivity, hard-clamped at err≥1.
    tanh:   1 − tanh(k·err) — dense contrast near err→0, smooth saturation for
            large err (no clamp needed; tanh handles insertions err>1 gracefully).
            Stretches within-group spread ~2.6× at CER≈0.12 (break-even ≈0.38).
    """
    if reward_shape == "tanh":
        return float(1.0 - np.tanh(reward_k * err))
    return 1.0 - min(1.0, err)


@register_reward("intelligibility", default_weight=1.0)
def intelligibility_reward(
    ctx: RewardContext,
    *,
    asr_model:    str = "mlx-community/whisper-large-v3-turbo",
    language:     str = "hi",
    metric:       str = "cer",       # "cer" or "wer"
    reward_shape: str = "linear",    # "linear" or "tanh" (see _shape_intel_reward)
    reward_k:     float = 3.0,       # tanh steepness
) -> Dict[str, List[float]]:
    """r_intel = shaped(1 − error_rate(ASR(audio), text)) per rollout, where
    error_rate is CER or WER per `metric`.

    Returns {"reward": [...], "cer": [...], "wer": [...], "hyp": [...]}. BOTH cer
    and wer are always computed (cheap string ops next to the ASR that dominates
    step time) so a run can watch them diverge — WER penalises harder (a partial-
    word error fails the whole word; for Devanagari a wrong matra flips a word). Only
    `metric` selects which one drives the reward; the other rides along as a metric.

    Uses the canonical `mlx_whisper.transcribe` per rollout. (A batched
    `decode()` path was tried — ~3× faster ASR — but ASR is only ~3% of step
    time, rollout sampling dominates, and the batched scorer drifted from
    `transcribe()` even on intelligible audio. Not worth the reward-quality risk.)
    """
    import mlx_whisper

    rewards, cers, wers, hyps = [], [], [], []
    for audio, text in zip(ctx.audios, ctx.texts):
        wav = _to_whisper_audio(audio, ctx.sample_rate)
        if wav.shape[0] < WHISPER_SR // 10:        # <100 ms → treat as empty
            rewards.append(0.0); cers.append(1.0); wers.append(1.0); hyps.append("")
            continue
        # temperature=0.0 (scalar) disables the 6-way temperature fallback, which
        # otherwise re-decodes degenerate/gibberish rollouts up to 6× (and can
        # hallucinate long repetitive transcripts each pass). Early in GRPO most
        # rollouts ARE degenerate, so this is the dominant cost guard; it also
        # makes the reward deterministic. condition_on_previous_text=False avoids
        # cross-segment drift on the short clips we score.
        result = mlx_whisper.transcribe(
            wav, path_or_hf_repo=asr_model, language=language, verbose=False,
            temperature=0.0, condition_on_previous_text=False,
        )
        hyp = result.get("text", "")
        cer_v = char_error_rate(hyp, text)
        wer_v = _wer(hyp, text)
        err = cer_v if metric == "cer" else wer_v      # metric drives the reward
        rewards.append(_shape_intel_reward(err, reward_shape, reward_k))
        cers.append(cer_v); wers.append(wer_v); hyps.append(hyp)
    return {"reward": rewards, "cer": cers, "wer": wers, "hyp": hyps}


def _wer(hyp: str, ref: str) -> float:
    ref_n, hyp_n = normalize_text(ref).split(), normalize_text(hyp).split()
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    try:
        import jiwer
        return float(jiwer.wer(normalize_text(ref), normalize_text(hyp)))
    except ImportError:
        return _levenshtein(hyp_n, ref_n) / len(ref_n)


@register_reward("length_penalty", default_weight=0.5)
def length_reward(
    ctx: RewardContext,
    *,
    no_eos_penalty:    float = 1.0,   # subtracted if generation hit the token cap
    silence_penalty:   float = 0.5,
    silence_frac_max:  float = 0.6,   # >this fraction of low-energy tail → penalty
    silence_rms_db:    float = -40.0, # frame considered silent below this
    # speaking-rate guard (anti reward-hacking): penalise VOICED speech below
    # `speaking_rate_min_cps` chars/sec, graded by the shortfall. 0 = off; ~8–10
    # for Hindi. Measured over voiced duration (trailing silence excluded).
    speaking_rate_min_cps: float = 0.0,
    speaking_rate_penalty: float = 0.5,
    # graded over-length guard: the dense gradient the binary no_eos cliff lacks.
    # Penalise voiced duration beyond `length_overrun_tol × expected` (expected =
    # chars / length_target_cps), graded ∝ overrun, saturating at `overrun_penalty`
    # at 2×tol×expected. 0 = off; set no_eos_penalty:0 to fully replace the cliff.
    length_target_cps:  float = 0.0,
    length_overrun_tol: float = 1.5,
    overrun_penalty:    float = 0.5,
    frame_ms: float = 20.0,
) -> Dict[str, List[float]]:
    """Degeneracy guard: penalise (a) hitting the token cap without EOS,
    (b) excessive trailing silence (padding with quiet to game the ASR),
    (c) speech slower than `speaking_rate_min_cps` chars/sec (over-enunciation
    that keeps CER low while degrading naturalness), and (d) speech much longer
    than the text warrants — a graded over-length penalty that gives the dense
    gradient the binary cap cliff (a) lacks. Reward ≤ 0 (it is a penalty).

    `ctx.texts` is required only for the speaking-rate and over-length terms;
    without it those terms are skipped.
    """
    audios, gen_lengths = ctx.audios, ctx.gen_lengths
    max_new_tokens, sample_rate = ctx.max_new_tokens, ctx.sample_rate
    rewards, sil_fracs, cps_list = [], [], []
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    thresh = 10.0 ** (silence_rms_db / 20.0)
    texts = ctx.texts if ctx.texts is not None else [None] * len(audios)
    for audio, glen, text in zip(audios, gen_lengths, texts):
        pen = 0.0
        if int(glen) >= max_new_tokens:
            pen -= no_eos_penalty
        wav = np.asarray(audio, dtype=np.float32)
        sil = _trailing_silence_frac(wav, win, thresh)
        if sil > silence_frac_max:
            pen -= silence_penalty

        # Text-relative terms measured over VOICED duration (total minus the
        # trailing-silence tail, so they target articulation/length, not the
        # padding the silence term already covers). cps = chars / voiced_seconds.
        cps = 0.0
        need_text_terms = (speaking_rate_min_cps > 0 or length_target_cps > 0)
        if need_text_terms and text:
            n_chars = len(normalize_text(text).replace(" ", ""))
            voiced_s = max((len(wav) / sample_rate) * (1.0 - sil), 1e-3)
            if n_chars > 0:
                cps = n_chars / voiced_s
                # (c) speaking-rate floor: penalise over-enunciation (too slow).
                if speaking_rate_min_cps > 0:
                    shortfall = (speaking_rate_min_cps - cps) / speaking_rate_min_cps
                    if shortfall > 0:
                        pen -= speaking_rate_penalty * min(1.0, shortfall)
                # (d) graded over-length: penalise voiced duration beyond
                # tol×expected, ramping to `overrun_penalty` at 2×tol×expected.
                if length_target_cps > 0:
                    expected_s = n_chars / length_target_cps
                    # overrun = 1.0 exactly at the tolerance edge, >1 past it.
                    overrun = voiced_s / (expected_s * length_overrun_tol)
                    if overrun > 1.0:
                        pen -= overrun_penalty * min(1.0, overrun - 1.0)
        rewards.append(pen); sil_fracs.append(sil); cps_list.append(cps)
    return {"reward": rewards, "silence_frac": sil_fracs, "speaking_rate": cps_list}


def _trailing_silence_frac(wav: np.ndarray, win: int, thresh: float) -> float:
    if wav.shape[0] < win:
        return 0.0
    n = wav.shape[0] // win
    frames = wav[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    silent = rms < thresh
    # count contiguous silent frames from the end
    trailing = 0
    for s in silent[::-1]:
        if s:
            trailing += 1
        else:
            break
    return trailing / n


@register_reward("speaker_similarity", default_weight=0.0)
def speaker_similarity_reward(ctx: RewardContext) -> Dict[str, List[float]]:
    """r_spk = cosine(speaker_encoder(mel(audio)), speaker_encoder(ref_mel)),
    Pipeline 2 only. Reuses the frozen speaker_encoder already loaded for SFT.
    Reads `ctx.model` (must have `speaker_encoder`) and `ctx.ref_mel`."""
    from data.audio_utils import mel_spectrogram  # repo's 24k mel for Qwen3-TTS

    model, ref_mel, sample_rate = ctx.model, ctx.ref_mel, ctx.sample_rate
    if getattr(model, "speaker_encoder", None) is None:
        raise RuntimeError("speaker_similarity_reward needs model.speaker_encoder (Pipeline 2).")
    if ref_mel is None:
        raise RuntimeError("speaker_similarity_reward needs ctx.ref_mel (Pipeline 2).")
    if ref_mel.ndim == 2:                                           # [T,128] → [1,T,128]
        ref_mel = ref_mel[None, ...]
    ref_vec = mx.stop_gradient(model.speaker_encoder(ref_mel))      # [1, D]
    ref_vec = ref_vec / (mx.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

    rewards = []
    for audio in ctx.audios:
        wav = np.asarray(audio, dtype=np.float32)
        # Degenerate rollouts (near-silent) can make mel_spectrogram overflow →
        # NaN embedding → NaN cosine, which would poison the whole group's
        # advantages. Guard those to the worst similarity (−1) instead.
        if wav.shape[0] < 256 or not np.all(np.isfinite(wav)):
            rewards.append(-1.0)
            continue
        mel = mel_spectrogram(wav, sr=sample_rate)
        if not np.all(np.isfinite(mel)):
            rewards.append(-1.0)
            continue
        vec = mx.stop_gradient(model.speaker_encoder(mx.array(mel)[None, ...]))
        vec = vec / (mx.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)
        sim = float((vec * ref_vec).sum())
        rewards.append(sim if np.isfinite(sim) else -1.0)
    return {"reward": rewards, "spk_sim": rewards}


# ──────────────────────────────────────────────────────────────────────────────
# Naturalness / MOS (DNSMOS) — reference-free quality, the lever beyond CER
# ──────────────────────────────────────────────────────────────────────────────

_DNSMOS = None


def _get_dnsmos():
    """Lazy-load DNSMOS (speechmos + onnxruntime). Cached; raises a clear error if
    the optional backend is missing so `w_mos>0` fails loudly, not silently."""
    global _DNSMOS
    if _DNSMOS is None:
        try:
            from speechmos import dnsmos
        except ImportError as e:
            raise RuntimeError(
                "naturalness reward (w_mos>0) needs DNSMOS: pip install speechmos onnxruntime"
            ) from e
        _DNSMOS = dnsmos
    return _DNSMOS


@register_reward("naturalness", default_weight=0.0)
def naturalness_reward(
    ctx: RewardContext,
    *,
    metric: str = "ovrl",           # ovrl | sig | bak (DNSMOS P.835 sub-score)
) -> Dict[str, List[float]]:
    """r_nat = (DNSMOS − 1) / 4 ∈ [0, 1] per rollout (P.835 MOS ~[1,5] rescaled).

    CER rewards legibility but is blind to quality (a robotic but transcribable
    clip scores well); DNSMOS adds the naturalness axis. It is English-trained, so
    treat it as a RELATIVE proxy for Hindi, not an absolute MOS. Degenerate/short
    rollouts score 0 (worst) rather than poisoning the group with a NaN.

    Returns {"reward": [...0..1...], "mos": [...raw OVRL...]}.
    """
    dnsmos = _get_dnsmos()
    key = f"{metric}_mos"                               # ovrl_mos | sig_mos | bak_mos
    rewards, mos = [], []
    for audio in ctx.audios:
        wav = _to_whisper_audio(audio, ctx.sample_rate)  # DNSMOS wants 16 kHz
        if wav.shape[0] < 256 or not np.all(np.isfinite(wav)):
            rewards.append(0.0); mos.append(1.0); continue
        try:
            mos_score = float(dnsmos.run(wav, sr=WHISPER_SR)[key])
        except Exception:
            rewards.append(0.0); mos.append(1.0); continue
        rewards.append(min(1.0, max(0.0, (mos_score - 1.0) / 4.0)))
        mos.append(mos_score)
    return {"reward": rewards, "mos": mos}


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration + advantages
# ──────────────────────────────────────────────────────────────────────────────

def combine_rewards(
    audios:      List[mx.array],
    texts:       List[str],
    gen_lengths: Sequence[int],
    max_new_tokens: int,
    cfg: RewardConfig,
    *,
    sample_rate: int,
    model=None,
    ref_mel: Optional[mx.array] = None,
) -> Dict[str, object]:
    """Run every reward listed (with weight != 0) in `cfg.rewards`, return the
    weighted per-rollout total + a metrics dict.

    total_i = Σ_name  weight(name) · r_name,i    (over active registered rewards)

    Each active reward contributes `r_<name>` (its per-rollout reward) plus any
    extra info it emits (e.g. intelligibility → `cer`/`hyp`, naturalness → `mos`,
    length → `silence_frac`/`speaking_rate`, speaker → `spk_sim`) into `info`.
    """
    ctx = RewardContext(
        audios=audios, texts=texts, sample_rate=sample_rate,
        gen_lengths=gen_lengths, max_new_tokens=max_new_tokens,
        model=model, ref_mel=ref_mel,
    )
    n = len(audios)
    total = np.zeros(n, dtype=np.float32)
    info: Dict[str, object] = {}

    for name in cfg.rewards:
        w = cfg.weight(name)
        if w == 0:
            continue
        r = _invoke(name, ctx, cfg.spec(name))
        total += w * np.asarray(r["reward"], dtype=np.float32)
        info[f"r_{name}"] = r["reward"]
        for k, v in r.items():
            if k != "reward":
                info[k] = v

    info["reward_total"] = total.tolist()
    return {"reward": total, "info": info}


def group_advantages(
    reward: np.ndarray,
    group_size: int,
    eps: float = 1e-4,
    adv_norm: str = "std",
) -> np.ndarray:
    """Group-relative advantages. `reward` is laid out as contiguous groups of
    `group_size` (all rollouts of prompt 0, then prompt 1…). Returns advantages of
    the same shape.

    adv_norm="std":  A_i = (r_i − mean_group) / (std_group + eps)  (DeepSeek default)
    adv_norm="none": A_i =  r_i − mean_group                       (Dr. GRPO; drops
                     the /std bias, but shrinks |A| ~10× → retune LR up ~5–10×)
    """
    reward = np.asarray(reward, dtype=np.float32)
    assert reward.shape[0] % group_size == 0, "reward length must be a multiple of group_size"
    groups = reward.reshape(-1, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    adv = groups - mean
    if adv_norm == "std":
        adv = adv / (groups.std(axis=1, keepdims=True) + eps)
    elif adv_norm != "none":
        raise ValueError(f"adv_norm must be 'std' or 'none', got {adv_norm!r}")
    return adv.reshape(-1)
