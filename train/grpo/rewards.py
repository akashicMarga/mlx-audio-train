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

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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

@dataclass
class RewardConfig:
    # weights (a reward absent from the run contributes 0)
    w_intel:   float = 1.0
    w_length:  float = 0.5
    w_speaker: float = 0.0          # Pipeline 2 only; >0 enables speaker-sim
    w_mos:     float = 0.0          # >0 enables naturalness/MOS (DNSMOS); needs speechmos

    # intelligibility / ASR
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    language:  str = "hi"
    metric:    str = "cer"          # "cer" or "wer"

    # naturalness / MOS (the lever beyond CER-only: CER rewards legibility, not
    # quality). Reference-free DNSMOS P.835; English-trained → a rough RELATIVE
    # proxy for Hindi naturalness, not a native MOS (see docs). 0 = off.
    mos_metric: str = "ovrl"        # ovrl | sig | bak (DNSMOS sub-score)

    # length / degeneracy guard
    no_eos_penalty:    float = 1.0   # subtracted if generation hit the token cap
    silence_frac_max:  float = 0.6   # >this fraction of low-energy tail → penalty
    silence_penalty:   float = 0.5
    silence_rms_db:    float = -40.0 # frame considered silent below this

    # speaking-rate guard (anti reward-hacking): CER rewards ASR-friendly speech,
    # which can degenerate into slow/over-enunciated audio that still transcribes
    # well. Penalise voiced speech below `speaking_rate_min_cps` chars/sec, graded
    # by the shortfall. Disabled by default (min_cps=0) — it is a run-dependent
    # defence; set ~8–10 for Hindi. Measured over VOICED duration (trailing
    # silence excluded so it doesn't double-count the silence penalty).
    speaking_rate_min_cps: float = 0.0   # 0 = off
    speaking_rate_penalty: float = 0.5   # magnitude when fully below the floor

    eps: float = 1e-4               # advantage std floor


# ──────────────────────────────────────────────────────────────────────────────
# Individual rewards
# ──────────────────────────────────────────────────────────────────────────────

def _to_whisper_audio(audio: mx.array, src_sr: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32)
    if src_sr != WHISPER_SR:
        wav = _resample(wav, src_sr, WHISPER_SR)
    return wav


def intelligibility_reward(
    audios: List[mx.array],
    texts:  List[str],
    cfg:    RewardConfig,
    *,
    sample_rate: int,
) -> Dict[str, List[float]]:
    """r_intel = 1 − min(1, error_rate(ASR(audio), text)) per rollout.

    Returns {"reward": [...], "cer": [...], "hyp": [...]} (hyp kept for logging).

    Uses the canonical `mlx_whisper.transcribe` per rollout. (A batched
    `decode()` path was tried — ~3× faster ASR — but ASR is only ~3% of step
    time, rollout sampling dominates, and the batched scorer drifted from
    `transcribe()` even on intelligible audio. Not worth the reward-quality risk.)
    """
    import mlx_whisper

    rewards, errs, hyps = [], [], []
    for audio, text in zip(audios, texts):
        wav = _to_whisper_audio(audio, sample_rate)
        if wav.shape[0] < WHISPER_SR // 10:        # <100 ms → treat as empty
            rewards.append(0.0); errs.append(1.0); hyps.append("")
            continue
        # temperature=0.0 (scalar) disables the 6-way temperature fallback, which
        # otherwise re-decodes degenerate/gibberish rollouts up to 6× (and can
        # hallucinate long repetitive transcripts each pass). Early in GRPO most
        # rollouts ARE degenerate, so this is the dominant cost guard; it also
        # makes the reward deterministic. condition_on_previous_text=False avoids
        # cross-segment drift on the short clips we score.
        result = mlx_whisper.transcribe(
            wav, path_or_hf_repo=cfg.asr_model, language=cfg.language, verbose=False,
            temperature=0.0, condition_on_previous_text=False,
        )
        hyp = result.get("text", "")
        err = char_error_rate(hyp, text) if cfg.metric == "cer" else _wer(hyp, text)
        rewards.append(1.0 - min(1.0, err))
        errs.append(err); hyps.append(hyp)
    return {"reward": rewards, "cer": errs, "hyp": hyps}


def _wer(hyp: str, ref: str) -> float:
    ref_n, hyp_n = normalize_text(ref).split(), normalize_text(hyp).split()
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    try:
        import jiwer
        return float(jiwer.wer(normalize_text(ref), normalize_text(hyp)))
    except ImportError:
        return _levenshtein(hyp_n, ref_n) / len(ref_n)


def length_reward(
    audios:      List[mx.array],
    gen_lengths: Sequence[int],
    max_new_tokens: int,
    cfg: RewardConfig,
    *,
    sample_rate: int,
    texts: Optional[List[str]] = None,
    frame_ms: float = 20.0,
) -> Dict[str, List[float]]:
    """Degeneracy guard: penalise (a) hitting the token cap without EOS,
    (b) excessive trailing silence (padding with quiet to game the ASR), and
    (c) speech slower than `speaking_rate_min_cps` chars/sec (over-enunciation
    that keeps CER low while degrading naturalness). Reward ≤ 0 (it is a penalty).

    `texts` (the reference transcript per rollout) is required only for the
    speaking-rate term; without it that term is skipped.
    """
    rewards, sil_fracs, cps_list = [], [], []
    win = max(1, int(sample_rate * frame_ms / 1000.0))
    thresh = 10.0 ** (cfg.silence_rms_db / 20.0)
    texts = texts if texts is not None else [None] * len(audios)
    for audio, glen, text in zip(audios, gen_lengths, texts):
        pen = 0.0
        if int(glen) >= max_new_tokens:
            pen -= cfg.no_eos_penalty
        wav = np.asarray(audio, dtype=np.float32)
        sil = _trailing_silence_frac(wav, win, thresh)
        if sil > cfg.silence_frac_max:
            pen -= cfg.silence_penalty

        # Speaking rate over VOICED duration (total minus the trailing-silence
        # tail, so this targets slow articulation, not the padding the silence
        # term already covers). cps = chars / voiced_seconds.
        cps = 0.0
        if cfg.speaking_rate_min_cps > 0 and text:
            n_chars = len(normalize_text(text).replace(" ", ""))
            voiced_s = max((len(wav) / sample_rate) * (1.0 - sil), 1e-3)
            if n_chars > 0:
                cps = n_chars / voiced_s
                shortfall = (cfg.speaking_rate_min_cps - cps) / cfg.speaking_rate_min_cps
                if shortfall > 0:
                    pen -= cfg.speaking_rate_penalty * min(1.0, shortfall)
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


def speaker_similarity_reward(
    model,
    audios: List[mx.array],
    ref_mel: mx.array,
    *,
    sample_rate: int,
) -> Dict[str, List[float]]:
    """r_spk = cosine(speaker_encoder(mel(audio)), speaker_encoder(ref_mel)),
    Pipeline 2 only. Reuses the frozen speaker_encoder already loaded for SFT."""
    from data.audio_utils import mel_spectrogram  # repo's 24k mel for Qwen3-TTS

    if getattr(model, "speaker_encoder", None) is None:
        raise RuntimeError("speaker_similarity_reward needs model.speaker_encoder (Pipeline 2).")
    if ref_mel.ndim == 2:                                           # [T,128] → [1,T,128]
        ref_mel = ref_mel[None, ...]
    ref_vec = mx.stop_gradient(model.speaker_encoder(ref_mel))      # [1, D]
    ref_vec = ref_vec / (mx.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

    rewards = []
    for audio in audios:
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
    return {"reward": rewards}


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


def naturalness_reward(
    audios: List[mx.array],
    cfg:    RewardConfig,
    *,
    sample_rate: int,
) -> Dict[str, List[float]]:
    """r_nat = (DNSMOS − 1) / 4 ∈ [0, 1] per rollout (P.835 MOS ~[1,5] rescaled).

    CER rewards legibility but is blind to quality (a robotic but transcribable
    clip scores well); DNSMOS adds the naturalness axis. It is English-trained, so
    treat it as a RELATIVE proxy for Hindi, not an absolute MOS. Degenerate/short
    rollouts score 0 (worst) rather than poisoning the group with a NaN.

    Returns {"reward": [...0..1...], "mos": [...raw OVRL...]}.
    """
    dnsmos = _get_dnsmos()
    key = f"{cfg.mos_metric}_mos"                       # ovrl_mos | sig_mos | bak_mos
    rewards, mos = [], []
    for audio in audios:
        wav = _to_whisper_audio(audio, sample_rate)     # DNSMOS wants 16 kHz
        if wav.shape[0] < 256 or not np.all(np.isfinite(wav)):
            rewards.append(0.0); mos.append(1.0); continue
        try:
            score = float(dnsmos.run(wav, sr=WHISPER_SR)[key])
        except Exception:
            rewards.append(0.0); mos.append(1.0); continue
        rewards.append(min(1.0, max(0.0, (score - 1.0) / 4.0)))
        mos.append(score)
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
    """Run the enabled rewards, return total per-rollout reward + a metrics dict.

    total_i = w_intel·r_intel + w_length·r_length + w_speaker·r_spk + w_mos·r_mos
    """
    n = len(audios)
    total = np.zeros(n, dtype=np.float32)
    info: Dict[str, object] = {}

    if cfg.w_intel > 0:
        r = intelligibility_reward(audios, texts, cfg, sample_rate=sample_rate)
        total += cfg.w_intel * np.asarray(r["reward"], dtype=np.float32)
        info["cer"] = r["cer"]; info["hyp"] = r["hyp"]
        info["r_intel"] = r["reward"]

    if cfg.w_length > 0:
        r = length_reward(audios, gen_lengths, max_new_tokens, cfg,
                          sample_rate=sample_rate, texts=texts)
        total += cfg.w_length * np.asarray(r["reward"], dtype=np.float32)
        info["silence_frac"] = r["silence_frac"]; info["r_length"] = r["reward"]
        info["speaking_rate"] = r["speaking_rate"]

    if cfg.w_speaker > 0:
        r = speaker_similarity_reward(model, audios, ref_mel, sample_rate=sample_rate)
        total += cfg.w_speaker * np.asarray(r["reward"], dtype=np.float32)
        info["r_speaker"] = r["reward"]

    if cfg.w_mos > 0:
        r = naturalness_reward(audios, cfg, sample_rate=sample_rate)
        total += cfg.w_mos * np.asarray(r["reward"], dtype=np.float32)
        info["mos"] = r["mos"]; info["r_mos"] = r["reward"]

    info["reward_total"] = total.tolist()
    return {"reward": total, "info": info}


def group_advantages(reward: np.ndarray, group_size: int, eps: float = 1e-4) -> np.ndarray:
    """Group-relative, std-normalised advantages. `reward` is laid out as
    contiguous groups of `group_size` (all rollouts of prompt 0, then prompt 1…).
    Returns advantages of the same shape."""
    reward = np.asarray(reward, dtype=np.float32)
    assert reward.shape[0] % group_size == 0, "reward length must be a multiple of group_size"
    groups = reward.reshape(-1, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1)
