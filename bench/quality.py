"""
Quality gate: per-item scores, paired statistics vs the parent run, and the
accept / reject / no-benefit proposal.

Scorers are reused from train/grpo/rewards.py so CER, DNSMOS and speaker
similarity are computed exactly as in the GRPO post (whisper CER capped at 1.0).
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def audio_hash(wav: np.ndarray) -> str:
    """Hash of the raw float32 waveform: equal hashes == bit-identical output."""
    return hashlib.sha1(np.ascontiguousarray(wav, dtype=np.float32).tobytes()).hexdigest()


def item_key(seed: int, prompt_idx: int) -> str:
    return f"{seed}:{prompt_idx}"


def score_items(model, items: List[dict], reward_cfg, sample_rate: int,
                with_dnsmos: bool, with_spk: bool) -> None:
    """Fill item['cer_capped'], ['dnsmos_ovrl'], ['spk_sim'] in place."""
    import mlx.core as mx
    from train.grpo.rewards import RewardContext, score

    for it in items:
        ctx = RewardContext(audios=[mx.array(it["_wav"])], texts=[it["text"]],
                            sample_rate=sample_rate, model=model,
                            ref_mel=it.get("_ref_mel"))
        intel = score("intelligibility", ctx, reward_cfg)
        it["cer_capped"] = min(1.0, float(intel["cer"][0]))
        if with_dnsmos:
            it["dnsmos_ovrl"] = float(score("naturalness", ctx, reward_cfg)["mos"][0])
        if with_spk and it.get("_ref_mel") is not None:
            it["spk_sim"] = float(score("speaker_similarity", ctx, reward_cfg)["reward"][0])


def sentence_means(items: List[dict], field: str) -> Dict[int, float]:
    """Per-sentence mean over seeds. Sentences, not clips, are the independent unit:
    300 clips from 100 sentences × 3 seeds are not 300 independent observations, and
    once outputs diverge, seed-matched clips are not true pairs either."""
    acc: Dict[int, list] = {}
    for r in items:
        if r.get(field) is not None:
            acc.setdefault(r["prompt_idx"], []).append(float(r[field]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(cur: List[dict], parent: List[dict], field: str):
    """Aligned (cur, parent) per-sentence means over sentences present in both runs."""
    a, b = sentence_means(cur, field), sentence_means(parent, field)
    keys = sorted(set(a) & set(b))
    return (np.asarray([a[k] for k in keys], dtype=float),
            np.asarray([b[k] for k in keys], dtype=float))


def wilcoxon_p(cur: List[dict], parent: List[dict], field: str = "cer_capped") -> Optional[float]:
    """Paired signed-rank test on per-sentence means (two-sided)."""
    a, b = paired(cur, parent, field)
    if a.size < 6:          # too few sentences for a meaningful signed-rank test
        return None
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def clustered_ci(cur: List[dict], parent: List[dict], field: str, level: float = 0.95,
                 n_boot: int = 10000, seed: int = 0):
    """Mean per-sentence difference (cur − parent) with a bootstrap CI over sentences.
    Returns (mean, lo, hi, n_sentences) or None if fewer than 6 sentences overlap."""
    a, b = paired(cur, parent, field)
    if a.size < 6:
        return None
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(d.mean()), float(lo), float(hi), int(d.size)


def _wall_speedup(cur: List[dict], parent: List[dict]) -> Optional[float]:
    """Parent ÷ current total wall time over the sentences both runs generated."""
    a, b = paired(cur, parent, "wall_s")
    if a.size == 0 or a.sum() <= 0:
        return None
    return float(b.sum() / a.sum())


def bit_exact(cur: List[dict], parent: List[dict]) -> Optional[bool]:
    p = {item_key(r["seed"], r["prompt_idx"]): r.get("audio_sha1") for r in parent}
    common = [r for r in cur if item_key(r["seed"], r["prompt_idx"]) in p]
    if not common:
        return None
    return all(r.get("audio_sha1") == p[item_key(r["seed"], r["prompt_idx"])] for r in common)


@dataclass
class GateConfig:
    alpha: float = 0.05
    min_speedup: float = 1.03          # below this the change is "no-benefit"
    max_dnsmos_drop: Optional[float] = None   # non-inferiority margin for DNSMOS OVRL (e.g. 0.05)
    max_spk_drop: Optional[float] = None      # non-inferiority margin for speaker similarity
    cer_margin: float = 0.02           # non-inferiority margin for capped CER (absolute)
    len_margin: float = 0.03           # equivalence margin for per-sentence audio length (relative)


def propose(cur_summary: Dict, parent_summary: Optional[Dict], cur_items, parent_items,
            gate: GateConfig) -> (str, str):
    """Return (decision, reason). Decision is a proposal; a human can override."""
    if parent_summary is None:
        return "baseline", "no parent run: this row is a baseline"

    speedup = (parent_summary["rtf"] / cur_summary["rtf"]
               if cur_summary.get("rtf") and parent_summary.get("rtf") else None)
    # RTF is per second of *output*: a lever that makes the model say the same text in
    # more audio looks faster without being faster. Judge speed on wall-clock time per
    # sentence too, and take the smaller of the two.
    wall_speedup = _wall_speedup(cur_items, parent_items)
    if speedup is not None and wall_speedup is not None:
        speedup = min(speedup, wall_speedup)
    exact = bit_exact(cur_items, parent_items)

    quality_ok, why = True, []
    if exact:
        why.append("bit-identical to parent")
    else:
        # Outputs differ: judge quality per sentence (seeds averaged), and require
        # non-inferiority — the CI's worst case must sit inside the margin — rather than
        # merely "not significantly worse", which a small sample passes by default.
        ci = clustered_ci(cur_items, parent_items, "cer_capped")
        if ci is None:
            return "", "output differs from parent and quality was not measured — no decision"
        m, lo, hi, n = ci
        p = wilcoxon_p(cur_items, parent_items)
        desc = f"ΔCER {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}], {n} sentences, Wilcoxon p={p:.2g}"
        significant = lo > 0 and p is not None and p < gate.alpha
        if hi > gate.cer_margin:
            quality_ok = False
            why.append(f"CER worse: {desc}" if significant else
                       f"CER non-inferiority not established (upper bound > +{gate.cer_margin}): {desc}")
        else:
            why.append(f"CER non-inferior within +{gate.cer_margin}"
                       + (" (small but significant increase)" if significant else "") + f": {desc}")
        ci_len = clustered_ci(cur_items, parent_items, "audio_s")
        if ci_len is not None:
            ml, ll, hl, nl = ci_len
            base_len = float(np.mean(list(sentence_means(parent_items, "audio_s").values())))
            rel = (ml / base_len, ll / base_len, hl / base_len)
            dl = (f"audio length {100 * rel[0]:+.1f}% [95% CI {100 * rel[1]:+.1f}%, {100 * rel[2]:+.1f}%] "
                  f"per sentence")
            if max(abs(rel[1]), abs(rel[2])) > gate.len_margin:
                quality_ok = False
                why.append(f"{dl} — length equivalence not established (±{100 * gate.len_margin:.0f}%)")
            else:
                why.append(dl)
        for field, margin in (("dnsmos_ovrl", gate.max_dnsmos_drop), ("spk_sim", gate.max_spk_drop)):
            if margin is None:
                continue
            ci_f = clustered_ci(cur_items, parent_items, field)
            if ci_f is None:
                continue
            m2, lo2, hi2, n2 = ci_f
            d2 = f"Δ{field} {m2:+.3f} [95% CI {lo2:+.3f}, {hi2:+.3f}], {n2} sentences"
            if lo2 < -margin:
                quality_ok = False
                why.append(f"{field} non-inferiority not established (lower bound < −{margin}): {d2}")
            else:
                why.append(f"{field} non-inferior within −{margin}: {d2}")

    if not quality_ok:
        return "rejected", "; ".join(why)
    # A speedup only counts if it clears both the fixed threshold and the measured
    # run-to-run noise (relative RTF spread of either run's repeats).
    noise = max(cur_summary.get("rtf_spread") or 0.0, parent_summary.get("rtf_spread") or 0.0)
    need = max(gate.min_speedup, 1.0 + noise)
    if speedup is None or speedup < need:
        sp = "n/a" if speedup is None else f"{speedup:.3f}×"
        return "no-benefit", f"speedup {sp} < required {need:.3f}× (noise {100 * noise:.1f}%); " + "; ".join(why)
    return "accepted", f"speedup {speedup:.3f}× ≥ {need:.3f}× (noise {100 * noise:.1f}%); " + "; ".join(why)
