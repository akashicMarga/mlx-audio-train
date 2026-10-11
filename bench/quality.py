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


def paired(cur: List[dict], parent: List[dict], field: str):
    """Aligned (cur, parent) arrays over items present in both runs."""
    p = {item_key(r["seed"], r["prompt_idx"]): r for r in parent}
    a, b = [], []
    for r in cur:
        q = p.get(item_key(r["seed"], r["prompt_idx"]))
        if q is not None and r.get(field) is not None and q.get(field) is not None:
            a.append(r[field])
            b.append(q[field])
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def wilcoxon_p(cur: List[dict], parent: List[dict], field: str = "cer_capped") -> Optional[float]:
    a, b = paired(cur, parent, field)
    if a.size < 6:          # too few pairs for a meaningful signed-rank test
        return None
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


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
    max_dnsmos_drop: Optional[float] = None
    max_spk_drop: Optional[float] = None


def propose(cur_summary: Dict, parent_summary: Optional[Dict], cur_items, parent_items,
            gate: GateConfig) -> (str, str):
    """Return (decision, reason). Decision is a proposal; a human can override."""
    if parent_summary is None:
        return "baseline", "no parent run: this row is a baseline"

    speedup = (parent_summary["rtf"] / cur_summary["rtf"]
               if cur_summary.get("rtf") and parent_summary.get("rtf") else None)
    exact = bit_exact(cur_items, parent_items)

    quality_ok, why = True, []
    if exact:
        why.append("bit-identical to parent")
    else:
        a, b = paired(cur_items, parent_items, "cer_capped")
        if a.size == 0:
            return "", "output differs from parent and quality was not measured — no decision"
        p = wilcoxon_p(cur_items, parent_items)
        if a.size and p is not None and a.mean() > b.mean() and p < gate.alpha:
            quality_ok = False
            why.append(f"CER worse ({b.mean():.4f}→{a.mean():.4f}, Wilcoxon p={p:.2g})")
        else:
            why.append(f"CER not significantly worse (p={p})")
        # Outputs differ, so seed-matched items are not true pairs: compare the two
        # distributions (Mann-Whitney). A drop counts only if it exceeds the threshold
        # AND is significant — with ~15 clips, seed-to-seed swings in mean DNSMOS
        # (~0.1) exceed a 0.05 threshold on their own.
        for field, limit in (("dnsmos_ovrl", gate.max_dnsmos_drop), ("spk_sim", gate.max_spk_drop)):
            if limit is None:
                continue
            x = np.asarray([r[field] for r in cur_items if r.get(field) is not None], dtype=float)
            y = np.asarray([r[field] for r in parent_items if r.get(field) is not None], dtype=float)
            if not (x.size and y.size):
                continue
            drop = y.mean() - x.mean()
            from scipy.stats import mannwhitneyu
            p_mw = float(mannwhitneyu(x, y, alternative="less").pvalue)
            if drop > limit and p_mw < gate.alpha:
                quality_ok = False
                why.append(f"{field} dropped {drop:.3f} > {limit} (Mann-Whitney p={p_mw:.2g}, n={x.size}/{y.size})")
            elif drop > limit:
                why.append(f"{field} dropped {drop:.3f} > {limit} but not significant (p={p_mw:.2g}, n={x.size}/{y.size})")

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
