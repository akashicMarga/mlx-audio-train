"""
grpo/eval.py — in-loop fixed-prompt held-out eval for GRPO.

Wired into the base `Trainer.audio_eval_fn` hook (which now fires on its own
`eval_every_n_steps` schedule, independent of a val_loader — GRPO has none). Every
eval window it rolls out a FIXED prompt set with a FIXED seed and logs, to
TensorBoard:

  • cer               — mean CER of the rollouts (the reward signal, on held-out text)
  • duration_s        — mean decoded-audio length (reward-hacking watch: over-long)
  • trailing_silence  — mean trailing-silence fraction (degeneracy watch)
  • kl                — mean k3 KL(π_θ ‖ π_ref) over valid cb0 tokens
  • sample            — one decoded waveform

Fixed seed → the eval prompts produce comparable rollouts across steps, so the
curves reflect policy change, not sampling noise. Unlike the post-hoc held-out
eval, this runs inside the loop, making a run self-documenting. The global MLX
RNG is reseeded with fresh entropy afterwards so training rollouts stay
stochastic (the exact pre-eval stream is not preserved — GRPO only needs
on-policy stochastic sampling, not stream continuity).
"""

import time
from typing import Dict, List

import numpy as np
import mlx.core as mx

from train.grpo.rollout import (
    sample_rollouts,
    sample_rollouts_interleaved,
    decode_codes_to_audio,
    grpo_codec_logits,
    gather_token_logprobs,
    _cb0_logits_from_tf_input,
)
from train.grpo.rewards import intelligibility_reward, _trailing_silence_frac, normalize_text


def make_grpo_audio_eval_fn(
    prompts: List[Dict],
    ref_params: Dict[str, mx.array],
    reward_cfg,
    *,
    layout: str = "interleaved",
    lang_code: str = "auto",
    group_size: int = 2,
    max_new_tokens: int = 240,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    sample_rate: int = 24000,
    seed: int = 12345,
    frame_ms: float = 20.0,
    eval_log: str = None,
):
    """Build an `audio_eval_fn(model, step, tb_writer, reference_only=False)`
    closure over a fixed held-out prompt set. At step 0 `reference_only=True`
    (policy == reference → KL ≈ 0, a sanity baseline).

    `eval_log` (optional): append one JSON line per eval (the same scalars logged
    to TensorBoard). Persisting to disk lets offline consumers — e.g. the
    ablation runner — collect curves without parsing TB event files."""

    win    = max(1, int(sample_rate * frame_ms / 1000.0))
    thresh = 10.0 ** (reward_cfg.silence_rms_db / 20.0)

    def _rollout(model, prompt):
        lang = prompt.get("lang_code", lang_code)
        if layout == "interleaved":
            out = sample_rollouts_interleaved(
                model, prompt["text"], lang_code=lang, group_size=group_size,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, ref_audio=prompt.get("ref_audio"),
                compute_ref=True, ref_params=ref_params,
            )
            T = out["codec_ids"].shape[1]
            logits = _cb0_logits_from_tf_input(
                model, out["tf_input_embeds"], int(out["cb0_offset"]), T)
            policy_logp = gather_token_logprobs(logits, out["codec_ids"])
        else:  # concatenated
            text_ids = prompt["text_ids"]
            text_ids = mx.array(text_ids) if not isinstance(text_ids, mx.array) else text_ids
            spk = prompt.get("spk_embeds")
            spk_b = mx.broadcast_to(spk, (group_size, spk.shape[-1])) if spk is not None else None
            out = sample_rollouts(
                model, text_ids, lang_code=lang, group_size=group_size,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, spk_embeds=spk_b,
                compute_ref=True, ref_params=ref_params,
            )
            tids = mx.broadcast_to(text_ids[None, :], (group_size, text_ids.shape[0]))
            logits, _, _ = grpo_codec_logits(
                model, tids, out["codec_ids"], [lang] * group_size, spk_embeds=spk_b)
            policy_logp = gather_token_logprobs(logits, out["codec_ids"])
        return out, policy_logp

    def _eval(model, step, tb_writer, reference_only: bool = False):
        cers, durs, sils, kls, rates = [], [], [], [], []
        first_audio = None

        for pi, prompt in enumerate(prompts):
            mx.random.seed(seed + pi)                 # comparable rollouts step→step
            out, policy_logp = _rollout(model, prompt)

            audios = decode_codes_to_audio(model, out["full_codes"], out["codec_mask"])
            r = intelligibility_reward(
                audios, [prompt["text"]] * group_size, reward_cfg, sample_rate=sample_rate)
            cers.extend(r["cer"])

            # k3 KL(π_θ ‖ π_ref), per rollout, averaged over valid cb0 tokens.
            mask = out["codec_mask"].astype(mx.float32)
            d   = mx.stop_gradient(out["ref_logprobs"]) - policy_logp
            kl_tok = (mx.exp(d) - d - 1.0) * mask
            kl_per = kl_tok.sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
            mx.eval(kl_per)
            kls.extend(np.asarray(kl_per, dtype=np.float32).tolist())

            n_chars = len(normalize_text(prompt["text"]).replace(" ", ""))
            for a in audios:
                wav = np.asarray(a, dtype=np.float32)
                dur = len(wav) / sample_rate
                sil = _trailing_silence_frac(wav, win, thresh)
                durs.append(dur); sils.append(sil)
                # voiced-duration speaking rate (chars/sec) — the anti-hack watch
                voiced = max(dur * (1.0 - sil), 1e-3)
                if n_chars > 0:
                    rates.append(n_chars / voiced)
            if first_audio is None and audios:
                first_audio = np.asarray(audios[0], dtype=np.float32)

        tag = "grpo_eval_ref" if reference_only else "grpo_eval"
        cer_m, dur_m = float(np.mean(cers)), float(np.mean(durs))
        sil_m, kl_m  = float(np.mean(sils)), float(np.mean(kls))
        rate_m       = float(np.mean(rates)) if rates else 0.0
        tb_writer.add_scalar(f"{tag}/cer",              cer_m, step)
        tb_writer.add_scalar(f"{tag}/duration_s",       dur_m, step)
        tb_writer.add_scalar(f"{tag}/trailing_silence", sil_m, step)
        tb_writer.add_scalar(f"{tag}/speaking_rate",    rate_m, step)
        tb_writer.add_scalar(f"{tag}/kl",               kl_m,  step)
        if first_audio is not None:
            tb_writer.add_audio(f"{tag}/sample", first_audio, step, sample_rate=sample_rate)
        print(f"[grpo-eval] step={step} cer={cer_m:.3f} dur={dur_m:.2f}s "
              f"sil={sil_m:.2f} cps={rate_m:.1f} kl={kl_m:.4f}")

        if eval_log:
            import json, os
            os.makedirs(os.path.dirname(eval_log) or ".", exist_ok=True)
            with open(eval_log, "a") as fh:
                fh.write(json.dumps({
                    "step": step, "reference_only": reference_only,
                    "cer": cer_m, "duration_s": dur_m, "trailing_silence": sil_m,
                    "speaking_rate": rate_m, "kl": kl_m,
                }) + "\n")

        # Restore stochasticity for the training rollouts that follow.
        mx.random.seed(int(time.time() * 1000) & 0xFFFFFFFF)

    return _eval
