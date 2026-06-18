"""
grpo/trainer.py — Phase A↔B orchestration for GRPO.

The base `train/trainer.py:Trainer` already gives us grad-accumulation, the
LoRA-only `value_and_grad` (+ `_strip_empty`), grad clipping, LR schedule,
checkpointing, TensorBoard/JSONL logging, and it auto-logs every metric the loss
returns. GRPO reuses all of it — we only swap the *data*:

  • `GRPORolloutLoader` — an iterable that, per pull, runs Phase A for ONE prompt
    (sample G rollouts → decode → reward → group advantages) and yields a Phase-B
    batch dict. Sampling uses the live model, so pulls during one accumulation
    window are on-policy w.r.t. the pre-step weights.
  • `GRPOTrainer` — thin `Trainer` subclass that snapshots the SFT reference once
    (`lora_snapshot`) and drives `Trainer.train()` with `qwen3_tts_grpo_loss`.

Set `grad_accumulation = prompts_per_step` (B): one optimizer step == B prompt
groups == B·G rollouts, the standard strictly-on-policy GRPO update.
"""

import random
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import mlx.core as mx

from train.trainer import Trainer, TrainerConfig
from train.lora import lora_snapshot
from train.grpo.rollout import sample_rollouts, sample_rollouts_interleaved, decode_codes_to_audio
from train.grpo.rewards import RewardConfig, combine_rewards, group_advantages
from train.losses.grpo_loss import qwen3_tts_grpo_loss


# ──────────────────────────────────────────────────────────────────────────────
# Rollout loader (Phase A)
# ──────────────────────────────────────────────────────────────────────────────

class GRPORolloutLoader:
    """Yields Phase-B GRPO batches, one prompt group per pull.

    `prompts` is a list of dicts, each at least:
        {"text_ids": mx.array|list[int], "text": str,   # text for ASR reference
         "lang_code": "hi",                              # optional, default loader.lang_code
         "spk_embeds": mx.array[1,D], "ref_mel": mx.array}  # optional, Pipeline 2

    Reshuffles each epoch. `__len__` is the prompt count, so the base Trainer's
    steps_per_epoch = len(prompts) // grad_accumulation.
    """

    def __init__(
        self,
        model,
        prompts: List[Dict],
        ref_params: Dict[str, mx.array],
        reward_cfg: RewardConfig,
        *,
        group_size: int = 4,
        max_new_tokens: int = 240,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        lang_code: str = "auto",
        layout: str = "interleaved",        # "interleaved" (matches generate()) | "concatenated"
        sample_rate: int = 24000,
        sft_lambda: float = 0.0,
        sft_loader=None,
        skip_zero_variance: bool = True,
        zero_var_eps: float = 1e-6,
        seed: int = 0,
    ):
        self.model = model
        self.prompts = prompts
        self.ref_params = ref_params
        self.reward_cfg = reward_cfg
        self.G = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.lang_code = lang_code
        self.layout = layout
        self.sample_rate = sample_rate
        self.sft_lambda = sft_lambda
        self.sft_loader = sft_loader
        self.skip_zero_variance = skip_zero_variance
        self.zero_var_eps = zero_var_eps
        self._sft_iter = iter(sft_loader) if sft_loader is not None else None
        self._rng = random.Random(seed)
        self._n_skipped = 0
        # Per-logging-window counters drained by the base Trainer via
        # pop_window_metrics() — surfaces the skip RATE per window, not just a
        # single end-of-run total (a high rate = sparse PG signal even when the
        # run "looks alive").
        self._window_skipped = 0
        self._window_built   = 0

    def __len__(self) -> int:
        return len(self.prompts)

    def pop_window_metrics(self) -> Dict[str, float]:
        """Return + reset this window's rollout-group accounting.

        skip_ratio = skipped / (skipped + built) over the window. Called by the
        base Trainer at each `log_every_n_steps` boundary, so the window aligns
        with the logging cadence.
        """
        s, b = self._window_skipped, self._window_built
        self._window_skipped = 0
        self._window_built   = 0
        total = s + b
        return {
            "skip_ratio": (s / total) if total else 0.0,
            "skipped":    float(s),
            "built":      float(b),
        }

    def _next_sft_batch(self):
        if self._sft_iter is None:
            return None
        try:
            return next(self._sft_iter)
        except StopIteration:
            self._sft_iter = iter(self.sft_loader)
            return next(self._sft_iter)

    def _build_batch(self, prompt: Dict) -> Dict:
        G = self.G
        lang = prompt.get("lang_code", self.lang_code)

        # ── Phase A: sample (per layout), decode, score ─────────────────────
        if self.layout == "interleaved":
            out = sample_rollouts_interleaved(
                self.model, prompt["text"], lang_code=lang, group_size=G,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k, ref_audio=prompt.get("ref_audio"),
                compute_ref=True, ref_params=self.ref_params,
            )
            spk_batched = None
            text_ids = None
        else:  # concatenated (SFT-matched)
            text_ids = prompt["text_ids"]
            text_ids = mx.array(text_ids) if not isinstance(text_ids, mx.array) else text_ids
            spk = prompt.get("spk_embeds")
            spk_batched = mx.broadcast_to(spk, (G, spk.shape[-1])) if spk is not None else None
            out = sample_rollouts(
                self.model, text_ids, lang_code=lang, group_size=G,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k, spk_embeds=spk_batched,
                compute_ref=True, ref_params=self.ref_params,
            )

        audios = decode_codes_to_audio(self.model, out["full_codes"], out["codec_mask"])
        scored = combine_rewards(
            audios, [prompt["text"]] * G, out["gen_lengths"].tolist(),
            self.max_new_tokens, self.reward_cfg, sample_rate=self.sample_rate,
            model=self.model, ref_mel=prompt.get("ref_mel"),
        )
        reward = np.asarray(scored["reward"], dtype=np.float32)
        # Zero-variance group: every rollout drew the same reward → advantages are
        # all 0 → no policy-gradient signal (only KL would act). Standard GRPO
        # practice (DeepSeekMath) drops these so each step accumulates only groups
        # that actually teach something. Phase A is already spent, but we save the
        # Phase-B forward/backward and avoid diluting the effective batch.
        if self.skip_zero_variance and float(reward.std()) < self.zero_var_eps:
            self._n_skipped += 1
            self._window_skipped += 1
            return None

        adv = group_advantages(reward, group_size=G, eps=self.reward_cfg.eps)
        info = scored["info"]
        cer_mean = float(np.mean(info["cer"])) if "cer" in info else 0.0
        mos_mean = float(np.mean(info["mos"])) if "mos" in info else 0.0

        batch = {
            "codec_ids":    out["codec_ids"],
            "codec_mask":   out["codec_mask"],
            "advantages":   mx.array(adv),
            "ref_logprobs": out["ref_logprobs"],
            "reward_mean":  float(np.mean(scored["reward"])),
            "cer_mean":     cer_mean,
            "mos_mean":     mos_mean,
        }
        if self.layout == "interleaved":
            batch["tf_input_embeds"] = out["tf_input_embeds"]
            batch["cb0_offset"]      = out["cb0_offset"]
        else:
            batch["text_ids"]   = mx.broadcast_to(text_ids[None, :], (G, text_ids.shape[0]))
            batch["lang_codes"] = [lang] * G
            if spk_batched is not None:
                batch["spk_embeds"] = spk_batched
        if self.sft_lambda > 0:
            batch["sft_batch"] = self._next_sft_batch()
        self._window_built += 1
        return batch

    def __iter__(self):
        order = list(range(len(self.prompts)))
        self._rng.shuffle(order)
        for i in order:
            batch = self._build_batch(self.prompts[i])
            if batch is None:        # zero-variance group → skipped
                continue
            yield batch


# ──────────────────────────────────────────────────────────────────────────────
# GRPO trainer
# ──────────────────────────────────────────────────────────────────────────────

class GRPOTrainer(Trainer):
    """`Trainer` that snapshots the SFT reference once, then runs GRPO."""

    def train_grpo(
        self,
        model,
        prompts: List[Dict],
        *,
        reward_cfg: RewardConfig,
        group_size: int = 4,
        max_new_tokens: int = 240,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        lang_code: str = "auto",
        layout: str = "interleaved",
        kl_beta: float = 0.05,
        kl_clip: float = 10.0,
        sub_pg_weight: float = 0.0,
        pg_norm: str = "token",
        sft_lambda: float = 0.0,
        sft_loader=None,
        skip_zero_variance: bool = True,
        sample_rate: int = 24000,
        eval_prompts: Optional[List[Dict]] = None,
        eval_group_size: int = 2,
        seed: int = 0,
    ):
        # Freeze the reference = current (SFT) adapter, once. KL anchors to this.
        ref_params = lora_snapshot(model)
        print(f"[grpo] snapshotted reference adapter ({len(ref_params)} tensors)  layout={layout}")

        # In-loop fixed-prompt held-out eval (legibility). Needs TensorBoard for
        # the scalar/audio sink; fires on the base Trainer's eval_every_n_steps.
        if eval_prompts and self._tb_writer is not None:
            from train.grpo.eval import make_grpo_audio_eval_fn
            self._audio_eval_fn = make_grpo_audio_eval_fn(
                eval_prompts, ref_params, reward_cfg,
                layout=layout, lang_code=lang_code, group_size=eval_group_size,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, sample_rate=sample_rate,
                eval_log=str(self.output_dir / "grpo_eval.jsonl"),
            )
            print(f"[grpo] in-loop eval on {len(eval_prompts)} fixed prompts "
                  f"every {self.cfg.eval_every_n_steps} steps")
        elif eval_prompts:
            print("[grpo] eval_prompts set but no tensorboard_dir — in-loop eval disabled")

        loader = GRPORolloutLoader(
            model, prompts, ref_params, reward_cfg,
            group_size=group_size, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, lang_code=lang_code,
            layout=layout, sample_rate=sample_rate, sft_lambda=sft_lambda,
            sft_loader=sft_loader, skip_zero_variance=skip_zero_variance, seed=seed,
        )

        loss_fn: Callable = partial(
            qwen3_tts_grpo_loss,
            kl_beta=kl_beta, kl_clip=kl_clip, sub_pg_weight=sub_pg_weight,
            pg_norm=pg_norm, sft_lambda=sft_lambda, lang_code=lang_code,
        )

        # Reuse the whole base loop (grad accum = prompts_per_step, logging, ckpt).
        self.train(model, loader, loss_fn)

        if loader._n_skipped:
            print(f"[grpo] skipped {loader._n_skipped} zero-variance groups "
                  f"(no reward spread → no PG signal)")
