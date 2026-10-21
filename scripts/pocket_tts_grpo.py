#!/usr/bin/env python3
"""
GRPO post-training for Pocket TTS in MLX: optimise WER/CER, UTMOS / DNSMOS and speaker
similarity directly on a trained model (typically a distilled student), reusing the
reward stack of the Qwen3-TTS GRPO (train/grpo/rewards.py). See models/pocket_tts/grpo.py
for the formulation and docs/pocket_tts_distillation.md for the experiment plan.

Each step: for prompts_per_step prompts, roll out group_size samples (exploration noise
sigma around the head output, stochastic stop), decode with the official Mimi, score,
take group-relative advantages; then one on-policy update over all groups
(advantage-weighted log-likelihood + KL to a frozen snapshot of the starting model +
optional anchor loss on real data).

Usage:
    python scripts/pocket_tts_grpo.py --smoke-test
    python scripts/pocket_tts_grpo.py --config configs/pocket_tts/hindi/grpo_student_4l.yaml
    python scripts/pocket_tts_grpo.py --config ... --resume checkpoints/.../step_0000100
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_map

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.config import flowlm_args_from_raw  # noqa: E402
from models.pocket_tts.data import LatentDataset, batches, load_tokenizer  # noqa: E402
from models.pocket_tts.distill import distill_loss, teacher_target, trainable_count  # noqa: E402
from models.pocket_tts.grpo import (  # noqa: E402
    Prompts, RolloutConfig, Scorer, evaluate_prompts, grpo_loss, make_group_batch, numeric_info,
    rollout_group, snapshot,
)
from models.pocket_tts.model import FlowLM  # noqa: E402
from models.pocket_tts.text import normalizer  # noqa: E402
from models.pocket_tts.train_utils import (  # noqa: E402
    JsonlLog, build_optimizer, load_checkpoint, merge, normalize, save_checkpoint,
    synthetic_manifest, tiny_raw,
)
from models.pocket_tts.weights import build_flow_lm, model_state, weights_from  # noqa: E402
from train.grpo.rewards import RewardConfig, group_advantages, register_reward  # noqa: E402

DEFAULTS = {
    "model": {"freeze_head": True, "train_eos_head": True},
    "grpo": {"group_size": 4, "prompts_per_step": 4, "temp": 0.3, "sigma": 0.1, "max_sec": 20.0,
             "min_frames": 6, "eos_threshold": -1.0, "eos_temp": 1.0, "train_eos": True,
             "kl_beta": 0.02, "pg_norm": "token", "adv_norm": "std", "eps": 1e-4,
             "skip_zero_variance": True, "zero_var_eps": 1e-6, "rewards": {}},
    "anchor": {"type": "none", "weight": 0.0, "batch_size": 8, "cfg_coef": 1.0},
    "data": {"max_voice_prompt_sec": 5.0, "prompt_trim_max_sec": 0.0, "max_prompt_sec": 15.0,
             "normalize_text": None, "skip_unscorable": True},
    "eval": {"num_prompts": 16, "every": 25, "seeds": [0]},
    "optim": {"lr": 1e-6, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8, "max_norm": 1.0,
              "warmup_steps": 10, "schedule": "constant", "lr_min_ratio": 0.0},
    "train": {"max_steps": 400, "log_freq": 1, "ckpt_freq": 50, "keep_last_n": 3, "seed": 0,
              "run_dir": "checkpoints/pocket-grpo"},
}


# ── Smoke test: no downloads, a fake decoder and a reward with a known optimum ─

@register_reward("smoke_target", default_weight=0.0)
def _smoke_reward(ctx, *, target: float = 1.0):
    audios = [np.asarray(a, dtype=np.float32) for a in ctx.audios]
    return {"reward": [-abs(float(a.mean()) - target) if a.size else -10.0 for a in audios]}


def smoke_setup(tmp: Path) -> dict:
    raw = tiny_raw(layers=2, d_model=64, z_dim=None)
    (tmp / "model.yaml").write_text(yaml.safe_dump(raw))
    mx.random.seed(0)
    m = FlowLM(flowlm_args_from_raw(raw))
    tensors = {f"flow_lm.{k}": v for k, v in model_state(m).items()}
    tensors["mimi.placeholder"] = mx.zeros((1,))
    mx.save_safetensors(str(tmp / "model.safetensors"), tensors)
    manifest = synthetic_manifest(32)
    return {
        "model": {"config": str(tmp / "model.yaml"), "weights": str(tmp / "model.safetensors"),
                  "freeze_head": False},
        "grpo": {"group_size": 4, "prompts_per_step": 4, "sigma": 0.3, "max_sec": 2.0, "kl_beta": 0.0,
                 "train_eos": False, "eos_threshold": 1e9,
                 "rewards": {"smoke_target": {"weight": 1.0, "target": 0.5}}},
        "data": {"train": manifest, "valid": manifest[:4], "tokenizer": lambda s: [ord(c) % 64 for c in s],
                 "skip_unscorable": False},
        "eval": {"num_prompts": 4, "every": 20},
        "optim": {"lr": 3e-3, "warmup_steps": 1},
        "train": {"max_steps": 60, "log_freq": 10, "ckpt_freq": 60, "run_dir": str(tmp / "run")},
        "_decode": lambda lat: np.repeat(lat.mean(-1), 8),    # fake "audio": per-frame latent mean
    }


def build_anchor(A: dict, d: dict, t: dict, tok, norm, ds_kw: dict):
    """(iterator of real batches, loss fn) for the anchor term, or (None, None)."""
    if A["type"] == "none" or A["weight"] <= 0:
        return None, None
    ds = LatentDataset(d["train"], (lambda s: tok(norm(s))) if norm else tok, seed=t["seed"] + 1, **ds_kw)
    it = batches(ds, A["batch_size"], 1, seed=t["seed"] + 1)
    if A["type"] == "distill":
        ta = A["teacher"]
        teacher, _ = build_flow_lm(ta["config"], ta.get("overrides"), ta["weights"],
                                   dtype=getattr(mx, ta.get("dtype", "float32")))
        teacher.freeze()

        def fn(model, batch):
            b = normalize(batch, teacher)
            return distill_loss(model, b, teacher_target(teacher, b, float(A["cfg_coef"])))
        return it, fn
    if A["type"] == "lsd":
        from models.pocket_tts.objectives import LSD, flow_lm_loss

        lsd = LSD(normalize=False)

        def fn(model, batch):
            return flow_lm_loss(model, lsd, normalize(batch, model), text_dropout=0.0,
                                voice_dropout=0.0, flow_batch_multiplier=1)[0]
        return it, fn
    raise ValueError(f"anchor.type must be none | distill | lsd, not {A['type']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    tmp = None
    if args.smoke_test:
        tmp = Path(tempfile.mkdtemp(prefix="pocket_grpo_smoke_"))
        user = smoke_setup(tmp)
    else:
        if not args.config:
            ap.error("--config is required (or --smoke-test)")
        user = yaml.safe_load(Path(args.config).read_text())
    decode_override = user.pop("_decode", None)
    cfg = merge(DEFAULTS, user)
    g, d, t, A, E = cfg["grpo"], cfg["data"], cfg["train"], cfg["anchor"], cfg["eval"]
    if args.max_steps:
        t["max_steps"] = args.max_steps
    mx.random.seed(t["seed"])
    np.random.seed(t["seed"])

    # ── Policy and its frozen reference ───────────────────────────────────────
    m = cfg["model"]
    policy, raw = build_flow_lm(m["config"], m.get("overrides"), m["weights"])
    weights = weights_from(m["weights"], raw)
    if m["freeze_head"]:
        policy.flow_net.freeze()
    if not m["train_eos_head"]:
        policy.out_eos.freeze()
    ref = snapshot(policy)
    rc = RolloutConfig(group_size=g["group_size"], temp=g["temp"], sigma=g["sigma"],
                       max_frames=int(g["max_sec"] * 12.5), min_frames=g["min_frames"],
                       eos_threshold=g["eos_threshold"], eos_temp=g["eos_temp"], train_eos=g["train_eos"])
    reward_cfg = RewardConfig(rewards=g["rewards"], adv_norm=g["adv_norm"], eps=float(g["eps"]))
    print(f"[grpo] policy {policy.args.num_layers}L d={policy.args.d_model}, trainable "
          f"{trainable_count(policy) / 1e6:.2f}M; G={rc.group_size} x {g['prompts_per_step']} prompts/step, "
          f"sigma={rc.sigma}, rewards={[k for k in g['rewards'] if reward_cfg.weight(k)]}")

    # ── Data, decoder, scorer ─────────────────────────────────────────────────
    tok = d.get("tokenizer") or load_tokenizer(d.get("tokenizer_path") or policy.args.tokenizer_path)
    norm = normalizer(d.get("normalize_text"))
    ds_kw = dict(max_voice_prompt_sec=d["max_voice_prompt_sec"], prompt_trim_max_sec=d["prompt_trim_max_sec"],
                 final_punct_dropout=0.0)
    p_kw = dict(skip_unscorable=d["skip_unscorable"], max_prompt_sec=d["max_prompt_sec"])
    train_prompts = iter(Prompts(LatentDataset(d["train"], tok, seed=t["seed"], **ds_kw), tok, norm,
                                 seed=t["seed"], **p_kw))
    eval_prompts = Prompts(LatentDataset(d.get("eval") or d.get("valid") or d["train"], tok, seed=7, **ds_kw),
                           tok, norm, **p_kw).first(E["num_prompts"])
    if decode_override is not None:
        decode = decode_override
    else:
        from models.pocket_tts.codec import Mimi

        decode = Mimi(m["config"]).decode
    scorer = Scorer(policy, reward_cfg, decode)
    anchor_iter, anchor_fn = build_anchor(A, d, t, tok, norm, ds_kw)

    def loss_fn(model, batch):
        loss, mets = grpo_loss(model, batch, rc, kl_beta=g["kl_beta"], pg_norm=g["pg_norm"])
        if "anchor" in batch:
            a_loss = anchor_fn(model, batch["anchor"])
            loss = loss + A["weight"] * a_loss
            mets["anchor"] = a_loss
        return loss, mets

    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    opt, sched = build_optimizer(cfg["optim"], t["max_steps"])
    run_dir = Path(t["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    plain = {k: v for k, v in cfg.items() if k != "data"}
    plain["data"] = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(plain, sort_keys=False))
    step = load_checkpoint(Path(args.resume), policy, None, opt) if args.resume else 0
    log = JsonlLog(run_dir / "train_log.jsonl")

    def evaluate(step_: int) -> dict:
        res = evaluate_prompts(policy, eval_prompts, scorer, rc, seeds=E["seeds"])
        if decode_override is None and eval_prompts:   # one listenable sample per eval
            import soundfile as sf

            mx.random.seed(0)
            p = eval_prompts[0]
            roll = rollout_group(policy, p["tokens"], mx.array(p["voice"]),
                                 RolloutConfig(**{**rc.__dict__, "group_size": 1}), sigma=0.0,
                                 stochastic_eos=False)
            audio = scorer.audios(roll)[0]
            if len(audio):
                (run_dir / "samples").mkdir(exist_ok=True)
                sf.write(str(run_dir / "samples" / f"step{step_:06d}.wav"), audio, Scorer.SAMPLE_RATE)
        mx.random.seed(t["seed"] + step_ + 1)
        np.random.seed(t["seed"] + step_ + 1)
        return {"step": step_, **{f"eval_{k}": round(float(v.mean()), 5) for k, v in res.items()}}

    first_eval = last_eval = evaluate(step)
    log(first_eval, prefix="[eval] ")
    skipped = built = 0
    while step < t["max_steps"]:
        t0 = time.perf_counter()
        # Phase A: every group of this step under the same (current) policy -> on-policy.
        groups, stats = [], {}
        while len(groups) < g["prompts_per_step"]:
            p = next(train_prompts)
            voice = mx.array(p["voice"])
            roll = rollout_group(policy, p["tokens"], voice, rc)
            scored, _ = scorer(p, roll, rc.max_frames)
            reward = np.asarray(scored["reward"], dtype=np.float32)
            for k, v in numeric_info(scored["info"]).items():
                stats.setdefault(k, []).append(v)
            stats.setdefault("no_eos", []).append(float(roll["hit_cap"].mean()))
            stats.setdefault("len_ratio", []).append(float(roll["lengths"].mean() / max(p["real_frames"], 1)))
            if g["skip_zero_variance"] and float(reward.std()) < g["zero_var_eps"]:
                skipped += 1      # all rollouts equal -> zero advantage, no signal
                continue
            built += 1
            adv = group_advantages(reward, rc.group_size, eps=reward_cfg.eps, adv_norm=reward_cfg.adv_norm)
            batch = make_group_batch(policy, ref, roll, p["tokens"], voice if voice.shape[0] else None, adv, rc)
            if anchor_iter is not None:
                batch["anchor"] = next(anchor_iter)
            groups.append(batch)
        t_roll = time.perf_counter() - t0

        # Phase B: one update over all groups.
        grads_sum, mets_sum = None, {}
        for batch in groups:
            (loss, mets), grads = loss_and_grad(policy, batch)
            grads_sum = grads if grads_sum is None else tree_map(mx.add, grads_sum, grads)
            mx.eval(grads_sum, loss, mets)
            mets_sum.setdefault("loss", []).append(loss.item())
            for k, v in mets.items():
                mets_sum.setdefault(k, []).append(v.item())
        grads, gnorm = optim.clip_grad_norm(tree_map(lambda x: x / len(groups), grads_sum),
                                            cfg["optim"]["max_norm"])
        opt.update(policy, grads)
        mx.eval(policy.parameters(), opt.state)
        step += 1

        if step % t["log_freq"] == 0 or step == 1:
            entry = {"step": step, "loss": round(float(np.mean(mets_sum.pop("loss"))), 6),
                     "grad_norm": round(gnorm.item(), 4), "lr": float(sched(mx.array(step)))}
            entry.update({k: round(float(np.mean(v)), 6) for k, v in mets_sum.items()})
            entry.update({f"{k}_mean": round(float(np.mean(v)), 5) for k, v in stats.items()})
            entry["skip_ratio"] = round(skipped / max(skipped + built, 1), 3)
            entry["sec_per_step"] = round(time.perf_counter() - t0, 2)
            entry["rollout_sec"] = round(t_roll, 2)
            log(entry)
        if step % E["every"] == 0 or step == t["max_steps"]:
            last_eval = evaluate(step)
            log(last_eval, prefix="[eval] ")
        if step % t["ckpt_freq"] == 0 or step == t["max_steps"]:
            ckpt = save_checkpoint(run_dir, step, policy, policy, raw, None, opt, weights, t["keep_last_n"])
            print(f"  [ckpt] {ckpt}", flush=True)

    if args.smoke_test:
        r0, r1 = first_eval.get("eval_r_smoke_target"), last_eval.get("eval_r_smoke_target")
        ok = r0 is not None and r1 is not None and r1 > r0 + 0.05
        print(f"[smoke] eval reward {r0:.4f} -> {r1:.4f}: {'OK' if ok else 'FAILED'}")
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
