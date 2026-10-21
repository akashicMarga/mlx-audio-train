"""
GRPO for Pocket TTS (continuous latents) in MLX.

Pocket TTS's sampler head is a deterministic one-step map from noise to a latent, so
there is no token log-probability to put in a policy gradient. Following Flow-GRPO,
the rollout adds a small Gaussian step around the head's output, which gives each
frame a tractable likelihood:

    eps ~ N(0, temp)         (the usual head noise, recorded)
    mu  = head(z_t, eps)     (z_t = backbone over prefix + previous frames)
    x_t = mu + sigma * xi,   log pi(x_t) = -||x_t - mu||^2 / (2 sigma^2)   (+ const)
    stop_t ~ Bernoulli(sigmoid((eos_logit_t - eos_threshold) / eos_temp))  (optional)

At inference sigma = 0 and EOS is the usual threshold, so the model is unchanged.

Per prompt, G rollouts are decoded, scored with the shared reward stack in
train/grpo/rewards.py (Whisper CER/WER, DNSMOS, UTMOS, speaker similarity, length
guards) and turned into group-relative advantages. The loss, on-policy (one
optimizer step per batch of rollouts, as in the Qwen3-TTS GRPO):

    L = pg + kl_beta * KL(pi || pi_ref) [+ anchor_weight * anchor loss on real data]
    pg = -A * (sum of the rollout's frame log-probs), reduced per token or per sequence
    KL closed form: ||mu - mu_ref||^2 / (2 sigma^2)  (+ Bernoulli KL of the stop decisions)

pi_ref is a frozen snapshot of the starting model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.pocket_tts.model import FlowLM, KVCache, sample_head


@dataclass
class RolloutConfig:
    group_size: int = 4
    temp: float = 0.3           # head noise temperature (the released models use 0.3)
    sigma: float = 0.1          # exploration std around the head output (normalized latent units)
    max_frames: int = 250       # 20 s; rollouts that never stop get the length penalty
    min_frames: int = 6         # no stop decisions before this (upstream ignores EOS early)
    eos_threshold: float = -1.0
    eos_temp: float = 1.0       # temperature of the stochastic stop decision
    train_eos: bool = True      # stochastic stop + its log-prob in the policy (fixes no-EOS runs)


def _prefix(model: FlowLM, tokens: np.ndarray, voice: mx.array | None, null: bool = False) -> mx.array:
    parts = []
    if "bos_before_voice" in model:
        parts.append(model.bos_before_voice[0])
    if not null:
        if voice is not None and "speaker_proj_weight" in model and voice.shape[0]:
            parts.append(voice @ model.speaker_proj_weight.T)
        parts.append(model.conditioner(mx.array(tokens, dtype=mx.int32)))
    if not parts:
        return mx.zeros((0, model.args.d_model))
    return mx.concatenate(parts, axis=0)


def rollout_group(model: FlowLM, tokens: np.ndarray, voice: mx.array | None, rc: RolloutConfig,
                  *, sigma: float | None = None, stochastic_eos: bool | None = None) -> dict:
    """G rollouts of one prompt, recording every draw. All arrays in normalized latent space.

    Returns x [G, T, C] (padded), eps [G, T, C], mu [G, T, C], lengths [G] (frames kept),
    hit_cap [G] bool. A row that stops at step t keeps frames 0..t-1 (the stop frame is
    not emitted, as in pocket-tts)."""
    a = model.args
    G = rc.group_size
    sigma = rc.sigma if sigma is None else sigma
    stochastic_eos = rc.train_eos if stochastic_eos is None else stochastic_eos
    p = _prefix(model, tokens, voice)
    prefix = mx.broadcast_to(p[None], (G,) + p.shape)
    cache = [KVCache() for _ in model.transformer.layers]
    x_lat = mx.broadcast_to(model.bos_emb[None, None, :], (G, 1, a.ldim))
    xs, epss, mus = [], [], []
    lengths = np.full(G, rc.max_frames, dtype=np.int64)
    alive = np.ones(G, dtype=bool)
    for t in range(rc.max_frames):
        inp = model.input_linear(x_lat)
        if t == 0:
            inp = mx.concatenate([prefix.astype(inp.dtype), inp], axis=1)
        z = model.backbone(inp, cache)[:, -1].astype(mx.float32)
        eps = (rc.temp ** 0.5) * mx.random.normal((G, a.ldim))
        mu = sample_head(a.flow_type, model.flow_net, z, eps, 1)
        x = mu + sigma * mx.random.normal((G, a.ldim)) if sigma > 0 else mu
        if t >= rc.min_frames:
            logit = np.array(model.out_eos(z)[:, 0])
            if stochastic_eos:
                p = 0.5 * (1 + np.tanh(0.5 * (logit - rc.eos_threshold) / rc.eos_temp))  # stable sigmoid
                stop = np.random.random(G) < p
            else:
                stop = logit > rc.eos_threshold
            newly = stop & alive
            lengths[newly] = t
            alive &= ~stop
        mx.eval(x, eps, mu)
        xs.append(x); epss.append(eps); mus.append(mu)
        if not alive.any():
            break
        x_lat = x[:, None, :]
    T = len(xs)
    lengths = np.minimum(lengths, T)
    return {"x": mx.stack(xs, axis=1), "eps": mx.stack(epss, axis=1), "mu": mx.stack(mus, axis=1),
            "lengths": lengths, "hit_cap": lengths >= rc.max_frames}


def _teacher_forced(model: FlowLM, batch: dict):
    """z, mu (head on the recorded noise) and stop logits for the recorded frames."""
    G = batch["x"].shape[0]
    ones = np.ones(G, dtype=bool)
    z = model.backbone_z(batch["x_in"], batch["text_tokens"], batch["voice"], batch["voice_lens"],
                         ones, ones).astype(mx.float32)
    mu = sample_head(model.args.flow_type, model.flow_net, z, batch["eps_in"], 1)
    eos = model.out_eos(z)[..., 0].astype(mx.float32)
    return mu, eos


def make_group_batch(model: FlowLM, ref: FlowLM, roll: dict, tokens: np.ndarray,
                     voice: mx.array | None, advantages: np.ndarray, rc: RolloutConfig) -> dict:
    """Phase-B inputs for one group, including the frozen reference's mu / stop logits."""
    G, T, C = roll["x"].shape
    lengths = roll["lengths"]
    # One extra (padding) frame so z exists at the stop step t = length.
    pad = mx.zeros((G, 1, C))
    x_in = mx.concatenate([roll["x"], pad], axis=1)
    eps_in = mx.concatenate([roll["eps"], pad], axis=1)
    frame_mask = np.arange(T + 1)[None, :] < lengths[:, None]                     # emitted frames
    decision = (np.arange(T + 1)[None, :] >= rc.min_frames) & (np.arange(T + 1)[None, :] <= lengths[:, None])
    decision &= np.arange(T + 1)[None, :] < rc.max_frames
    stop_label = np.arange(T + 1)[None, :] == lengths[:, None]                    # the stop step
    if not rc.train_eos:
        decision[:] = False
    Tv = 0 if voice is None else voice.shape[0]
    batch = {
        "x": x_in, "x_in": x_in, "eps_in": eps_in,
        "text_tokens": [tokens] * G,
        "voice": mx.broadcast_to(voice[None], (G,) + voice.shape) if Tv else None,
        "voice_lens": np.full(G, Tv),
        "frame_mask": mx.array(frame_mask), "decision": mx.array(decision),
        "stop_label": mx.array(stop_label), "advantages": mx.array(advantages.astype(np.float32)),
        "mu_roll": mx.concatenate([roll["mu"], pad], axis=1),
    }
    mu_ref, eos_ref = _teacher_forced(ref, batch)
    batch["mu_ref"], batch["eos_ref"] = mx.stop_gradient(mu_ref), mx.stop_gradient(eos_ref)
    mx.eval(batch["mu_ref"], batch["eos_ref"])
    return batch


def _reduce(values: mx.array, mask: mx.array, pg_norm: str) -> mx.array:
    """token: sum / #tokens (length-weighted); sequence: mean of per-rollout means (Dr. GRPO)."""
    if pg_norm == "sequence":
        per = (values * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
        return per.mean()
    return (values * mask).sum() / mx.maximum(mask.sum(), 1.0)


def grpo_loss(model: FlowLM, batch: dict, rc: RolloutConfig, *, kl_beta: float = 0.02,
              pg_norm: str = "token") -> tuple[mx.array, dict]:
    mu, eos = _teacher_forced(model, batch)
    fm = batch["frame_mask"].astype(mx.float32)
    dm = batch["decision"].astype(mx.float32)
    s2 = 2 * rc.sigma ** 2
    gauss_lp = -mx.sum((batch["x"] - mu) ** 2, axis=-1) / s2                        # [G, T+1]
    l = (eos - rc.eos_threshold) / rc.eos_temp
    l_ref = (batch["eos_ref"] - rc.eos_threshold) / rc.eos_temp
    eos_lp = mx.where(batch["stop_label"], -nn.softplus(-l), -nn.softplus(l))        # log sigmoid(+-l)
    logp = gauss_lp * fm + eos_lp * dm
    mask = mx.maximum(fm, dm)
    adv = batch["advantages"][:, None]
    pg = _reduce(-adv * logp, mask, pg_norm)

    kl_gauss = mx.sum((mu - batch["mu_ref"]) ** 2, axis=-1) / s2
    p, p_ref = mx.sigmoid(l), mx.sigmoid(l_ref)
    eps = 1e-6
    kl_eos = p * (mx.log(p + eps) - mx.log(p_ref + eps)) + (1 - p) * (
        mx.log(1 - p + eps) - mx.log(1 - p_ref + eps))
    kl = _reduce(kl_gauss * fm + kl_eos * dm, mask, pg_norm)
    loss = pg + kl_beta * kl
    parity = mx.max(mx.abs(mx.stop_gradient(mu) - batch["mu_roll"]) * fm[..., None])
    return loss, {"pg": pg, "kl": kl, "parity": parity}


def snapshot(model: FlowLM) -> FlowLM:
    """Frozen copy of the current policy (the KL reference)."""
    from mlx.utils import tree_flatten

    ref = FlowLM(model.args)
    if "speaker_proj_weight" not in model:
        ref.pop("speaker_proj_weight")
    ref.load_weights([(k, mx.array(v)) for k, v in tree_flatten(model.parameters())], strict=True)
    ref.freeze()
    return ref


# ── Scoring / evaluation shared by the trainer and the held-out comparison ────

UNSCORABLE = set("0123456789@/_#%$&+=*<>\\|~^")


class Prompts:
    """(text, tokens, voice latents) prompts from a latent manifest. Skips text an ASR
    reward cannot score (digits / symbols, as in Kyutai's REFL) and over-long targets."""

    def __init__(self, ds, tokenize, norm=None, *, skip_unscorable: bool = True,
                 max_prompt_sec: float = 15.0, seed: int = 0):
        self.ds, self.tokenize, self.norm = ds, tokenize, norm
        self.skip_unscorable, self.max_prompt_sec = skip_unscorable, max_prompt_sec
        self.rng = np.random.default_rng(seed)

    def get(self, i: int) -> dict | None:
        target, text, voice = self.ds.sample_text(i)
        # Checked on the RAW text: once "25" is spelled out, Whisper may still write "25"
        # and the CER would punish correct speech.
        if self.skip_unscorable and any(c in UNSCORABLE for c in text):
            return None
        text = self.norm(text) if self.norm else text
        if len(target) / 12.5 > self.max_prompt_sec or not text.strip():
            return None
        return {"text": text, "tokens": np.array(self.tokenize(text), dtype=np.int32),
                "voice": voice, "real_frames": len(target)}

    def first(self, n: int) -> list[dict]:
        out = []
        for i in range(len(self.ds)):
            p = self.get(i)
            if p is not None:
                out.append(p)
            if len(out) == n:
                break
        return out

    def __iter__(self):
        while True:
            for i in self.rng.permutation(len(self.ds)):
                p = self.get(int(i))
                if p is not None:
                    yield p


class Scorer:
    """Decode rollouts (official Mimi, or a test stub) and score them with the shared
    reward stack of train/grpo/rewards.py."""

    SAMPLE_RATE = 24000

    def __init__(self, model: FlowLM, reward_cfg, decode):
        self.reward_cfg, self.decode = reward_cfg, decode
        self.std, self.mean = np.array(model.emb_std), np.array(model.emb_mean)
        self.need_ref_audio = reward_cfg.weight("wavlm_speaker_similarity") != 0

    def audios(self, roll: dict) -> list[np.ndarray]:
        x = np.array(roll["x"])
        return [self.decode(x[i, : int(n)] * self.std + self.mean) if n > 0 else np.zeros(0, np.float32)
                for i, n in enumerate(roll["lengths"])]

    def __call__(self, prompt: dict, roll: dict, max_frames: int):
        from train.grpo.rewards import combine_rewards

        audios = self.audios(roll)
        ref_audios = None
        if self.need_ref_audio:
            if "voice_wav" not in prompt:
                prompt["voice_wav"] = self.decode(prompt["voice"]) if len(prompt["voice"]) else None
            ref_audios = [prompt["voice_wav"]] * len(audios)
        scored = combine_rewards([mx.array(a) for a in audios], [prompt["text"]] * len(audios),
                                 roll["lengths"].tolist(), max_frames, self.reward_cfg,
                                 sample_rate=self.SAMPLE_RATE, ref_audios=ref_audios)
        return scored, audios


def numeric_info(info: dict) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in info.items()
            if k != "hyp" and isinstance(v, list) and v and isinstance(v[0], (int, float))}


def evaluate_prompts(model: FlowLM, prompts: list[dict], scorer: Scorer, rc: RolloutConfig,
                     seeds=(0,)) -> dict[str, np.ndarray]:
    """Inference settings (sigma 0, threshold EOS), one rollout per prompt per seed.
    Returns metric -> [len(seeds) * len(prompts)] values (paired across models)."""
    eval_rc = RolloutConfig(**{**rc.__dict__, "group_size": 1})
    out: dict[str, list] = {}
    for seed in seeds:
        for i, p in enumerate(prompts):
            mx.random.seed(seed * 100003 + i)
            np.random.seed(seed * 100003 + i)
            voice = mx.array(p["voice"])
            roll = rollout_group(model, p["tokens"], voice, eval_rc, sigma=0.0, stochastic_eos=False)
            scored, _ = scorer(p, roll, rc.max_frames)
            for k, v in numeric_info(scored["info"]).items():
                out.setdefault(k, []).append(v)
            out.setdefault("no_eos", []).append(float(roll["hit_cap"].mean()))
            out.setdefault("sec", []).append(float(roll["lengths"].mean() / 12.5))
    return {k: np.array(v) for k, v in out.items()}
