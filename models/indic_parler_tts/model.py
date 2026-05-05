"""
Full IndicParlerTTS model: weight loading + inference.

Generation flow:
  description_ids  → T5Encoder          → enc_hidden [B, T_desc, 1024]
  prompt_ids       → embed_prompts      → prompt_emb [B, T_prompt, 1024]
  decoder:
    prompt_emb (prefill) → KV-cached layers → audio token generation (delay pattern)
  codes [9, T]     → DACDecoder.decode → audio [T*512]

Delay pattern (9 codebooks):
  At generation step t, codebook k feeds token max(0, t-k) back as input;
  codebook k uses BOS for the first k steps.
  After num_codebooks extra steps, tokens[k, k:k+T] are the audio tokens.
"""

import numpy as np
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mxu

from .config import IndicParlerTTSConfig
from .t5_encoder import T5Encoder
from .decoder import ParlerDecoder
from .dac import DACDecoder


# ── weight key remapping ──────────────────────────────────────────────────────

def _remap_weights(flat: dict) -> dict:
    """
    Convert HuggingFace weight keys → our MLX module attribute paths.
    Also transposes Conv weights and reshapes DAC snake alphas.
    """
    out = {}

    def add(mlx_key: str, val: np.ndarray):
        out[mlx_key] = mx.array(val)

    for hf_key, val in flat.items():

        # ── T5 encoder ───────────────────────────────────────────────────────
        if hf_key == "text_encoder.shared.weight":
            add("text_encoder.shared.weight", val)
            continue

        if hf_key == "text_encoder.encoder.final_layer_norm.weight":
            add("text_encoder.final_ln.weight", val)
            continue

        if hf_key.startswith("text_encoder.encoder.block."):
            rest = hf_key[len("text_encoder.encoder.block."):]
            idx, rest = rest.split(".", 1)

            if rest.startswith("layer.0.SelfAttention."):
                sub = rest[len("layer.0.SelfAttention."):]
                attn_names = {"q": "q", "k": "k", "v": "v", "o": "o"}
                for pt, mx_name in attn_names.items():
                    if sub == f"{pt}.weight":
                        add(f"text_encoder.blocks.{idx}.attn.{mx_name}.weight", val)
                        break
                else:
                    if sub == "relative_attention_bias.weight":
                        add(f"text_encoder.blocks.{idx}.attn.relative_attention_bias.weight", val)
                continue

            if rest == "layer.0.layer_norm.weight":
                add(f"text_encoder.blocks.{idx}.attn_ln.weight", val)
                continue

            if rest.startswith("layer.1.DenseReluDense."):
                sub = rest[len("layer.1.DenseReluDense."):]
                name_map = {"wi_0.weight": "wi_0.weight", "wi_1.weight": "wi_1.weight", "wo.weight": "wo.weight"}
                if sub in name_map:
                    add(f"text_encoder.blocks.{idx}.ffn.{name_map[sub]}", val)
                continue

            if rest == "layer.1.layer_norm.weight":
                add(f"text_encoder.blocks.{idx}.ffn_ln.weight", val)
                continue

        # ── embed_prompts ─────────────────────────────────────────────────────
        if hf_key == "embed_prompts.weight":
            add("embed_prompts.weight", val)
            continue

        # ── Parler decoder model ──────────────────────────────────────────────
        if hf_key.startswith("decoder.model.decoder."):
            rest = hf_key[len("decoder.model.decoder."):]

            if rest.startswith("embed_tokens."):
                _, idx, param = rest.split(".", 2)
                add(f"decoder.embed_tokens.{idx}.{param}", val)
                continue

            if rest == "embed_positions.weights":
                add("decoder.embed_positions.weight", val)
                continue

            if rest in ("layer_norm.weight", "layer_norm.bias"):
                param = rest.split(".")[1]
                add(f"decoder.final_ln.{param}", val)
                continue

            if rest.startswith("layers."):
                rest2 = rest[len("layers."):]
                idx, rest2 = rest2.split(".", 1)

                proj_map = {
                    "self_attn.q_proj.weight": "self_attn.q.weight",
                    "self_attn.k_proj.weight": "self_attn.k.weight",
                    "self_attn.v_proj.weight": "self_attn.v.weight",
                    "self_attn.out_proj.weight": "self_attn.out.weight",
                    "encoder_attn.q_proj.weight": "cross_attn.q.weight",
                    "encoder_attn.k_proj.weight": "cross_attn.k.weight",
                    "encoder_attn.v_proj.weight": "cross_attn.v.weight",
                    "encoder_attn.out_proj.weight": "cross_attn.out.weight",
                    "self_attn_layer_norm.weight": "self_attn_ln.weight",
                    "self_attn_layer_norm.bias": "self_attn_ln.bias",
                    "encoder_attn_layer_norm.weight": "cross_attn_ln.weight",
                    "encoder_attn_layer_norm.bias": "cross_attn_ln.bias",
                    "fc1.weight": "fc1.weight",
                    "fc2.weight": "fc2.weight",
                    "final_layer_norm.weight": "ffn_ln.weight",
                    "final_layer_norm.bias": "ffn_ln.bias",
                }
                if rest2 in proj_map:
                    add(f"decoder.layers.{idx}.{proj_map[rest2]}", val)
                continue

        if hf_key == "decoder.lm_heads.weight":
            add("decoder.lm_heads.weight", val)
            continue

        # ── DAC quantizer ─────────────────────────────────────────────────────
        if hf_key.startswith("audio_encoder.quantizer.quantizers."):
            rest = hf_key[len("audio_encoder.quantizer.quantizers."):]
            idx, rest = rest.split(".", 1)
            k = int(idx)

            if rest == "codebook.weight":
                add(f"dac.quantizer.codebooks.{k}", val)
            elif rest == "out_proj.weight":
                # PyTorch Conv1d (1, out, in): squeeze kernel dim → linear (out, in)
                w = val[:, :, 0]               # (hidden_size=1024, codebook_dim=8)
                add(f"dac.quantizer.out_proj_w.{k}", w)
            elif rest == "out_proj.bias":
                add(f"dac.quantizer.out_proj_b.{k}", val)
            # in_proj not needed for inference (decoding only)
            continue

        # ── DAC decoder ───────────────────────────────────────────────────────
        if hf_key.startswith("audio_encoder.decoder."):
            rest = hf_key[len("audio_encoder.decoder."):]

            if rest == "conv1.weight":
                add("dac.conv1.weight", val.transpose(0, 2, 1))  # (out,k,in)
                continue
            if rest == "conv1.bias":
                add("dac.conv1.bias", val)
                continue
            if rest == "conv2.weight":
                add("dac.conv2.weight", val.transpose(0, 2, 1))
                continue
            if rest == "conv2.bias":
                add("dac.conv2.bias", val)
                continue
            if rest == "snake1.alpha":
                # alpha after all blocks: PyTorch (1, C, 1) → MLX (1, 1, C)
                add("dac.end_snake", val.transpose(0, 2, 1))
                continue

            if rest.startswith("block."):
                rest2 = rest[len("block."):]
                bidx, rest2 = rest2.split(".", 1)

                if rest2.startswith("snake1.alpha"):
                    add(f"dac.blocks.{bidx}.snake", val.transpose(0, 2, 1))
                    continue
                if rest2.startswith("conv_t1.weight"):
                    # ConvTranspose1d PyTorch (in, out, k) → MLX (out, k, in)
                    add(f"dac.blocks.{bidx}.upsample.weight", val.transpose(1, 2, 0))
                    continue
                if rest2.startswith("conv_t1.bias"):
                    add(f"dac.blocks.{bidx}.upsample.bias", val)
                    continue

                # res_unit{1,2,3} → 0-indexed
                for ru in range(1, 4):
                    prefix = f"res_unit{ru}."
                    if rest2.startswith(prefix):
                        sub = rest2[len(prefix):]
                        ri = ru - 1
                        if sub == "snake1.alpha":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.alpha1",
                                val.transpose(0, 2, 1))
                        elif sub == "snake2.alpha":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.alpha2",
                                val.transpose(0, 2, 1))
                        elif sub == "conv1.weight":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.conv1.weight",
                                val.transpose(0, 2, 1))
                        elif sub == "conv1.bias":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.conv1.bias", val)
                        elif sub == "conv2.weight":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.conv2.weight",
                                val.transpose(0, 2, 1))
                        elif sub == "conv2.bias":
                            add(f"dac.blocks.{bidx}.res_units.{ri}.conv2.bias", val)
                        break
                continue

        # skip encoder weights (not needed for inference)


    return out


def _model_weight_keys(model: nn.Module) -> set[str]:
    """Return model parameter keys in MLX load_weights format."""
    return {key for key, _ in mxu.tree_flatten(model.parameters())}


def _ignored_missing_keys(keys: Iterable[str]) -> set[str]:
    """Weights intentionally not loaded for the inference-only path."""
    ignored = set()
    for key in keys:
        if key.startswith("dac.quantizer.in_proj_"):
            ignored.add(key)
    return ignored


def _audit_remapped_weights(model: nn.Module, weights: dict) -> tuple[list[str], list[str]]:
    expected = _model_weight_keys(model)
    supplied = set(weights)
    missing = sorted(expected - supplied - _ignored_missing_keys(expected))
    unexpected = sorted(supplied - expected)
    return missing, unexpected


# ── Main model ────────────────────────────────────────────────────────────────

class IndicParlerTTS(nn.Module):
    def __init__(self, cfg: IndicParlerTTSConfig = None):
        super().__init__()
        if cfg is None:
            cfg = IndicParlerTTSConfig()
        self.cfg = cfg
        self.text_encoder = T5Encoder(cfg.t5)
        self.embed_prompts = nn.Embedding(cfg.decoder.prompt_vocab_size, cfg.t5.d_model)
        self.decoder = ParlerDecoder(cfg.decoder)
        self.dac = DACDecoder(cfg.dac)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "ai4bharat/indic-parler-tts",
        *,
        strict: bool = True,
        audit: bool = True,
    ) -> "IndicParlerTTS":
        from huggingface_hub import hf_hub_download
        import safetensors

        path = hf_hub_download(repo_id, "model.safetensors")
        st = safetensors.safe_open(path, framework="numpy", device="cpu")
        flat = {k: st.get_tensor(k) for k in st.keys()}

        cfg = IndicParlerTTSConfig(repo_id=repo_id)
        model = cls(cfg)

        weights = _remap_weights(flat)
        if audit:
            missing, unexpected = _audit_remapped_weights(model, weights)
            if missing or unexpected:
                message = (
                    "[indic-parler-tts] HF->MLX weight remap audit failed:\n"
                    f"  missing expected keys: {len(missing)}\n"
                    f"  unexpected remapped keys: {len(unexpected)}"
                )
                if missing:
                    message += "\n  first missing: " + ", ".join(missing[:10])
                if unexpected:
                    message += "\n  first unexpected: " + ", ".join(unexpected[:10])
                if strict:
                    raise ValueError(message)
                print(message)

        # strict=False because DAC in_proj weights exist upstream but are not used
        # for decode-only inference.
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        return model

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        description_ids: mx.array,
        prompt_ids: mx.array,
        max_audio_length_s: float = 10.0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        description_ids : [1, T_desc]  — T5-tokenized style description
        prompt_ids       : [1, T_prompt] — custom-tokenized text to speak
        Returns          : float32 numpy array of audio samples at 44100 Hz
        """
        cfg = self.cfg.decoder
        rng = np.random.default_rng(seed)
        num_cb = cfg.num_codebooks
        bos = cfg.bos_token_id
        eos = cfg.eos_token_id
        # target_T = max audio frames; max_steps adds (num_cb-1) delay drain steps
        target_T = int(max_audio_length_s * 44100 / 512)
        max_steps = target_T + num_cb - 1

        # 1. Encode style description
        enc_hidden = self.text_encoder(description_ids)  # [1, T_desc, 1024]

        # 2. First decoder pass: prompt text embeddings are prepended to the
        # initial all-BOS audio step. This mirrors Parler's prompt_cross_attention
        # false path; the prompt is decoder context, not a separate cached decode.
        T_prompt = prompt_ids.shape[1]
        prompt_emb = self.embed_prompts(prompt_ids)  # [1, T_prompt, 1024]

        # KV caches: list of lists [self_k, self_v] and [cross_k, cross_v]
        self_caches = [[] for _ in range(cfg.num_layers)]
        cross_caches = [[] for _ in range(cfg.num_layers)]

        bos_tokens = mx.full((1, num_cb), bos, dtype=mx.int32)
        bos_audio_emb = self.decoder.embed_audio_no_pos(bos_tokens)
        first_emb = mx.concatenate([prompt_emb, bos_audio_emb], axis=1)
        first_emb = first_emb + self.decoder.decoder_position(0, first_emb.shape[1])

        hidden = self.decoder.forward_layers(
            first_emb, enc_hidden,
            mask=_causal_mask(first_emb.shape[1], dtype=first_emb.dtype),
            self_caches=self_caches,
            cross_caches=cross_caches,
        )
        logits = self.decoder.logits(hidden[:, -1:, :])
        mx.eval(logits)
        logits_np = np.array(logits[0, 0])

        generated = [[bos] * (max_steps + 2) for _ in range(num_cb)]
        for k in range(num_cb):
            cb_temp = temperature * max(0.25, 1.0 - k * 0.08)
            token = _sample_token(logits_np[k], cb_temp, top_k, rng, top_p)
            generated[k][1] = token

        # 3. Autoregressive audio generation with delay pattern
        #
        # Input: at step t, CB k uses its own prediction from step t-1 (BOS for first k steps).
        # Output assembly: frame p = CB k token from step p+k.
        finished = (generated[0][1] == eos)
        last_step = 0

        for step in range(1, max_steps):
            last_step = step
            offset = T_prompt + step

            # Each codebook uses its own prediction from the immediately previous step.
            # The delay lives only in the output assembly (frame p = CB k at step p+k),
            # not in the input feedback — the first k steps use BOS instead.
            tokens_this_step = mx.array(
                [[generated[k][step] if step > k else bos for k in range(num_cb)]],
                dtype=mx.int32,
            )  # [1, num_codebooks]
            audio_emb = self.decoder.embed_audio(tokens_this_step, offset=offset)

            hidden = self.decoder.forward_layers(
                audio_emb, enc_hidden,
                mask=None,
                self_caches=self_caches,
                cross_caches=cross_caches,
            )
            logits = self.decoder.logits(hidden)  # [1, 1, 9, 1088]
            mx.eval(logits)
            logits_np = np.array(logits[0, 0])    # [9, 1088]

            for k in range(num_cb):
                cb_temp = temperature * max(0.25, 1.0 - k * 0.08)
                token = _sample_token(logits_np[k], cb_temp, top_k, rng, top_p)
                generated[k][step + 1] = token

            # CB0 leads the staggered EOS pattern; stop as soon as it generates EOS.
            # CB1..8 will also EOS in subsequent steps but we don't wait — the frame
            # extraction filter drops any frame containing a special token.
            if generated[0][step + 1] == eos:
                finished = True
                break

        # 4. Extract audio tokens using the same invariant as Parler's delay mask:
        # a decoded frame is valid only if every codebook contributes a real DAC
        # token. Special tokens are >= codebook_size and must be dropped, not
        # clamped to zero, otherwise DAC decodes a long tonal tail.
        max_frames = (last_step - num_cb + 2) if finished else target_T
        max_frames = max(max_frames, 0)
        frames: list[list[int]] = []
        for p in range(max_frames):
            frame = [generated[k][k + p + 1] for k in range(num_cb)]
            if all(0 <= token < self.cfg.dac.codebook_size for token in frame):
                frames.append(frame)

        if not frames:
            return np.zeros(0, dtype=np.float32)

        codes = np.array(frames, dtype=np.int32).T
        codes_mx = mx.array(codes)

        # 5. DAC decode
        audio = self.dac.decode(codes_mx)
        mx.eval(audio)
        return np.array(audio, dtype=np.float32)


def _causal_mask(T: int, dtype=mx.float32) -> mx.array:
    mask = mx.triu(mx.full((T, T), -1e9, dtype=dtype), k=1)
    return mask[None, None, :, :]  # [1, 1, T, T]


def _sample_token(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
    top_p: float = 1.0,
) -> int:
    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -1e9, logits)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumprobs = np.cumsum(probs[sorted_idx])
        cutoff = sorted_idx[np.searchsorted(cumprobs, top_p)]
        probs[probs < probs[cutoff]] = 0.0
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
