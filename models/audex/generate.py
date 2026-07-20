"""
Top-level Audex inference API (MLX): text generation and text-to-speech.

    from models.audex import load_model, text_generate, tts_generate
    import soundfile as sf

    m = load_model("/path/to/audex_mlx")
    print(text_generate(m, "Explain RoPE in one sentence."))
    wav = tts_generate(m, "The weather is so good this morning.")
    sf.write("out.wav", wav, 16000)

`load_model` expects a converted MLX checkpoint directory (see convert.py)
containing: lm.safetensors, speech_decoder.safetensors, lm_config.json,
speech_decoder_config.json, and the HF tokenizer files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .lm import LMConfig, NemotronDenseForCausalLM
from .speech_decoder import SpeechDecoderConfig, AudexSpeechDecoder

SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools."
)


@dataclass
class AudexModel:
    lm: NemotronDenseForCausalLM
    decoder: AudexSpeechDecoder
    tokenizer: object
    speech_codec: dict          # token_id -> codec_id
    markers: dict               # name -> token_id
    eos_ids: list


def load_model(model_dir: str) -> AudexModel:
    from transformers import AutoTokenizer, logging as hf_logging

    model_dir = Path(model_dir)
    lm_cfg = LMConfig(**json.loads((model_dir / "lm_config.json").read_text()))
    dec_cfg = SpeechDecoderConfig(**json.loads((model_dir / "speech_decoder_config.json").read_text()))

    lm = NemotronDenseForCausalLM(lm_cfg)
    lm.load_weights(str(model_dir / "lm.safetensors"))
    lm.eval()

    decoder = AudexSpeechDecoder(dec_cfg)
    decoder.load_weights(str(model_dir / "speech_decoder.safetensors"))
    decoder.eval()

    mx.eval(lm.parameters(), decoder.parameters())

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    speech_codec, markers = _build_codec_maps(tokenizer)
    eos_ids = _resolve_eos(tokenizer)
    print(f"[audex] Ready. {len(speech_codec)} speechcodec tokens, markers={markers}, eos={eos_ids}")
    return AudexModel(lm, decoder, tokenizer, speech_codec, markers, eos_ids)


def _build_codec_maps(tokenizer):
    speech_codec, markers = {}, {}
    marker_names = {"speechgen_start", "speechgen_end", "audiogen_start", "audiogen_end"}
    for tok_str, tok_id in tokenizer.get_vocab().items():
        m = re.match(r"<speechcodec_(\d+)>", tok_str)
        if m:
            speech_codec[tok_id] = int(m.group(1))
        elif tok_str.strip("<>") in marker_names:
            markers[tok_str.strip("<>")] = tok_id
    return speech_codec, markers


def _resolve_eos(tokenizer):
    ids = set()
    for t in ("<|im_end|>", "</s>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            ids.add(i)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return list(ids)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def _sample(logits: mx.array, temperature: float, top_k: int, top_p: float) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    logits = logits * (1.0 / temperature)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        idx = mx.argpartition(-logits, k - 1)[:k]
        sub = logits[idx]
        probs = mx.softmax(sub, axis=-1)
        choice = mx.random.categorical(mx.log(probs)).item()
        return int(idx[choice].item())
    if top_p and 0.0 < top_p < 1.0:
        order = mx.argsort(-logits)
        sorted_logits = logits[order]
        probs = mx.softmax(sorted_logits, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        mask = cum - probs > top_p          # keep tokens until cumulative>top_p
        sorted_logits = mx.where(mask, mx.full(sorted_logits.shape, -1e9), sorted_logits)
        choice = mx.random.categorical(sorted_logits).item()
        return int(order[choice].item())
    return int(mx.random.categorical(logits).item())


def _generate_ids(model: AudexModel, prompt_ids, max_new_tokens, temperature, top_k, top_p,
                  stop_ids, cfg_scale=1.0, uncond_ids=None, marker_only=None):
    """Autoregressive loop. Returns list of generated token ids (excluding prompt).

    If cfg_scale>1 and uncond_ids given, runs a parallel unconditional stream and
    blends logits: uncond + scale*(cond-uncond).
    """
    lm = model.lm
    cache = lm.make_cache()
    x = mx.array(prompt_ids, dtype=mx.int32)[None]
    logits = lm(x, cache=cache)[:, -1, :]

    use_cfg = cfg_scale and cfg_scale > 1.0 and uncond_ids is not None
    if use_cfg:
        ucache = lm.make_cache()
        ux = mx.array(uncond_ids, dtype=mx.int32)[None]
        ulogits = lm(ux, cache=ucache)[:, -1, :]
        logits = ulogits + cfg_scale * (logits - ulogits)

    out = []
    stop_set = set(stop_ids)
    for _ in range(max_new_tokens):
        tok = _sample(logits[0], temperature, top_k, top_p)
        if tok in stop_set:
            break
        out.append(tok)
        nxt = mx.array([[tok]], dtype=mx.int32)
        logits = lm(nxt, cache=cache)[:, -1, :]
        if use_cfg:
            ulogits = lm(nxt, cache=ucache)[:, -1, :]  # cfg-synced: feed same token
            logits = ulogits + cfg_scale * (logits - ulogits)
    return out


# ---------------------------------------------------------------------------
# text generation
# ---------------------------------------------------------------------------
def _chat_prompt(model: AudexModel, user_text: str, reasoning: bool = False) -> list:
    think = "" if reasoning else "<think></think>"
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}"
    )
    return model.tokenizer.encode(text, add_special_tokens=False)


def text_generate(model: AudexModel, prompt: str, *, max_new_tokens: int = 512,
                  temperature: float = 0.7, top_k: int = 0, top_p: float = 0.9,
                  reasoning: bool = False, seed: int | None = None) -> str:
    if seed is not None:
        mx.random.seed(seed)
    ids = _chat_prompt(model, prompt, reasoning=reasoning)
    gen = _generate_ids(model, ids, max_new_tokens, temperature, top_k, top_p, model.eos_ids)
    return model.tokenizer.decode(gen, skip_special_tokens=False).strip()


# ---------------------------------------------------------------------------
# text-to-speech
# ---------------------------------------------------------------------------
def _tts_prompt_text(transcription: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n<|text to speech|> Generate speech for this transcription. "
        f"{transcription}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think><speechgen_start>"
    )


def _build_null_prompt(model: AudexModel, cond_ids: list, transcription: str) -> list:
    """Match token length of cond by replacing the transcription with <unk> tokens."""
    tok = model.tokenizer
    target = len(cond_ids)
    def tmpl(null_text):
        return tok.encode(_tts_prompt_text(null_text), add_special_tokens=False)
    n_unk = max(1, target - len(tmpl("")))
    ids = tmpl("<unk>" * n_unk)
    for _ in range(64):
        if len(ids) == target:
            break
        n_unk += 1 if len(ids) < target else -1
        n_unk = max(1, n_unk)
        ids = tmpl("<unk>" * n_unk)
    return ids


def tts_generate(model: AudexModel, transcription: str, *, max_new_tokens: int = 1024,
                 temperature: float = 0.1, top_k: int = 80, top_p: float = 1.0,
                 cfg_scale: float = 1.0, seed: int | None = 0) -> np.ndarray:
    if seed is not None:
        mx.random.seed(seed)
    cond_ids = model.tokenizer.encode(_tts_prompt_text(transcription), add_special_tokens=False)
    uncond_ids = _build_null_prompt(model, cond_ids, transcription) if cfg_scale > 1.0 else None
    stop = [model.markers["speechgen_end"]] + model.eos_ids

    gen = _generate_ids(model, cond_ids, max_new_tokens, temperature, top_k, top_p,
                        stop, cfg_scale=cfg_scale, uncond_ids=uncond_ids)
    codes = [model.speech_codec[t] for t in gen if t in model.speech_codec]
    if not codes:
        return np.zeros(int(0.1 * model.decoder.cfg.sample_rate), dtype=np.float32)
    wav = model.decoder.decode(codes)
    mx.eval(wav)
    return np.array(wav, dtype=np.float32)
