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
from .audio_encoder import AudioEncoderConfig, AudioTower
from . import features as feat

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
    audio_tower: object = None  # AudioTower or None (text/TTS-only checkpoint)
    sound_token_id: int = None  # <so_embedding>


def _quantize(module, bits: int, group_size: int):
    """Quantize a module's Linear/Embedding layers whose in-dim divides group_size.

    embed_positions is excluded: it is read as a dense `.weight` in the encoder
    forward, so a QuantizedEmbedding (packed weight) would break it.
    """
    import mlx.nn as nn

    def predicate(path, m):
        if "embed_positions" in path:
            return False
        return (isinstance(m, (nn.Linear, nn.Embedding))
                and m.weight.shape[-1] % group_size == 0)

    nn.quantize(module, group_size=group_size, bits=bits, class_predicate=predicate)


def load_model(model_dir: str, *, quantize: int = None, q_group_size: int = 64,
               quantize_audio: bool = False, quantize_decoder: bool = False) -> AudexModel:
    """Load the MLX Audex checkpoint.

    quantize: bits (4 or 8) to quantize the LM on a bf16 checkpoint, or None.
    The LM dominates memory and decode time so it is the default target; the
    speech decoder and audio encoder stay bf16 unless quantize_decoder/
    quantize_audio are set.

    If the checkpoint dir contains quant.json (a pre-quantized checkpoint, e.g.
    from scripts/quantize_audex.py), the stored quantization is applied to the
    listed targets and the `quantize` argument is ignored.
    """
    from transformers import AutoTokenizer, logging as hf_logging

    model_dir = Path(model_dir)
    lm_cfg = LMConfig(**json.loads((model_dir / "lm_config.json").read_text()))
    dec_cfg = SpeechDecoderConfig(**json.loads((model_dir / "speech_decoder_config.json").read_text()))

    # Pre-quantized checkpoint? Apply structure before loading weights.
    quant = None
    qpath = model_dir / "quant.json"
    if qpath.exists():
        quant = json.loads(qpath.read_text())

    def _load(module, path, name, runtime_q):
        pre = quant and name in quant.get("targets", [])
        if pre:
            _quantize(module, quant["bits"], quant["group_size"])
            module.load_weights(str(path))
        else:
            module.load_weights(str(path))
            if not quant and runtime_q:
                _quantize(module, quantize, q_group_size)
        module.eval()

    lm = NemotronDenseForCausalLM(lm_cfg)
    _load(lm, model_dir / "lm.safetensors", "lm", quantize)

    decoder = AudexSpeechDecoder(dec_cfg)
    _load(decoder, model_dir / "speech_decoder.safetensors", "decoder", quantize and quantize_decoder)

    to_eval = [lm.parameters(), decoder.parameters()]

    audio_tower = None
    audio_path = model_dir / "audio.safetensors"
    audio_cfg_path = model_dir / "audio_config.json"
    if audio_path.exists() and audio_cfg_path.exists():
        aud_cfg = AudioEncoderConfig(**json.loads(audio_cfg_path.read_text()))
        audio_tower = AudioTower(aud_cfg)
        _load(audio_tower, audio_path, "audio", quantize and quantize_audio)
        to_eval.append(audio_tower.parameters())

    mx.eval(*to_eval)

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    speech_codec, markers = _build_codec_maps(tokenizer)
    eos_ids = _resolve_eos(tokenizer)
    sound_token_id = tokenizer.convert_tokens_to_ids("<so_embedding>")
    if quant:
        qdesc = f"{quant['bits']}-bit ({'+'.join(quant['targets'])})"
    elif quantize:
        qdesc = f"{quantize}-bit LM"
        if quantize_audio or quantize_decoder:
            qdesc += f" (+{'audio' if quantize_audio else ''}{'/decoder' if quantize_decoder else ''})"
    else:
        qdesc = "bf16"
    print(f"[audex] Ready [{qdesc}]. {len(speech_codec)} speechcodec tokens, "
          f"audio_understanding={'on' if audio_tower else 'off'}, eos={eos_ids}")
    return AudexModel(lm, decoder, tokenizer, speech_codec, markers, eos_ids,
                      audio_tower=audio_tower, sound_token_id=sound_token_id)


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


# ---------------------------------------------------------------------------
# audio understanding (speech -> text) and speech-to-speech
# ---------------------------------------------------------------------------
def encode_audio(model: AudexModel, audio: np.ndarray) -> mx.array:
    """audio waveform -> projected audio embeddings [num_clips*750, 2048]."""
    if model.audio_tower is None:
        raise RuntimeError("This checkpoint has no audio encoder (audio.safetensors missing). "
                           "Re-run `python -m models.audex.convert` to include it.")
    features = feat.extract_features(audio)          # (clips, 128, 3000)
    projected = model.audio_tower(features)          # (clips, 750, 2048)
    n = projected.shape[0] * projected.shape[1]
    return projected.reshape(n, projected.shape[-1])


def _understanding_embeds(model: AudexModel, audio: np.ndarray, instruction: str):
    """Build inputs_embeds with audio injected at <so_embedding> positions."""
    n_embed = feat.EMBEDDINGS_PER_CLIP * feat.extract_features(audio).shape[0]
    sound = "<so_start>" + ("<so_embedding>" * n_embed) + "<so_end>"
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{sound}\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n<think></think>"
    )
    ids = mx.array(model.tokenizer.encode(text, add_special_tokens=False), dtype=mx.int32)
    embeds = model.lm.embed(ids[None])[0]            # [L, 2048]
    audio_emb = encode_audio(model, audio).astype(embeds.dtype)

    mask = ids == model.sound_token_id
    n_slots = int(mask.sum().item())
    if n_slots != audio_emb.shape[0]:
        raise ValueError(f"<so_embedding> slots ({n_slots}) != audio tokens ({audio_emb.shape[0]})")
    # scatter audio embeddings into the placeholder rows (order preserved)
    pos = mx.array(np.where(np.array(mask))[0])
    embeds[pos] = audio_emb
    return embeds[None]                              # [1, L, 2048]


def _decode_from_embeds(model: AudexModel, inputs_embeds, max_new_tokens,
                        temperature, top_k, top_p, stop_ids):
    lm = model.lm
    cache = lm.make_cache()
    logits = lm(inputs_embeds=inputs_embeds, cache=cache)[:, -1, :]
    out, stop_set = [], set(stop_ids)
    for _ in range(max_new_tokens):
        tok = _sample(logits[0], temperature, top_k, top_p)
        if tok in stop_set:
            break
        out.append(tok)
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache)[:, -1, :]
    return out


def audio_generate(model: AudexModel, audio, instruction: str = "", *,
                   max_new_tokens: int = 512, temperature: float = 0.0,
                   top_k: int = 0, top_p: float = 1.0, seed: int | None = None) -> str:
    """Speech (+ optional text instruction) -> text response/transcription."""
    if seed is not None:
        mx.random.seed(seed)
    if isinstance(audio, str):
        audio = feat.load_audio(audio)
    embeds = _understanding_embeds(model, audio, instruction)
    gen = _decode_from_embeds(model, embeds, max_new_tokens, temperature, top_k, top_p, model.eos_ids)
    text = model.tokenizer.decode(gen, skip_special_tokens=False)
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


ASR_PROMPT = "Transcribe the input speech."
# Conversational persona for the reply turn (from NVIDIA's cascaded_s2s reference).
S2S_TEXT_PROMPT = (
    "You are Audex, a helpful voice assistant. Respond to the user's message "
    "conversationally. Write in plain, unformatted prose — no markdown, bullet "
    "points, lists, or headers — suitable for reading aloud."
)


def _clean_transcription(text: str) -> str:
    """Model returns e.g. ...is 'the actual words'. -> extract the quoted span."""
    a, b = text.find("'"), text.rfind("'")
    return text[a + 1:b].strip() if a != -1 and b > a else text.strip()


def transcribe(model: AudexModel, audio, *, max_new_tokens: int = 256) -> str:
    """Speech -> text (ASR)."""
    raw = audio_generate(model, audio, ASR_PROMPT, max_new_tokens=max_new_tokens, temperature=0.0)
    return _clean_transcription(raw)


def s2s_generate(model: AudexModel, audio, instruction: str = "", *,
                 max_new_tokens: int = 512, tts_cfg_scale: float = 1.0,
                 seed: int | None = 0, return_transcript: bool = False):
    """Full speech-to-speech (cascaded, same model throughout):

        speech --ASR--> transcript --text chat--> reply --TTS--> speech

    Returns (reply_text, waveform@16kHz), or (transcript, reply_text, waveform)
    if return_transcript=True. `instruction` prepends extra guidance to the text
    turn (leave empty to just answer the spoken query).
    """
    transcript = transcribe(model, audio)
    persona = f"{S2S_TEXT_PROMPT}\n\n{instruction}".strip() if instruction else S2S_TEXT_PROMPT
    user_turn = f"{persona}\n\n{transcript}"
    reply = text_generate(model, user_turn, max_new_tokens=max_new_tokens, temperature=0.7, seed=seed)
    wav = tts_generate(model, reply, cfg_scale=tts_cfg_scale, seed=seed)
    return (transcript, reply, wav) if return_transcript else (reply, wav)
