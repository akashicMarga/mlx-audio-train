"""
MiniMindOmni — full speech-to-speech model in MLX.

Thinker (semantic LM) + Talker (acoustic head) + SenseVoice + Mimi.
Faithful MLX port of minimind-o's MiniMindOmni.

Upstream: https://github.com/jingyaogong/minimind-o

Forward input shape: (batch, 9, T)
  channels 0-7: audio code history (one per Mimi codebook)
  channel   8:  text token IDs

Forward outputs:
  text_logits:  (batch, seq, vocab_size)
  audio_logits: list of 8 tensors, each (batch, seq, audio_vocab_size)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.minimind_o.config import OmniConfig
from models.minimind_o.projectors import MMAudioProjector, MMVisionProjector
from models.minimind_o.speech_encoder import load_sensevoice
from models.minimind_o.talker import TalkerModule
from models.minimind_o.thinker import MiniMindModel


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_token(
    logits_np: np.ndarray,
    temperature: float = 0.85,
    top_p: float = 0.85,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    seen_tokens: Optional[List[int]] = None,
) -> int:
    logits = logits_np.astype(np.float32) / max(temperature, 1e-9)

    if repetition_penalty != 1.0 and seen_tokens:
        for tok in set(seen_tokens):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty

    if top_k > 0 and top_k < len(logits):
        thresh = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < thresh, -np.inf, logits)

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-9

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p)
        keep = sorted_idx[: cutoff + 1]
        mask = np.zeros_like(probs)
        mask[keep] = 1.0
        probs = probs * mask
        probs /= probs.sum() + 1e-9

    return int(np.random.choice(len(probs), p=probs))


def _sample_audio_code(logits_np: np.ndarray, prev_codes: List[int], temperature: float = 0.2) -> int:
    """Sample a Mimi code with light repetition penalty on recent tokens."""
    logits = logits_np.astype(np.float32) / max(temperature, 1e-9)
    for code in prev_codes[-3:]:
        if logits[code] > 0:
            logits[code] /= 1.05
        else:
            logits[code] *= 1.05

    top_k = 50
    if top_k < len(logits):
        thresh = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < thresh, -np.inf, logits)

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-9
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MiniMindOmni(nn.Module):
    """
    Speech-to-speech model.

    Architecture:
        [SenseVoice (frozen PyTorch)] → MMAudioProjector (MLX)
                    ↓ injected into text embedding stream
        [Thinker: MiniMindModel, 8 layers] — produces text logits
                    ↓ bridge layer hidden states
        [Talker: TalkerModule, 4 layers]  — produces 8 Mimi codebook logits
                    ↓
        [Mimi decoder (PyTorch/HF)]       — reconstructs audio waveform
    """

    def __init__(
        self,
        config: Optional[OmniConfig] = None,
        audio_encoder_path: str = "./model/SenseVoiceSmall",
        vision_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or OmniConfig()

        self.model = MiniMindModel(self.config)  # kept as "model" to match weight keys
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            pass

        self.talker = TalkerModule(self.config)

        self.audio_proj = MMAudioProjector(self.config.audio_hidden_size, self.config.hidden_size)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size)

        self.audio_encoder = None
        self.audio_processor = None
        if audio_encoder_path:
            self.audio_encoder, self.audio_processor = load_sensevoice(audio_encoder_path)

        self.vision_encoder = None
        self.vision_processor = None
        if vision_model_path and os.path.exists(vision_model_path):
            self._load_vision(vision_model_path)

    # ------------------------------------------------------------------
    # Vision encoder (optional, PyTorch)
    # ------------------------------------------------------------------

    def _load_vision(self, path: str) -> None:
        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            import transformers.utils.logging as hf_log
            hf_log.set_verbosity_error()
            model = SiglipVisionModel.from_pretrained(path)
            for p in model.parameters():
                p.requires_grad = False
            self.vision_encoder = model.eval()
            self.vision_processor = SiglipImageProcessor.from_pretrained(path)
        except Exception as e:
            warnings.warn(f"[Vision] Failed to load: {e}")

    # ------------------------------------------------------------------
    # Audio encoding: SenseVoice (PyTorch) → MMAudioProjector (MLX)
    # ------------------------------------------------------------------

    def encode_audio_inputs(
        self, audio_inputs, audio_lens=None
    ) -> Optional[List[Optional[mx.array]]]:
        """
        audio_inputs: torch.Tensor (batch, T, 560) fbank features
        Returns list of mx.array per batch item (or None for empty items).
        """
        import torch

        if audio_inputs is None or self.audio_encoder is None:
            return None
        if not audio_inputs.any():
            return None

        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)

        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor(
                [valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device
            )

        with torch.no_grad():
            enc_out, _ = self.audio_encoder(valid_fbank, valid_lens)  # (valid, T, 512)

        emb_list: List[mx.array] = []
        for i in range(enc_out.size(0)):
            length = max(1, min(int(valid_lens[i].item()), enc_out.size(1)))
            feat_np = enc_out[i, :length].float().cpu().numpy()  # (T, 512)
            feat_mx = mx.array(feat_np)[None, ...]                # (1, T, 512)
            projected = self.audio_proj(feat_mx).squeeze(0)       # (T, hidden)
            emb_list.append(projected)

        if batch_mask.all():
            return emb_list
        out: List[Optional[mx.array]] = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    def inject_audio_features(
        self,
        tokens: mx.array,
        hidden: mx.array,
        audio_feats: Optional[List],
        seq_len: int,
    ) -> mx.array:
        """Replace <|audio_pad|> token positions with projected SenseVoice features."""
        if audio_feats is None or not self.config.audio_ids:
            return hidden

        marker = self.config.audio_ids[0]
        batch_hidden = []
        tokens_np = np.array(tokens.tolist())

        for b in range(hidden.shape[0]):
            hb = hidden[b]          # (seq, hidden)
            seq = tokens_np[b].tolist()
            af = audio_feats[b] if audio_feats[b] is not None else None
            i = 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.shape[0], i - start)
                        parts = []
                        if start > 0:
                            parts.append(hb[:start])
                        parts.append(af[:inject_len])
                        parts.append(hb[start + inject_len:])
                        hb = mx.concatenate(parts, axis=0)
                        af = None
                else:
                    i += 1
            batch_hidden.append(hb)

        return mx.stack(batch_hidden, axis=0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        audio_inputs=None,
        audio_lens=None,
        pixel_values=None,
        spk_emb: Optional[mx.array] = None,
        mlx_audio_feats: Optional[List[Optional[mx.array]]] = None,
        mlx_vision_feats: Optional[mx.array] = None,
    ) -> Dict[str, Any]:
        # Unpack 9-channel input: channels 0-7 = audio, channel 8 = text
        if input_ids.ndim == 2:
            batch_size, seq_len = input_ids.shape
            text_ids = input_ids
            audio_ids = mx.full(
                (batch_size, 8, seq_len), self.config.audio_pad_token, dtype=mx.int32
            )
        else:
            batch_size, _, seq_len = input_ids.shape
            text_ids = input_ids[:, 8, :]
            audio_ids = input_ids[:, :8, :]

        n_thinker = len(self.model.layers)
        n_talker = len(self.talker.layers)
        total_layers = n_thinker + n_talker
        past_key_values = past_key_values or [None] * total_layers

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # ===== Thinker forward =====
        hidden_states = self.model.dropout(self.model.embed_tokens(text_ids))

        if start_pos == 0:
            if mlx_audio_feats is not None:
                hidden_states = self.inject_audio_features(text_ids, hidden_states, mlx_audio_feats, seq_len)
            elif audio_inputs is not None:
                audio_feats = self.encode_audio_inputs(audio_inputs, audio_lens)
                hidden_states = self.inject_audio_features(text_ids, hidden_states, audio_feats, seq_len)

        if (pixel_values is not None or mlx_vision_feats is not None) and start_pos == 0:
            hidden_states = self._inject_vision(text_ids, hidden_states, pixel_values, seq_len, mlx_vision_feats)

        cos = mx.array(self.model._freqs_cos[start_pos: start_pos + seq_len])
        sin = mx.array(self.model._freqs_sin[start_pos: start_pos + seq_len])

        bridge_states = hidden_states
        thinker_presents: List = []
        for i, (layer, past_kv) in enumerate(
            zip(self.model.layers, past_key_values[:n_thinker])
        ):
            hidden_states, present = layer(
                hidden_states, cos, sin, past_kv, use_cache, attention_mask
            )
            thinker_presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states

        h_thinker = self.model.norm(hidden_states)

        # ===== Talker forward =====
        audio_logits, talker_presents = self.talker(
            bridge_states=bridge_states,
            audio_ids=audio_ids,
            start_pos=start_pos,
            past_key_values=past_key_values[n_thinker:],
            use_cache=use_cache,
            attention_mask=attention_mask,
            spk_emb=spk_emb,
            audio_spk_token=self.config.audio_spk_token,
        )

        all_presents = thinker_presents + talker_presents

        aux = mx.array(0.0)
        if self.config.use_moe:
            from models.minimind_o.thinker import MoEFeedForward
            for block in list(self.model.layers) + list(self.talker.layers):
                if isinstance(block.mlp, MoEFeedForward):
                    aux = aux + block.mlp.aux_loss

        sl = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        text_logits = self.lm_head(h_thinker[:, sl, :])

        if logits_to_keep > 0:
            audio_logits = [al[:, sl, :] for al in audio_logits]

        return {
            "logits": text_logits,
            "audio_logits": audio_logits,
            "past_key_values": all_presents,
            "aux_loss": aux,
        }

    # ------------------------------------------------------------------
    # Vision injection helper
    # ------------------------------------------------------------------

    def _inject_vision(self, tokens, hidden, pixel_values, seq_len, mlx_vision_feats=None):
        # MLX-native path: pre-encoded features passed directly
        if mlx_vision_feats is not None:
            vision_tensors = self.vision_proj(mlx_vision_feats)
        elif self.vision_encoder is not None:
            import torch
            if hasattr(pixel_values, "keys"):
                pv = {k: v for k, v in pixel_values.items()}
                with torch.no_grad():
                    img_emb = self.vision_encoder(**pv).last_hidden_state
            else:
                with torch.no_grad():
                    img_emb = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
            img_emb_np = img_emb.float().cpu().numpy()
            vision_tensors = self.vision_proj(mx.array(img_emb_np))
        else:
            return hidden

        marker = self.config.image_ids[0]
        tokens_np = np.array(tokens.tolist())
        batch_hidden = []
        for b in range(hidden.shape[0]):
            hb = hidden[b]
            seq = tokens_np[b].tolist()
            vt = vision_tensors[b] if vision_tensors.ndim == 3 else vision_tensors
            i = k = 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vt.shape[0]:
                        n = i - start
                        parts = []
                        if start > 0:
                            parts.append(hb[:start])
                        parts.append(vt[:n])
                        parts.append(hb[i:])
                        hb = mx.concatenate(parts, axis=0)[:seq_len]
                        k += 1
                else:
                    i += 1
            batch_hidden.append(hb)
        return mx.stack(batch_hidden, axis=0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: mx.array,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.90,
        stream: bool = False,
        rp: float = 1.0,
        use_cache: bool = True,
        return_audio_codes: bool = False,
        **kwargs,
    ):
        gen = self.stream_generate(
            input_ids, eos_token_id, max_new_tokens, temperature, top_p,
            rp, use_cache, return_audio_codes, **kwargs
        )
        if stream:
            return gen
        tokens = list(gen)
        return tokens[-1] if tokens else (input_ids, None)

    def stream_generate(
        self,
        input_ids: mx.array,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.90,
        rp: float = 1.0,
        use_cache: bool = True,
        return_audio_codes: bool = False,
        **kwargs,
    ) -> Generator:
        """
        Auto-regressive generation — yields (text_ids, audio_frame_or_None) per step.

        audio_frame: list of 8 Mimi code integers for one 80ms frame, or None.
        Staggered codebook generation: codebook i starts at step i+1.
        """
        start_pos = input_ids.shape[1]
        past_kvs = None
        text_finished = False
        first_finished = True
        open_thinking = kwargs.get("open_thinking", False)

        audio_codes: List[List[int]] = [[] for _ in range(8)]
        audio_stop_pos: List[Optional[int]] = [None] * 8

        audio_buffer = mx.full(
            (1, 8, start_pos), self.config.audio_pad_token, dtype=mx.int32
        )

        spk_emb = kwargs.get("spk_emb", None)
        ref_codes = kwargs.get("ref_codes", None)  # (1, 8, ref_frames)
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0

        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            ref_np = np.array(ref_codes.tolist())
            ab_np = np.full((1, 8, start_pos), self.config.audio_pad_token, dtype=np.int32)
            ab_np[:, :, fill_start:fill_end] = ref_np[:, :, -(fill_end - fill_start):]
            audio_buffer = mx.array(ab_np)
        if spk_emb is not None and fill_start > 0:
            ab_np = np.array(audio_buffer.tolist())
            ab_np[:, :, fill_start - 1] = self.config.audio_spk_token
            audio_buffer = mx.array(ab_np)

        think_end_step: Optional[int] = None
        generated_tokens: Optional[List[int]] = [] if open_thinking else None

        enter_token_id = kwargs.get("enter_token_id", 201)
        pad_token_id = kwargs.get("pad_token_id", 0)

        while input_ids.shape[1] < start_pos + max_new_tokens:
            nine_ch = mx.concatenate(
                [audio_buffer, input_ids[:, None, :] if input_ids.ndim == 2 else input_ids],
                axis=1,
            ) if input_ids.ndim == 2 else mx.concatenate(
                [audio_buffer, mx.expand_dims(input_ids, 1)], axis=1
            )

            if past_kvs is None or not use_cache:
                out = self(nine_ch, past_key_values=past_kvs, use_cache=use_cache, **{
                    k: v for k, v in kwargs.items()
                    if k in ("audio_inputs", "audio_lens", "pixel_values", "spk_emb", "mlx_audio_feats", "mlx_vision_feats")
                })
            else:
                nine_ch_last = mx.concatenate(
                    [audio_buffer[:, :, -1:], mx.expand_dims(input_ids[:, -1:], 1)], axis=1
                )
                out = self(nine_ch_last, past_key_values=past_kvs, use_cache=use_cache, **{
                    k: v for k, v in kwargs.items()
                    if k in ("spk_emb",)
                })

            mx.eval(out["logits"])
            past_kvs = out["past_key_values"] if use_cache else None

            text_logits_np = np.array(out["logits"][0, -1, :].tolist(), dtype=np.float32)
            if text_finished:
                text_token = enter_token_id if first_finished else pad_token_id
                first_finished = False
            else:
                seen = input_ids[0].tolist()
                text_token = _sample_token(
                    text_logits_np, temperature, top_p, top_k=50, repetition_penalty=rp,
                    seen_tokens=seen,
                )

            step = input_ids.shape[1] - start_pos

            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and (
                    generated_tokens[-len(self.config.think_end_ids):]
                    == list(self.config.think_end_ids)
                ):
                    think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1
            else:
                audio_step = step - 1

            mx.eval(out["audio_logits"][0])
            for i, al in enumerate(out["audio_logits"]):
                if audio_step < i:
                    audio_codes[i].append(self.config.audio_pad_token)
                else:
                    al_np = np.array(al[0, -1, :].tolist(), dtype=np.float32)
                    code = _sample_audio_code(al_np, audio_codes[i])
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active = sum(
                    1 for i in range(8)
                    if audio_stop_pos[i] is None
                    or step - 7 + i < audio_stop_pos[i]
                )
                if active >= 8:
                    audio_frame = frame

            new_text = mx.array([[text_token]], dtype=mx.int32)
            input_ids = mx.concatenate([input_ids, new_text], axis=1)

            new_audio = mx.full((1, 8, 1), self.config.audio_pad_token, dtype=mx.int32)
            audio_buffer = mx.concatenate([audio_buffer, new_audio], axis=2)
            for i in range(min(audio_step + 1, 8)):
                ab_np = np.array(audio_buffer.tolist())
                ab_np[0, i, -1] = audio_codes[i][-1]
                audio_buffer = mx.array(ab_np)

            if text_finished and all(s is not None for s in audio_stop_pos):
                break

            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights_from_pt(
        self, checkpoint_path: str, strict: bool = False, device: str = "cpu"
    ) -> None:
        """Load weights from a minimind-o .pth checkpoint, converting key names."""
        import torch

        raw = torch.load(checkpoint_path, map_location=device)
        if "model" in raw and not isinstance(raw, dict):
            state_dict = raw
        elif isinstance(raw, dict) and any(k.startswith("model.") or k.startswith("lm_head") for k in raw):
            state_dict = raw
        else:
            state_dict = raw.get("model", raw)

        weights = convert_pt_to_mlx(state_dict)
        self.load_weights(list(weights.items()), strict=strict)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        audio_encoder_path: str = "./model/SenseVoiceSmall",
        vision_model_path: Optional[str] = None,
        config: Optional[OmniConfig] = None,
    ) -> "MiniMindOmni":
        """Load model from a directory containing a .pth checkpoint."""
        import glob

        if config is None:
            config = OmniConfig()

        model = cls(config, audio_encoder_path, vision_model_path)

        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in {model_dir}")
        checkpoint = sorted(pth_files)[-1]
        print(f"Loading weights from {checkpoint}")
        model.load_weights_from_pt(checkpoint)
        return model

    def count_params(self, trainable_only: bool = True) -> int:
        total = 0
        for name, val in self.parameters().items():
            if "audio_encoder" in name or "vision_encoder" in name:
                continue
            if hasattr(val, "size"):
                total += val.size
            elif hasattr(val, "__len__"):
                total += len(val)
        return total


# ---------------------------------------------------------------------------
# Weight key converter: minimind-o .pth → MLX naming
# ---------------------------------------------------------------------------

def convert_pt_to_mlx(state_dict: dict) -> dict:
    """
    Convert a minimind-o PyTorch state dict to MLX parameter names.

    Key renames:
      model.*              → model.*          (thinker, same prefix)
      lm_head.*            → lm_head.*        (same)
      talker.codec_proj.0  → talker.codec_proj.fc1
      talker.codec_proj.2  → talker.codec_proj.fc2
      talker.codec_proj.3  → talker.codec_proj.norm
      talker.embed_proj.*  → talker.embed_proj.*  (same pattern)
      talker.embed_tokens.adapters.N.0.weight → talker.embed_tokens.adapters.N.emb.weight
      talker.embed_tokens.adapters.N.2.weight → talker.embed_tokens.adapters.N.proj.weight
      talker.lm_head.adapters.N.0.weight → talker.lm_head.adapters.N.fc1.weight
      talker.lm_head.adapters.N.2.weight → talker.lm_head.adapters.N.fc2.weight
      audio_proj.mlp.0     → audio_proj.ln
      audio_proj.mlp.1     → audio_proj.fc1
      audio_proj.mlp.3     → audio_proj.fc2
      vision_proj.*        → vision_proj.*  (same pattern)
    """
    import re

    result = {}
    for k, v in state_dict.items():
        if k.startswith("audio_encoder.") or k.startswith("vision_encoder."):
            continue

        if hasattr(v, "numpy"):
            arr = v.float().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arr = v.astype(np.float32)
        else:
            continue

        new_k = k

        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.0\.", r"talker.\1.fc1.", new_k)
        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.2\.", r"talker.\1.fc2.", new_k)
        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.3\.", r"talker.\1.norm.", new_k)

        new_k = re.sub(
            r"talker\.embed_tokens\.adapters\.(\d+)\.0\.weight",
            r"talker.embed_tokens.adapters.\1.emb.weight",
            new_k,
        )
        new_k = re.sub(
            r"talker\.embed_tokens\.adapters\.(\d+)\.2\.weight",
            r"talker.embed_tokens.adapters.\1.proj.weight",
            new_k,
        )

        new_k = re.sub(
            r"talker\.lm_head\.adapters\.(\d+)\.0\.weight",
            r"talker.lm_head.adapters.\1.fc1.weight",
            new_k,
        )
        new_k = re.sub(
            r"talker\.lm_head\.adapters\.(\d+)\.2\.weight",
            r"talker.lm_head.adapters.\1.fc2.weight",
            new_k,
        )

        new_k = re.sub(r"audio_proj\.mlp\.0\.", "audio_proj.ln.", new_k)
        new_k = re.sub(r"audio_proj\.mlp\.1\.", "audio_proj.fc1.", new_k)
        new_k = re.sub(r"audio_proj\.mlp\.3\.", "audio_proj.fc2.", new_k)

        new_k = re.sub(r"vision_proj\.mlp\.0\.", "vision_proj.ln.", new_k)
        new_k = re.sub(r"vision_proj\.mlp\.1\.", "vision_proj.fc1.", new_k)
        new_k = re.sub(r"vision_proj\.mlp\.3\.", "vision_proj.fc2.", new_k)

        result[new_k] = arr

    return result
