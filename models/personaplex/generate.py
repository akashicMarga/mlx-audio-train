# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .lm import Lm
from .modules.conditioner import ConditionTensor
from .utils import sampling

SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]


class LmGen:
    def __init__(
        self,
        model: Lm,
        max_steps: int,
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
        batch_size: int = 1,
        cfg_coef: float = 1.0,
        check: bool = False,
        on_text_hook=None,
        on_audio_hook=None,
        audio_silence_frame_cnt: int = 1,
        cfg_is_masked_until: list[int] | None = None,
        cfg_is_no_text: bool = False,
    ):
        self.batch_size = batch_size
        self.model = model
        self.text_sampler = text_sampler
        self.audio_sampler = audio_sampler
        self.max_steps = max_steps
        self.check = check
        self.cfg_coef = cfg_coef
        self.on_text_hook = on_text_hook
        self.on_audio_hook = on_audio_hook
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.cfg_is_masked_until = cfg_is_masked_until
        self.cfg_is_no_text = cfg_is_no_text
        self.num_codebooks = 1 + model.cfg.audio_codebooks
        self.assistant_codebooks = model.cfg.audio_tokens_per_stream
        self.user_codebooks = model.cfg.audio_codebooks - self.assistant_codebooks
        self.delays = model.all_delays
        self.max_delay = max(self.delays)
        self.cache_len = self.max_delay + 3
        self.audio_padding_token = self.model.cfg.audio_padding_token
        self.zero_text_code = 3
        self.text_prompt_tokens: list[int] | None = None
        self.voice_prompt = None
        self.voice_prompt_cache: Optional[mx.array] = None
        self.voice_prompt_embeddings: Optional[list[mx.array]] = None
        self._init_state()

    def _init_state(self) -> None:
        self.cache = mx.full(
            (self.batch_size, self.num_codebooks, self.cache_len),
            self.ungenerated_token,
            dtype=mx.int32,
        )
        self.provided = mx.zeros(
            (self.batch_size, self.num_codebooks, self.cache_len),
            dtype=mx.bool_,
        )
        text_initial = mx.full(
            (self.batch_size, 1, 1),
            self.model.cfg.text_out_vocab_size,
            dtype=mx.int32,
        )
        audio_initial = mx.full(
            (self.batch_size, self.model.cfg.audio_codebooks, 1),
            self.audio_padding_token,
            dtype=mx.int32,
        )
        self.initial = mx.concatenate([text_initial, audio_initial], axis=1)
        self.step_idx = 0

    def reset_streaming(self) -> None:
        self._init_state()
        for c in self.model.transformer_cache:
            c.reset()
        for c in self.model.depformer_cache:
            c.reset()

    @property
    def zero_token(self) -> int:
        return -1

    @property
    def ungenerated_token(self) -> int:
        return -2

    def _encode_zero_frame(self) -> mx.array:
        if self.assistant_codebooks != len(SILENCE_TOKENS):
            raise ValueError(
                f"expected {len(SILENCE_TOKENS)} assistant codebooks, got {self.assistant_codebooks}"
            )
        return mx.array(SILENCE_TOKENS, dtype=mx.int32).reshape(1, self.assistant_codebooks, 1)

    def _encode_sine_frame(self) -> mx.array:
        if self.user_codebooks != len(SINE_TOKENS):
            raise ValueError(
                f"expected {len(SINE_TOKENS)} user codebooks, got {self.user_codebooks}"
            )
        return mx.array(SINE_TOKENS, dtype=mx.int32).reshape(1, self.user_codebooks, 1)

    def load_voice_prompt_embeddings(self, path: str) -> None:
        if path.endswith(".npz"):
            data = np.load(path)
            embedding_keys = sorted(
                [k for k in data.files if k.startswith("embedding_")],
                key=lambda k: int(k.split("_")[1]),
            )
            if not embedding_keys or "cache" not in data:
                raise RuntimeError(f"invalid voice prompt npz: {path}")
            self.voice_prompt = path
            self.voice_prompt_embeddings = [
                mx.array(data[k]).astype(mx.bfloat16) for k in embedding_keys
            ]
            self.voice_prompt_cache = mx.array(data["cache"], dtype=mx.int32)
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "loading PersonaPlex .pt voice prompts requires torch to be installed"
            ) from exc

        state = torch.load(path, map_location="cpu")
        embeddings_t = state["embeddings"].to(device="cpu", dtype=torch.float32)
        cache_t = state["cache"].to(device="cpu")
        embeddings = embeddings_t.numpy()
        cache = cache_t.numpy()
        self.voice_prompt = path
        self.voice_prompt_embeddings = [mx.array(emb).astype(mx.bfloat16) for emb in embeddings]
        self.voice_prompt_cache = mx.array(cache, dtype=mx.int32)

    def _prepare_step_input(
        self,
        input_tokens: mx.array | None = None,
        moshi_tokens: mx.array | None = None,
        text_token: mx.array | int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
        if self.step_idx >= self.max_steps:
            raise ValueError(f"reached max-steps {self.max_steps}")
        ct = self.cache_len

        if input_tokens is not None:
            if input_tokens.shape != (self.batch_size, self.user_codebooks, 1):
                raise ValueError(
                    f"expected input_tokens shape {(self.batch_size, self.user_codebooks, 1)}, got {input_tokens.shape}"
                )
            for q_other in range(self.user_codebooks):
                k = 1 + self.assistant_codebooks + q_other
                delay = self.delays[k]
                write_position = (self.step_idx + delay) % ct
                self.cache[:, k, write_position : write_position + 1] = input_tokens[
                    :, q_other
                ]
                self.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            if moshi_tokens.shape != (self.batch_size, self.assistant_codebooks, 1):
                raise ValueError(
                    f"expected moshi_tokens shape {(self.batch_size, self.assistant_codebooks, 1)}, got {moshi_tokens.shape}"
                )
            for q_moshi in range(self.assistant_codebooks):
                k = 1 + q_moshi
                delay = self.delays[k]
                write_position = (self.step_idx + delay) % ct
                self.cache[:, k, write_position : write_position + 1] = moshi_tokens[
                    :, q_moshi
                ]
                self.provided[:, k, write_position : write_position + 1] = True

        if text_token is not None:
            if isinstance(text_token, int):
                text_token = mx.full((self.batch_size,), text_token, dtype=mx.int32)
            elif text_token.shape == (self.batch_size, 1):
                text_token = text_token[:, 0]
            elif text_token.shape != (self.batch_size,):
                raise ValueError(
                    f"expected text_token shape {(self.batch_size,)} or {(self.batch_size, 1)}, got {text_token.shape}"
                )
            write_position = (self.step_idx + self.delays[0]) % ct
            self.cache[:, 0, write_position] = text_token
            self.provided[:, 0, write_position] = True

        for k, delay in enumerate(self.delays):
            if self.step_idx <= delay:
                cur = self.step_idx % ct
                self.cache[:, k, cur] = self.initial[:, k, 0]
                self.provided[:, k, cur] = True

        if self.step_idx == 0:
            self.cache[:, :, 0] = self.initial[:, :, 0]
            self.step_idx += 1
            return None

        model_input_position = (self.step_idx - 1) % ct
        target_position = self.step_idx % ct
        input_ = self.cache[:, :, model_input_position : model_input_position + 1]
        target_ = self.cache[:, :, target_position : target_position + 1]
        provided_ = self.provided[:, :, target_position : target_position + 1]

        if self.check and (input_ == self.ungenerated_token).any():
            raise ValueError(f"ungenerated token in model input at step {self.step_idx}")
        return input_, provided_, target_, model_input_position, target_position

    def _process_step_output(
        self,
        transformer_out: mx.array,
        text_logits: mx.array,
        provided_: mx.array,
        target_: mx.array,
        model_input_position: int,
        target_position: int,
    ) -> mx.array:
        sampled_text, _ = self.text_sampler(text_logits)
        sampled_text = sampled_text[:, 0]
        next_text = mx.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text)

        sampled_audio = self.model.depformer.sample(
            transformer_out,
            self.audio_sampler,
            next_text[:, None],
            self.model.depformer_cache,
            cfg_coef=self._effective_cfg_coef(),
            forced_audio_tokens=target_[:, 1:, 0],
            forced_audio_mask=provided_[:, 1:, 0],
        )
        if self.on_text_hook is not None:
            self.on_text_hook(sampled_text[:, None])
        if self.on_audio_hook is not None:
            self.on_audio_hook(sampled_audio)

        self.provided[:, :, model_input_position] = False

        target_text = self.cache[:, 0, target_position]
        self.cache[:, 0, target_position] = mx.where(
            self.provided[:, 0, target_position],
            target_text,
            sampled_text,
        )

        generated_cb = sampled_audio.shape[1]
        old_audio = self.cache[:, 1 : generated_cb + 1, target_position]
        mask = self.provided[:, 1 : generated_cb + 1, target_position]
        self.cache[:, 1 : generated_cb + 1, target_position] = mx.where(
            mask,
            old_audio,
            sampled_audio[:, :, 0],
        )

        self.step_idx += 1
        return sampled_text[:, None]

    def _effective_cfg_coef(self) -> float:
        """Return the effective CFG coefficient for the current step.

        When cfg_is_masked_until is set, CFG is disabled (coef=1.0)
        until the step reaches the masked-until threshold for all batch items.
        """
        if self.cfg_coef == 1.0:
            return 1.0
        if self.cfg_is_masked_until is not None:
            if all(self.step_idx < until for until in self.cfg_is_masked_until):
                return 1.0
        return self.cfg_coef

    def step(
        self,
        other_audio_tokens: mx.array | None = None,
        ct: ConditionTensor | None = None,
        cross_attention_src: mx.array | None = None,
        input_tokens: mx.array | None = None,
        moshi_tokens: mx.array | None = None,
        text_token: mx.array | int | None = None,
    ) -> mx.array | None:
        """Run one streaming step.

        Note: CFG (cfg_coef != 1) is only supported via the _sample() path
        used by TTSModel.generate(). The streaming step() path used by
        local/web inference does not apply CFG — it relies on cfg_coef=1.0.
        """
        if input_tokens is None:
            input_tokens = other_audio_tokens
        prepared = self._prepare_step_input(
            input_tokens=input_tokens,
            moshi_tokens=moshi_tokens,
            text_token=text_token,
        )
        if prepared is None:
            return None
        input_, provided_, target_, model_input_position, target_position = prepared
        xs = self.model.embed_codes(input_, ct=ct)
        transformer_out, text_logits = self.model.forward_embeddings(
            xs, cross_attention_src=cross_attention_src
        )
        return self._process_step_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
        )

    def step_embeddings(self, embeddings: mx.array) -> mx.array | None:
        if embeddings.shape[0] != self.batch_size:
            raise ValueError(
                f"expected embeddings batch {self.batch_size}, got {embeddings.shape[0]}"
            )
        # Match PyTorch LMGen.step_embeddings(): replay embeddings while forcing
        # both audio streams with the model's initial audio token.
        dummy_audio = mx.full(
            (self.batch_size, self.model.cfg.audio_codebooks, 1),
            self.audio_padding_token,
            dtype=mx.int32,
        )
        dummy_input = dummy_audio[:, self.assistant_codebooks :, :]
        dummy_moshi = dummy_audio[:, : self.assistant_codebooks, :]
        while True:
            prepared = self._prepare_step_input(
                input_tokens=dummy_input,
                moshi_tokens=dummy_moshi,
                text_token=self.zero_text_code,
            )
            if prepared is not None:
                break
        _, provided_, target_, model_input_position, target_position = prepared
        transformer_out, text_logits = self.model.forward_embeddings(embeddings)
        return self._process_step_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
        )

    def step_system_prompts(self) -> None:
        if self.voice_prompt_embeddings is not None:
            for emb in self.voice_prompt_embeddings:
                self.step_embeddings(emb)
            if self.voice_prompt_cache is not None:
                self.cache = self.voice_prompt_cache.astype(mx.int32)

        for _ in range(self.audio_silence_frame_cnt):
            self.step(
                input_tokens=self._encode_sine_frame(),
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
            )

        if self.text_prompt_tokens:
            for tok in self.text_prompt_tokens:
                self.step(
                    input_tokens=self._encode_sine_frame(),
                    moshi_tokens=self._encode_zero_frame(),
                    text_token=int(tok),
                )

        for _ in range(self.audio_silence_frame_cnt):
            self.step(
                input_tokens=self._encode_sine_frame(),
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
            )

    def last_audio_tokens(self) -> Optional[mx.array]:
        gen_idx = self.step_idx - 1 - self.max_delay
        if gen_idx < 0:
            return None
        tokens: list[mx.array] = []
        for q in range(self.assistant_codebooks):
            k = 1 + q
            pos = (gen_idx + self.delays[k]) % self.cache_len
            tokens.append(self.cache[:, k, pos : pos + 1])
        out = mx.concatenate(tokens, axis=1)
        if (out == self.audio_padding_token).any():  # type: ignore
            return None
        if (out == self.ungenerated_token).any():  # type: ignore
            raise ValueError(f"ungenerated value in last-audio tokens at step {self.step_idx}")
        return out

    def resolve_voice_prompt_path(self, voice_prompt: str, voice_prompt_dir: str | None) -> Path:
        name = voice_prompt
        if not name.endswith(".pt") and "." not in Path(name).name:
            name = f"{name}.pt"
        path = Path(name)
        if not path.is_absolute():
            if voice_prompt_dir is not None:
                path = Path(voice_prompt_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"voice prompt not found: {path}")
        return path
