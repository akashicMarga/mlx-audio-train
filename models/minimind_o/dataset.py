"""
OmniDataset — training data loader for MiniMind-O, ported from minimind-o.

Reads Parquet files with columns:
  conversations   — JSON chat turns
  question_audios — list of audio bytes (per user turn)
  answer_audios   — flat list of Mimi code ints (interleaved 8 codebooks per frame)
  image_bytes     — optional image bytes
  ref_audios      — optional reference audio codes for voice cloning
  spk_emb         — optional CAM++ speaker embedding floats

Returns (input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb)
where input_ids is shape (9, T): channels 0-7 = audio codes, channel 8 = text.
"""

from __future__ import annotations

import io
import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from PIL import Image
from scipy.signal import resample
from torch.utils.data import Dataset

import pyarrow as pa
import pyarrow.parquet as pq
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Chat prompt helpers (ported verbatim)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
]


def _pre_processing_chat(conversations, add_system_ratio: float = 0.2):
    if any(c.get("tools") for c in conversations):
        return conversations
    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(_SYSTEM_PROMPTS)}] + conversations
    return conversations


def _post_processing_chat(prompt: str, empty_think_ratio: float = 0.2) -> str:
    if "<think>\n\n</think>\n\n" in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    return prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OmniDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        audio_processor=None,
        vision_processor=None,
        max_length: int = 1200,
        audio_special_token: str = "<|audio_pad|>",
        image_special_token: str = "<|image_pad|>",
        audio_stop_token: int = 2050,
        audio_pad_token: int = 2049,
        audio_spk_token: int = 2051,
        audio_vocab_size: int = 2112,
        scheduled_sampling: float = 0.05,
        image_token_len: int = 64,
    ):
        super().__init__()
        tables = [
            pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches())
            for p in data_path.split(",")
        ]
        tables = [
            t.cast(
                pa.schema(
                    [
                        f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f
                        for f in t.schema
                    ]
                )
            )
            for t in tables
        ]
        self.table = pa.concat_tables(tables, promote_options="default")
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.table)

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def process_audio(audio_path: str, audio_processor) -> Tuple[torch.Tensor, int]:
        """Load audio from file → fbank (T, 560) and valid encoder frame count."""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(
            wav.astype(np.float32), sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        )
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def _augment_wav(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        if random.random() < 0.3:
            wav = wav + np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start: start + sr // 4] = 0
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode="same").astype(np.float32)
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode="same").astype(np.float32)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def _augment_mel(self, fbank: np.ndarray) -> np.ndarray:
        T, D = fbank.shape
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0: f0 + f] = 0
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0: t0 + t, :] = 0
        return fbank

    def _load_audio_inputs(self, audio_bytes: bytes) -> Tuple[Optional[torch.Tensor], int]:
        if not audio_bytes:
            return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = self._augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(
            wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        valid_len = inputs.attention_mask.sum().item()
        fbank = self._augment_mel(inputs.input_features.squeeze(0).numpy())
        return torch.tensor(fbank).unsqueeze(0), valid_len

    def _load_image_inputs(self, image_bytes: bytes):
        if not image_bytes or self.vision_processor is None:
            return None
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.vision_processor(images=image, return_tensors="pt")
        return {k: v for k, v in inputs.items()} if hasattr(inputs, "keys") else inputs.pixel_values

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def _generate_text_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        ranges = []
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges

    def _create_chat_prompt(self, conversations, audio_features_length: int = 0) -> str:
        conversations = _pre_processing_chat(conversations)
        messages = []
        user_turns = [j for j, t in enumerate(conversations) if t["role"] == "user"]
        is_last_user = lambda i: i == max(user_turns)
        for idx, turn in enumerate(conversations):
            role, content = turn["role"], turn["content"]
            if role == "user" and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4:
                    content = ap
                elif r < 0.6:
                    content = content
                elif r < 0.8:
                    content = ap + "\n\n" + content
                else:
                    content = content + "\n\n" + ap
            if "<image>" in content:
                r = random.random()
                clean = content.replace("<image>", "").strip()
                content = (
                    "<image>\n" + clean if r < 0.2 else
                    "<image>\n\n" + clean if r < 0.4 else
                    clean + "\n<image>" if r < 0.6 else
                    clean + "\n\n<image>"
                )
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return _post_processing_chat(prompt)

    def _apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        if self.scheduled_sampling_prob <= 0:
            return input_ids
        audio_mask = (
            (audio_labels != -100).any(dim=0)
            & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        )
        for i in range(8):
            input_ids[i] = torch.where(
                audio_mask,
                torch.randint(0, self.audio_vocab_size, input_ids[i].shape),
                input_ids[i],
            )
        text_mask = (
            (text_labels != -100)
            & (input_ids[8] != self.image_token_id)
            & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        )
        input_ids[8] = torch.where(
            text_mask,
            torch.randint(0, self.text_vocab_size, input_ids[8].shape),
            input_ids[8],
        )
        return input_ids

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        conversations = json.loads(self.table["conversations"][index].as_py())
        cols = self.table.column_names
        question_audios = self.table["question_audios"][index].as_py() if "question_audios" in cols else []
        answer_audios = self.table["answer_audios"][index].as_py() if "answer_audios" in cols else []
        image_bytes = self.table["image_bytes"][index].as_py() if "image_bytes" in cols else []
        if image_bytes and not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        ref_audios = self.table["ref_audios"][index].as_py() if "ref_audios" in cols else []
        spk_emb_raw = self.table["spk_emb"][index].as_py() if "spk_emb" in cols else []

        asst_idx = [i for i, t in enumerate(conversations) if t["role"] == "assistant"]
        if len(asst_idx) > 1:
            rand_idx = random.randint(0, len(asst_idx) - 1)
            for i in range(rand_idx, -1, -1):
                conversations = conversations[: asst_idx[i] + 1]
                test = self._create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test).input_ids) + 100 < self.max_length:
                    break

        pixel_values = None
        if image_bytes and self.vision_processor:
            pixel_values = self._load_image_inputs(image_bytes[0])

        audio_inputs, audio_len, audio_features_length = None, 0, 0
        user_count = sum(1 for t in conversations if t["role"] == "user")
        if question_audios and user_count <= len(question_audios) and self.audio_processor:
            ab = question_audios[user_count - 1]
            if ab:
                mel, vlen = self._load_audio_inputs(ab)
                if mel is not None:
                    audio_inputs = mel
                    audio_len = vlen
                    audio_features_length = vlen or 1

        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {"pixel_values": torch.zeros(1, 3, 256, 256)}

        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t["role"] == "assistant")
        if answer_audios and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                layers = [[] for _ in range(8)]
                for i in range(0, len(tokens) - 7, 8):
                    for j in range(8):
                        layers[j].append(tokens[i + j])
                for layer in layers:
                    layer.append(self.audio_stop_token)
                last_audio_codes = layers

        prompt = self._create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None:
            prompt = prompt.replace("<image>", self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        text_labels, asst_ranges = self._generate_text_labels(input_ids)
        for start, end in asst_ranges[:-1]:
            me = min(end + len(self.eos_id), self.max_length)
            text_labels[start:me] = [-100] * (me - start)

        Y_audio = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]

        if asst_ranges and last_audio_codes:
            asst_start, _ = asst_ranges[-1]
            for pos in range(asst_start, min(asst_start + 50, self.max_length)):
                if input_ids[pos: pos + len(self.think_end_ids)] == self.think_end_ids:
                    asst_start = pos + len(self.think_end_ids)
                    break

            has_spk = bool(spk_emb_raw)
            has_ref = bool(ref_audios) and random.random() > 0.5
            spk_reserve = 1 if has_spk else 0

            if has_ref:
                ref_layer = [[] for _ in range(8)]
                for i in range(0, len(ref_audios) - 7, 8):
                    for j in range(8):
                        ref_layer[j].append(ref_audios[i + j])
                ref_len = len(ref_layer[0])
                ref_start = max(spk_reserve, asst_start - ref_len)
                for li in range(8):
                    codes = ref_layer[li][-(asst_start - ref_start):]
                    for i, c in enumerate(codes):
                        Y_audio[li][ref_start + i] = c
            else:
                ref_start = asst_start

            if has_spk and ref_start > 0:
                for li in range(8):
                    Y_audio[li][ref_start - 1] = self.audio_spk_token

            for li in range(8):
                for i, c in enumerate(last_audio_codes[li]):
                    pos = asst_start + li + 1 + i
                    if pos < self.max_length:
                        Y_audio[li][pos] = c
                        audio_labels[li][pos] = c

        X_audio = torch.tensor([layer[:-1] for layer in Y_audio], dtype=torch.long)  # (8, T-1)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)                       # (T-1,)
        input_ids_9ch = torch.cat([X_audio, X_text.unsqueeze(0)], dim=0)              # (9, T-1)
        text_labels_t = torch.tensor(text_labels[1:], dtype=torch.long)               # (T-1,)
        audio_labels_t = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)

        input_ids_9ch = self._apply_scheduled_sampling(input_ids_9ch, audio_labels_t, text_labels_t)
        spk_emb_t = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)

        return input_ids_9ch, text_labels_t, audio_labels_t, audio_inputs, audio_len, pixel_values, spk_emb_t
