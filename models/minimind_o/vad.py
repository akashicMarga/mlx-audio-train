"""
Voice Activity Detection + Realtime session management.

Direct port of minimind-o's VAD classes — pure numpy/ONNX, no changes needed.
SileroVAD runs an ONNX model on 1024-sample chunks.
RealtimeSession handles speech start/end detection and barge-in interruption.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SileroVAD:
    """Silero VAD via ONNX Runtime. Returns speech probability per 1024-sample chunk."""

    def __init__(self, path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self) -> None:
        self.h[:] = 0
        self.c[:] = 0

    def __call__(self, chunk: np.ndarray, sr: int = 16000) -> float:
        out, self.h, self.c = self.session.run(
            None,
            {
                "input": chunk.reshape(1, -1).astype(np.float32),
                "h": self.h,
                "c": self.c,
                "sr": np.array(sr, dtype="int64"),
            },
        )
        return float(out[0][0])


class RealtimeSession:
    """
    Manages a real-time audio session with VAD-based turn detection.

    Push audio chunks via push_chunk(); it returns:
      'listening'    — no speech detected
      'speech_end'   — user finished speaking, call get_audio()
      'interrupt'    — user started speaking while model was generating (barge-in)
    """

    def __init__(
        self,
        vad_path: str,
        sr: int = 16000,
        threshold: float = 0.8,
        min_speech_ms: int = 128,
        min_silence_ms: int = 800,
    ):
        self.vad = SileroVAD(vad_path)
        self.sr = sr
        self.threshold = threshold
        self.min_speech = int(sr * min_speech_ms / 1000)
        self.min_silence = int(sr * min_silence_ms / 1000)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.buffer: List[np.ndarray] = []
        self.ring: List[np.ndarray] = []
        self.speaking = False
        self.generating = False
        self.interrupt = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.tail_silence = 0

    def push_chunk(self, chunk: np.ndarray, W: int = 1024) -> str:
        for i in range(0, max(len(chunk), 1), W):
            w = chunk[i: i + W]
            if len(w) < W:
                w = np.pad(w, (0, W - len(w)))

            prob = self.vad(w, self.sr)

            if prob > self.threshold:
                self.silence_samples = self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.generating and self.speaking:
                    self.interrupt = True
                    return "interrupt"

            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1):]
                    self.speaking = False
                    self.speech_samples = 0
                    self.silence_samples = 0
                    self.tail_silence = 0
                    return "speech_end"

            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]

        return "listening"

    def get_audio(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio
