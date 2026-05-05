"""
Pure MLX DAC (Descript Audio Codec) decoder — inference only.

Architecture:
- Quantizer: 9 codebooks, codebook_dim=8, hidden=1024
  - For each token index: lookup codebook → 8-dim → out_proj → 1024-dim
  - Sum across 9 codebooks → latent [T, 1024]
- Decoder: conv → 4× (snake → conv_transpose → 3× res_unit) → snake → conv → tanh
  - Upsampling ratios: 8, 8, 4, 2 (total 512x = hop_length)
  - Output: float32 audio at 44100 Hz

Snake activation (anti-alias):
  snake(x, α) = x + sin²(α·x) / α

Weight loading note: DAC weights use PyTorch channels-first (C, T).
MLX Conv1d expects (B, T, C) inputs and weight (out, kernel, in).
All conv weights need .transpose(0, 2, 1) from PyTorch format.
ConvTranspose1d weights need .transpose(1, 2, 0) from PyTorch format.
Alpha tensors need .reshape(1, 1, C) from PyTorch (1, C, 1).
"""

import mlx.core as mx
import mlx.nn as nn

from .config import DACConfig


def snake(x: mx.array, alpha: mx.array) -> mx.array:
    # alpha: [1, 1, C] for MLX channels-last (B, T, C)
    return x + (1.0 / (alpha + 1e-9)) * mx.sin(alpha * x) ** 2


class DACQuantizer(nn.Module):
    def __init__(self, cfg: DACConfig):
        super().__init__()
        self.n_codebooks = cfg.n_codebooks
        self.codebooks = [
            mx.zeros((cfg.codebook_size, cfg.codebook_dim))
            for _ in range(cfg.n_codebooks)
        ]
        # Treat 1x1 convs as Linear layers (kernel=1 squeezed)
        self.in_proj_w = [
            mx.zeros((cfg.codebook_dim, cfg.hidden_size))
            for _ in range(cfg.n_codebooks)
        ]
        self.in_proj_b = [
            mx.zeros((cfg.codebook_dim,))
            for _ in range(cfg.n_codebooks)
        ]
        self.out_proj_w = [
            mx.zeros((cfg.hidden_size, cfg.codebook_dim))
            for _ in range(cfg.n_codebooks)
        ]
        self.out_proj_b = [
            mx.zeros((cfg.hidden_size,))
            for _ in range(cfg.n_codebooks)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        """
        codes: [n_codebooks, T] int32
        Returns: [T, hidden_size] float32
        """
        T = codes.shape[1]
        out = None
        for k in range(self.n_codebooks):
            z = self.codebooks[k][codes[k]]      # [T, codebook_dim]
            z = z @ self.out_proj_w[k].T + self.out_proj_b[k]  # [T, hidden]
            out = z if out is None else out + z
        return out  # [T, hidden_size]


class DACResUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # snake activations stored as [1, 1, C] after weight loading
        self.alpha1 = mx.ones((1, 1, channels))
        self.alpha2 = mx.ones((1, 1, channels))
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=7,
            padding=3 * dilation, dilation=dilation, bias=True
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = snake(x, self.alpha1)
        h = self.conv1(h)
        h = snake(h, self.alpha2)
        h = self.conv2(h)
        return x + h


class DACDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.snake = mx.ones((1, 1, in_ch))  # alpha, loaded as [1, 1, C]
        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=stride * 2, stride=stride, padding=stride // 2, bias=True
        )
        dilations = [1, 3, 9]
        self.res_units = [DACResUnit(out_ch, d) for d in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        x = snake(x, self.snake)
        x = self.upsample(x)
        for unit in self.res_units:
            x = unit(x)
        return x


class DACDecoder(nn.Module):
    def __init__(self, cfg: DACConfig):
        super().__init__()
        self.quantizer = DACQuantizer(cfg)

        ratios = cfg.upsampling_ratios          # [8, 8, 4, 2]
        ch = cfg.decoder_hidden_size            # 1536

        self.conv1 = nn.Conv1d(cfg.hidden_size, ch, kernel_size=7, padding=3, bias=True)

        blocks = []
        ch_in = ch
        for r in ratios:
            ch_out = ch_in // 2
            blocks.append(DACDecoderBlock(ch_in, ch_out, stride=r))
            ch_in = ch_out
        self.blocks = blocks                    # ch_in ends at 96

        self.end_snake = mx.ones((1, 1, ch_in))  # alpha after all blocks
        self.conv2 = nn.Conv1d(ch_in, 1, kernel_size=7, padding=3, bias=True)

    def decode(self, codes: mx.array) -> mx.array:
        """
        codes: [n_codebooks, T] int32
        Returns: [T_audio] float32 waveform at 44100 Hz
        """
        z = self.quantizer.decode(codes)        # [T, hidden]
        x = z[None]                             # [1, T, hidden]

        x = self.conv1(x)                       # [1, T, 1536]
        for block in self.blocks:
            x = block(x)
        x = snake(x, self.end_snake)
        x = self.conv2(x)                       # [1, T*512, 1]
        x = mx.tanh(x)
        return x[0, :, 0]                       # [T_audio]
