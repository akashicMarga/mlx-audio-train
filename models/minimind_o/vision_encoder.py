"""
MLX-native SigLIP2 ViT vision encoder for MiniMind-O.

Loads weights directly from safetensors — no PyTorch required.
Returns per-patch hidden states (batch, num_patches, 768) compatible
with the existing MMVisionProjector.

Architecture (google/siglip2-base-patch16-256):
  - 256×256 image, 16×16 patches → 256 patch tokens
  - 12 transformer layers, hidden=768, heads=12, mlp=3072
  - Pre-LayerNorm, no CLS token
  - Learned 2D position embeddings (256 positions)
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image preprocessor (pure numpy — no torchvision needed)
# ---------------------------------------------------------------------------

class SiglipImageProcessor:
    """Resize + normalize PIL images to SigLIP2 input format."""

    def __init__(self, size: int = 256, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __call__(self, image: Image.Image) -> mx.array:
        """Returns (1, H, W, 3) mx.array, float32, normalised."""
        img = image.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3) in [0,1]
        arr = (arr - self.mean) / self.std                       # normalise
        return mx.array(arr)[None, ...]                          # (1, H, W, 3)


# ---------------------------------------------------------------------------
# ViT building blocks
# ---------------------------------------------------------------------------

class SiglipAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads    = heads
        self.head_dim = hidden // heads
        self.scale    = self.head_dim ** -0.5
        self.q_proj   = nn.Linear(hidden, hidden, bias=True)
        self.k_proj   = nn.Linear(hidden, hidden, bias=True)
        self.v_proj   = nn.Linear(hidden, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn   = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out    = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class SiglipMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SiglipBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn   = SiglipAttention(hidden, heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp         = SiglipMLP(hidden, intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full SigLIP2 ViT encoder (vision_model only, no text head)
# ---------------------------------------------------------------------------

class SiglipVisionEncoder(nn.Module):
    """
    Minimal MLX port of SigLIP2-base-patch16-256 vision encoder.

    Output: SimpleNamespace with .last_hidden_state of shape (B, num_patches, 768)
    Compatible with the MMVisionProjector in MiniMind-O.
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        hidden: int = 768,
        heads: int = 12,
        layers: int = 12,
        intermediate: int = 3072,
    ):
        super().__init__()
        self.patch_size  = patch_size
        num_patches      = (image_size // patch_size) ** 2  # 256

        # Patch embedding via conv (weight: out_ch, in_ch, kH, kW)
        self.patch_embedding    = nn.Conv2d(3, hidden, patch_size, stride=patch_size, bias=True)
        self.position_embedding = nn.Embedding(num_patches, hidden)
        self.encoder_layers     = [SiglipBlock(hidden, heads, intermediate) for _ in range(layers)]
        self.post_layernorm     = nn.LayerNorm(hidden)

        self._pos_ids = mx.arange(num_patches)[None, :]  # (1, num_patches)

    def __call__(self, pixel_values: mx.array) -> SimpleNamespace:
        """
        pixel_values: (B, H, W, 3) — MLX channel-last format
        Returns SimpleNamespace(.last_hidden_state = (B, num_patches, hidden))
        """
        # Patch embed: (B, H, W, 3) → (B, H/P, W/P, hidden) → (B, num_patches, hidden)
        x = self.patch_embedding(pixel_values)
        B, gh, gw, C = x.shape
        x = x.reshape(B, gh * gw, C)

        # Add position embeddings
        x = x + self.position_embedding(self._pos_ids)

        # Transformer blocks
        for block in self.encoder_layers:
            x = block(x)

        x = self.post_layernorm(x)
        return SimpleNamespace(last_hidden_state=x)


# ---------------------------------------------------------------------------
# Weight loader: safetensors → SiglipVisionEncoder MLX params
# ---------------------------------------------------------------------------

def load_siglip2_mlx(model_dir: str) -> tuple[SiglipVisionEncoder, SiglipImageProcessor]:
    """
    Load SigLIP2 weights from safetensors into an MLX SiglipVisionEncoder.
    Auto-detects patch size from the conv weight shape.
    Returns (encoder, preprocessor).
    """
    from safetensors import numpy as sf

    sf_path = Path(model_dir) / "model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

    raw = sf.load_file(str(sf_path))

    # Auto-detect architecture from weight shapes
    conv_w   = raw["vision_model.embeddings.patch_embedding.weight"]  # (out, in, kH, kW)
    pos_w    = raw["vision_model.embeddings.position_embedding.weight"]
    patch_size   = conv_w.shape[2]           # kH (= kW for square patches)
    num_patches  = pos_w.shape[0]            # e.g. 64 for p32-256, 256 for p16-256
    hidden       = conv_w.shape[0]
    image_size   = int(num_patches ** 0.5) * patch_size  # e.g. 8*32 = 256
    num_layers   = sum(1 for k in raw if k.startswith("vision_model.encoder.layers.")
                       and k.endswith(".layer_norm1.weight"))
    mlp_key      = "vision_model.encoder.layers.0.mlp.fc1.weight"
    intermediate = raw[mlp_key].shape[0]
    heads        = 12  # standard for base models

    print(f"  SigLIP2: patch={patch_size}, image={image_size}, "
          f"patches={num_patches}, layers={num_layers}, hidden={hidden}")

    model = SiglipVisionEncoder(
        image_size=image_size,
        patch_size=patch_size,
        hidden=hidden,
        heads=heads,
        layers=num_layers,
        intermediate=intermediate,
    )

    # ---- patch_embedding: Conv2d weights are (out, in, kH, kW) in safetensors,
    #      MLX Conv2d expects (out, kH, kW, in)
    conv_w = raw["vision_model.embeddings.patch_embedding.weight"]  # (768, 3, 16, 16)
    conv_w = np.transpose(conv_w, (0, 2, 3, 1))                     # (768, 16, 16, 3)
    model.patch_embedding.weight = mx.array(conv_w)
    model.patch_embedding.bias   = mx.array(raw["vision_model.embeddings.patch_embedding.bias"])

    model.position_embedding.weight = mx.array(
        raw["vision_model.embeddings.position_embedding.weight"]
    )

    # ---- Transformer blocks
    for i, block in enumerate(model.encoder_layers):
        pfx = f"vision_model.encoder.layers.{i}"
        block.layer_norm1.weight = mx.array(raw[f"{pfx}.layer_norm1.weight"])
        block.layer_norm1.bias   = mx.array(raw[f"{pfx}.layer_norm1.bias"])
        block.layer_norm2.weight = mx.array(raw[f"{pfx}.layer_norm2.weight"])
        block.layer_norm2.bias   = mx.array(raw[f"{pfx}.layer_norm2.bias"])

        attn = block.self_attn
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(attn, proj).weight = mx.array(raw[f"{pfx}.self_attn.{proj}.weight"])
            getattr(attn, proj).bias   = mx.array(raw[f"{pfx}.self_attn.{proj}.bias"])

        block.mlp.fc1.weight = mx.array(raw[f"{pfx}.mlp.fc1.weight"])
        block.mlp.fc1.bias   = mx.array(raw[f"{pfx}.mlp.fc1.bias"])
        block.mlp.fc2.weight = mx.array(raw[f"{pfx}.mlp.fc2.weight"])
        block.mlp.fc2.bias   = mx.array(raw[f"{pfx}.mlp.fc2.bias"])

    model.post_layernorm.weight = mx.array(raw["vision_model.post_layernorm.weight"])
    model.post_layernorm.bias   = mx.array(raw["vision_model.post_layernorm.bias"])

    model.eval()
    processor = SiglipImageProcessor()
    return model, processor
