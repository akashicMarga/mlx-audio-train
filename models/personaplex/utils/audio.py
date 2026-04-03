# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mlx.core as mx
import numpy as np


def reshape_input_tokens(encoded: np.ndarray, user_codebooks: int) -> mx.array:
    """Reshape audio tokenizer output to the shape expected by LmGen.step().

    The rustymimi tokenizer outputs shape (B, codebooks, T) or (B, T, codebooks).
    This function normalizes it to (B, user_codebooks, 1) for single-step inference.

    Args:
        encoded: numpy array from audio tokenizer encode_step.
        user_codebooks: number of user (input) codebooks expected by the model.

    Returns:
        MLX array of shape (B, user_codebooks, 1).
    """
    tokens = mx.array(encoded).transpose(0, 2, 1)[:, :, :user_codebooks]
    if tokens.shape[1] == user_codebooks and tokens.shape[2] == 1:
        return tokens
    if tokens.shape[1] == 1 and tokens.shape[2] == user_codebooks:
        return tokens.transpose(0, 2, 1)
    raise ValueError(f"unexpected encoded shape {tokens.shape}")
