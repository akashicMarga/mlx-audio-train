#!/usr/bin/env python3
"""
Compare PyTorch vs MLX forward pass for ai4bharat/indic-parler-tts.

Checks two things:
  1. T5 encoder hidden states (description encoding)
  2. Parler decoder logits at step 0 (after prompt prefill)

Run:
    python scripts/compare_pt_mlx.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ID  = "ai4bharat/indic-parler-tts"
DESC     = "A female speaker delivers clear speech at a moderate pace."
TEXT     = "Hello, how are you?"


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def rel_err(a, b):
    return float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PYTORCH REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PYTORCH SIDE")
print("=" * 60)

import torch
import safetensors
from transformers import AutoTokenizer, T5EncoderModel

# -- tokenizers
desc_tok   = AutoTokenizer.from_pretrained("google/flan-t5-large")
prompt_tok = AutoTokenizer.from_pretrained(REPO_ID)

desc_ids_pt   = torch.tensor(desc_tok(DESC, return_tensors="np").input_ids, dtype=torch.long)
prompt_ids_pt = torch.tensor(prompt_tok(TEXT, return_tensors="np").input_ids, dtype=torch.long)
print(f"desc_ids   shape: {desc_ids_pt.shape}")
print(f"prompt_ids shape: {prompt_ids_pt.shape}")

# -- T5 encoder (HF reference)
print("\n[PT] Loading T5 encoder (flan-t5-large) ...")
t5_pt = T5EncoderModel.from_pretrained("google/flan-t5-large")
t5_pt.eval()
with torch.no_grad():
    enc_pt = t5_pt(input_ids=desc_ids_pt).last_hidden_state  # [1, T_desc, 1024]
enc_pt_np = enc_pt.numpy()
print(f"[PT] T5 encoder output: {enc_pt_np.shape}  mean={enc_pt_np.mean():.4f}  std={enc_pt_np.std():.4f}")

# -- load HF safetensors for Parler decoder part
print("\n[PT] Loading HF safetensors for decoder comparison ...")
from huggingface_hub import hf_hub_download
path = hf_hub_download(REPO_ID, "model.safetensors")
st   = safetensors.safe_open(path, framework="pt", device="cpu")

def w(key):
    return st.get_tensor(key)


# -- Minimal PyTorch Parler decoder — just enough for one forward step

HIDDEN   = 1024
NUM_CB   = 9
NUM_HEAD = 16
HEAD_DIM = HIDDEN // NUM_HEAD
BOS_ID   = 1025
EMBED_V  = 1089   # embed_tokens vocab size


def pt_layer_norm(x, weight, bias):
    return torch.layer_norm(x, [HIDDEN], weight, bias)

def pt_attn(x, q_w, k_w, v_w, o_w, mask=None, kv=None):
    """Self-attention. Returns (out, (k, v))."""
    B, T, _ = x.shape
    q = (x @ q_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
    if kv is None:
        k = (x @ k_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
        v = (x @ v_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
    else:
        k_cur = (x @ k_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
        v_cur = (x @ v_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
        k = torch.cat([kv[0], k_cur], dim=2)
        v = torch.cat([kv[1], v_cur], dim=2)
    scale  = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
    out  = (attn @ v).transpose(1, 2).reshape(B, -1, HIDDEN)
    return out @ o_w.T, (k, v)

def pt_cross_attn(x, enc, q_w, k_w, v_w, o_w, enc_kv=None):
    """Cross-attention to enc. Returns (out, (enc_k, enc_v))."""
    B, T, _ = x.shape
    q = (x @ q_w.T).reshape(B, T, NUM_HEAD, HEAD_DIM).transpose(1, 2)
    if enc_kv is None:
        Ts = enc.shape[1]
        ek = (enc @ k_w.T).reshape(B, Ts, NUM_HEAD, HEAD_DIM).transpose(1, 2)
        ev = (enc @ v_w.T).reshape(B, Ts, NUM_HEAD, HEAD_DIM).transpose(1, 2)
    else:
        ek, ev = enc_kv
    scale  = HEAD_DIM ** -0.5
    scores = (q @ ek.transpose(-2, -1)) * scale
    attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
    out  = (attn @ ev).transpose(1, 2).reshape(B, T, HIDDEN)
    return out @ o_w.T, (ek, ev)

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))

def pt_ffn(x, fc1_w, fc2_w):
    return gelu(x @ fc1_w.T) @ fc2_w.T

def pt_decoder_layer(x, enc, idx, mask=None, self_kv=None, cross_kv=None):
    pfx = f"decoder.model.decoder.layers.{idx}."
    # pre-norm self-attn
    h_ln  = pt_layer_norm(x, w(pfx+"self_attn_layer_norm.weight"), w(pfx+"self_attn_layer_norm.bias"))
    h, new_self_kv = pt_attn(
        h_ln,
        w(pfx+"self_attn.q_proj.weight"), w(pfx+"self_attn.k_proj.weight"),
        w(pfx+"self_attn.v_proj.weight"), w(pfx+"self_attn.out_proj.weight"),
        mask=mask, kv=self_kv,
    )
    x = x + h
    # pre-norm cross-attn
    h_ln  = pt_layer_norm(x, w(pfx+"encoder_attn_layer_norm.weight"), w(pfx+"encoder_attn_layer_norm.bias"))
    h, new_cross_kv = pt_cross_attn(
        h_ln, enc,
        w(pfx+"encoder_attn.q_proj.weight"), w(pfx+"encoder_attn.k_proj.weight"),
        w(pfx+"encoder_attn.v_proj.weight"), w(pfx+"encoder_attn.out_proj.weight"),
        enc_kv=cross_kv,
    )
    x = x + h
    # pre-norm FFN
    h_ln = pt_layer_norm(x, w(pfx+"final_layer_norm.weight"), w(pfx+"final_layer_norm.bias"))
    x = x + pt_ffn(h_ln, w(pfx+"fc1.weight"), w(pfx+"fc2.weight"))
    return x, new_self_kv, new_cross_kv

print("[PT] Running decoder prefill ...")
with torch.no_grad():
    enc = torch.tensor(enc_pt_np)  # [1, T_desc, 1024]

    # embed_prompts + embed_positions for prompt tokens
    ep_w   = w("embed_prompts.weight")                        # [90714, 1024]
    pos_w  = w("decoder.model.decoder.embed_positions.weights") # [4096, 1024]
    et_w   = [w(f"decoder.model.decoder.embed_tokens.{k}.weight") for k in range(NUM_CB)]

    T_p = prompt_ids_pt.shape[1]
    prompt_emb = ep_w[prompt_ids_pt[0]]                       # [T_p, 1024]

    # BOS audio embedding (all codebooks BOS)
    bos_emb = sum(et_w[k][BOS_ID] for k in range(NUM_CB))    # [1024]
    bos_emb = bos_emb.unsqueeze(0)                            # [1, 1024]

    first_seq = torch.cat([prompt_emb, bos_emb], dim=0)       # [T_p+1, 1024]
    first_seq = first_seq + pos_w[:T_p+1]                     # add positions
    x = first_seq.unsqueeze(0)                                 # [1, T_p+1, 1024]

    T = x.shape[1]
    causal_mask = torch.triu(torch.full((T, T), -1e9), diagonal=1)[None, None]

    self_kvs  = [None] * 24
    cross_kvs = [None] * 24
    for i in range(24):
        x, self_kvs[i], cross_kvs[i] = pt_decoder_layer(
            x, enc, i, mask=causal_mask, self_kv=None, cross_kv=None
        )

    # final layer norm
    x = pt_layer_norm(x, w("decoder.model.decoder.layer_norm.weight"),
                         w("decoder.model.decoder.layer_norm.bias"))

    # logits at last position
    lm_w    = w("decoder.lm_heads.weight")  # [9*1088, 1024]
    logits_pt = (x[:, -1:, :] @ lm_w.T).reshape(1, 1, NUM_CB, 1088)
    logits_pt_np = logits_pt[0, 0].numpy()  # [9, 1088]
    print(f"[PT] Decoder logits step 0: {logits_pt_np.shape}  max={logits_pt_np.max():.3f}")
    greedy_pt = np.argmax(logits_pt_np, axis=-1)
    print(f"[PT] Greedy tokens step 0: {greedy_pt}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MLX REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MLX SIDE")
print("=" * 60)

import mlx.core as mx
from models.indic_parler_tts import load_model

print("[MLX] Loading model ...")
model, tokenizers = load_model(REPO_ID, strict_load=True, audit_load=True)

desc_ids_mlx   = mx.array(desc_tok(DESC, return_tensors="np").input_ids, dtype=mx.int32)
prompt_ids_mlx = mx.array(prompt_tok(TEXT, return_tensors="np").input_ids, dtype=mx.int32)

# T5 encoder
enc_mlx = model.text_encoder(desc_ids_mlx)
mx.eval(enc_mlx)
enc_mlx_np = np.array(enc_mlx)
print(f"[MLX] T5 encoder output: {enc_mlx_np.shape}  mean={enc_mlx_np.mean():.4f}  std={enc_mlx_np.std():.4f}")

# Decoder step 0
NUM_LAYERS = 24
T_p_mlx = prompt_ids_mlx.shape[1]

prompt_emb_mlx = model.embed_prompts(prompt_ids_mlx)  # [1, T_p, 1024]
bos_tokens_mlx = mx.full((1, NUM_CB), BOS_ID, dtype=mx.int32)
bos_emb_mlx    = model.decoder.embed_audio_no_pos(bos_tokens_mlx)  # [1, 1, 1024]
first_emb_mlx  = mx.concatenate([prompt_emb_mlx, bos_emb_mlx], axis=1)
first_emb_mlx  = first_emb_mlx + model.decoder.decoder_position(0, first_emb_mlx.shape[1])

from models.indic_parler_tts.model import _causal_mask
mask_mlx = _causal_mask(first_emb_mlx.shape[1], dtype=first_emb_mlx.dtype)

self_caches  = [[] for _ in range(NUM_LAYERS)]
cross_caches = [[] for _ in range(NUM_LAYERS)]
hidden_mlx = model.decoder.forward_layers(
    first_emb_mlx, enc_mlx,
    mask=mask_mlx,
    self_caches=self_caches, cross_caches=cross_caches,
)
logits_mlx = model.decoder.logits(hidden_mlx[:, -1:, :])
mx.eval(logits_mlx)
logits_mlx_np = np.array(logits_mlx[0, 0])  # [9, 1088]
greedy_mlx = np.argmax(logits_mlx_np, axis=-1)
print(f"[MLX] Decoder logits step 0: {logits_mlx_np.shape}  max={logits_mlx_np.max():.3f}")
print(f"[MLX] Greedy tokens step 0: {greedy_mlx}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\n--- T5 Encoder ---")
print(f"  cosine similarity : {cosine_sim(enc_pt_np, enc_mlx_np):.6f}  (1.0 = identical)")
print(f"  relative error    : {rel_err(enc_mlx_np, enc_pt_np):.6f}  (0.0 = identical)")
print(f"  max abs diff      : {np.abs(enc_pt_np - enc_mlx_np).max():.6f}")

print("\n--- Decoder logits step 0 (per codebook) ---")
print(f"  {'CB':>4}  {'cos_sim':>10}  {'rel_err':>10}  {'PT greedy':>10}  {'MLX greedy':>10}  {'match':>6}")
for k in range(NUM_CB):
    cs = cosine_sim(logits_pt_np[k], logits_mlx_np[k])
    re = rel_err(logits_mlx_np[k], logits_pt_np[k])
    match = greedy_pt[k] == greedy_mlx[k]
    print(f"  {k:>4}  {cs:>10.6f}  {re:>10.6f}  {greedy_pt[k]:>10}  {greedy_mlx[k]:>10}  {'✓' if match else '✗':>6}")

print("\nDone.")
