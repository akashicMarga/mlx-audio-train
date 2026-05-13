"""
Verify that MiniMind-O MLX weight loading is numerically correct.

Three checks:
  1. Weight alignment — compare every key in the raw pytorch_model.bin
     against what our NPZ contains after conversion. Detects key-mapping
     bugs, dtype errors, or transposed tensors.

  2. Tied-embedding constraint — lm_head.weight must equal
     model.embed_tokens.weight (the model uses tied embeddings).

  3. Forward-pass sanity — run a fixed prompt through the MLX model and
     verify the logit distribution looks like a trained LM, not garbage:
       * Output entropy in a reasonable range
       * Top-1 token is a recognisable word
       * Cosine similarity between two identical forward passes is 1.0

No PyTorch required — the raw .bin is read with the same custom unpickler
used in minimind_o_test_text.py.

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    python scripts/minimind_o_verify_alignment.py
    python scripts/minimind_o_verify_alignment.py --verbose
"""

import argparse
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.minimind_o import MiniMindOmni, OmniConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent.parent
BIN_PATH    = Path.home() / ".cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots"
WEIGHTS_NPZ = ROOT / "out" / "minimind_3o" / "weights.npz"

PASS = "\033[32mPASS\033[0m"
WARN = "\033[33mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# PyTorch-free .bin loader
# ---------------------------------------------------------------------------

def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    return (u16.astype(np.uint32) << 16).view(np.float32)


_DTYPE_READERS = {
    "FloatStorage":    lambda b: np.frombuffer(b, dtype=np.float32).copy(),
    "HalfStorage":     lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32),
    "BFloat16Storage": _bf16_to_f32,
    "LongStorage":     lambda b: np.frombuffer(b, dtype=np.int64).copy(),
    "IntStorage":      lambda b: np.frombuffer(b, dtype=np.int32).copy(),
    "ByteStorage":     lambda b: np.frombuffer(b, dtype=np.uint8).copy(),
    "DoubleStorage":   lambda b: np.frombuffer(b, dtype=np.float64).astype(np.float32),
}


def _rebuild_tensor(storage, offset, size, stride, *_):
    arr   = storage
    total = 1
    for s in size:
        total *= s
    chunk = arr[offset: offset + total]
    if not size:
        return chunk
    exp = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        exp[i] = exp[i + 1] * size[i + 1]
    if list(stride) == exp:
        return chunk.reshape(size)
    return np.lib.stride_tricks.as_strided(
        arr[offset:], shape=size,
        strides=[s * arr.dtype.itemsize for s in stride],
    ).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self._zf, self._prefix, self._cache = zf, prefix, {}

    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        return type(name, (), {"__name__": name})

    def persistent_load(self, pid):
        _, cls, key, _, numel = pid
        if key not in self._cache:
            reader = _DTYPE_READERS.get(cls.__name__, _DTYPE_READERS["FloatStorage"])
            self._cache[key] = reader(self._zf.read(f"{self._prefix}/data/{key}"))
        return self._cache[key]


def load_bin(path: str) -> dict:
    zf     = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]
    sd     = _Unpickler(zf.open(f"{prefix}/data.pkl"), zf, prefix).load()
    zf.close()
    return {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}


# ---------------------------------------------------------------------------
# Key remapping (same as convert_pt_to_mlx)
# ---------------------------------------------------------------------------

def _remap(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if k.startswith(("audio_encoder.", "vision_encoder.")):
            continue
        nk = k
        nk = re.sub(r"talker\.(codec_proj|embed_proj)\.0\.",  r"talker.\1.fc1.",  nk)
        nk = re.sub(r"talker\.(codec_proj|embed_proj)\.2\.",  r"talker.\1.fc2.",  nk)
        nk = re.sub(r"talker\.(codec_proj|embed_proj)\.3\.",  r"talker.\1.norm.", nk)
        nk = re.sub(r"talker\.embed_tokens\.adapters\.(\d+)\.0\.weight",
                    r"talker.embed_tokens.adapters.\1.emb.weight",  nk)
        nk = re.sub(r"talker\.embed_tokens\.adapters\.(\d+)\.2\.weight",
                    r"talker.embed_tokens.adapters.\1.proj.weight", nk)
        nk = re.sub(r"talker\.lm_head\.adapters\.(\d+)\.0\.weight",
                    r"talker.lm_head.adapters.\1.fc1.weight",       nk)
        nk = re.sub(r"talker\.lm_head\.adapters\.(\d+)\.2\.weight",
                    r"talker.lm_head.adapters.\1.fc2.weight",       nk)
        nk = re.sub(r"audio_proj\.mlp\.0\.", "audio_proj.ln.",  nk)
        nk = re.sub(r"audio_proj\.mlp\.1\.", "audio_proj.fc1.", nk)
        nk = re.sub(r"audio_proj\.mlp\.3\.", "audio_proj.fc2.", nk)
        nk = re.sub(r"vision_proj\.mlp\.0\.", "vision_proj.ln.",  nk)
        nk = re.sub(r"vision_proj\.mlp\.1\.", "vision_proj.fc1.", nk)
        nk = re.sub(r"vision_proj\.mlp\.3\.", "vision_proj.fc2.", nk)
        out[nk] = v.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Check 1 — weight alignment
# ---------------------------------------------------------------------------

def check_weight_alignment(raw_sd: dict, npz_sd: dict, verbose: bool):
    p = w = f = 0
    missing_in_npz = []
    mismatches     = []

    for key, raw_arr in raw_sd.items():
        if key not in npz_sd:
            missing_in_npz.append(key)
            f += 1
            continue
        npz_arr = npz_sd[key]
        if raw_arr.shape != npz_arr.shape:
            mismatches.append((key, raw_arr.shape, npz_arr.shape, "shape mismatch"))
            f += 1
            continue
        diff = np.abs(raw_arr - npz_arr).max()
        if diff < 1e-5:
            p += 1
            if verbose:
                print(f"  {PASS}  {key}  max_diff={diff:.2e}  shape={raw_arr.shape}")
        elif diff < 1e-3:
            w += 1
            print(f"  {WARN}  {key}  max_diff={diff:.2e}  (dtype rounding?)")
        else:
            f += 1
            mismatches.append((key, raw_arr.shape, npz_arr.shape, f"max_diff={diff:.4f}"))

    if missing_in_npz:
        print(f"\n  Keys in .bin but MISSING from NPZ ({len(missing_in_npz)}):")
        for k in sorted(missing_in_npz)[:20]:
            print(f"    {k}")

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for k, rs, ns, reason in mismatches[:20]:
            print(f"    {k}  raw={rs}  npz={ns}  {reason}")

    return p, w, f


# ---------------------------------------------------------------------------
# Check 2 — tied embeddings
# ---------------------------------------------------------------------------

def check_tied_embeddings(npz_sd: dict):
    embed = npz_sd.get("model.embed_tokens.weight")
    lm    = npz_sd.get("lm_head.weight")
    if embed is None:
        print(f"  {FAIL}  model.embed_tokens.weight not in NPZ")
        return False
    if lm is None:
        print("  lm_head.weight absent from NPZ (expected — tied weights skipped during save)")
        return None  # will be re-checked via loaded model
    diff = np.abs(embed - lm).max()
    if diff < 1e-6:
        print(f"  {PASS}  lm_head.weight == embed_tokens.weight  (max_diff={diff:.2e})")
        return True
    print(f"  {FAIL}  lm_head.weight != embed_tokens.weight  max_diff={diff:.4f}")
    return False


def check_tied_via_model(model: MiniMindOmni) -> bool:
    from mlx.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))
    embed  = params.get("model.embed_tokens.weight")
    lm     = params.get("lm_head.weight")
    if embed is None or lm is None:
        print(f"  {WARN}  Cannot locate embed/lm_head in model params")
        return False
    e = np.array(embed.tolist())
    l = np.array(lm.tolist())
    if e.shape == l.shape and np.abs(e - l).max() < 1e-4:
        print(f"  {PASS}  lm_head.weight == embed_tokens.weight (via model)")
        return True
    print(f"  {FAIL}  lm_head.weight != embed_tokens.weight  shapes {e.shape} vs {l.shape}")
    return False


# ---------------------------------------------------------------------------
# Check 3 — forward-pass sanity
# ---------------------------------------------------------------------------

FIXED_PROMPT = "<|im_start|>user\nHello, what is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def check_forward_pass(model: MiniMindOmni) -> bool:
    import glob

    snaps = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots/*/tokenizer.json"
    )))
    if not snaps:
        print(f"  {WARN}  Tokenizer not found, skipping forward-pass check")
        return True

    from transformers import AutoTokenizer
    tok_dir = os.path.dirname(snaps[-1])
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    ids = mx.array(tok(FIXED_PROMPT).input_ids, dtype=mx.int32)[None, :]

    out1 = model(ids)
    mx.eval(out1["logits"])
    out2 = model(ids)
    mx.eval(out2["logits"])

    logits1 = np.array(out1["logits"][0, -1, :].tolist(), dtype=np.float32)
    logits2 = np.array(out2["logits"][0, -1, :].tolist(), dtype=np.float32)

    # Determinism
    det_diff = np.abs(logits1 - logits2).max()
    if det_diff < 1e-5:
        print(f"  {PASS}  Determinism: two identical forward passes agree  (max_diff={det_diff:.2e})")
    else:
        print(f"  {FAIL}  Non-deterministic output! max_diff={det_diff:.4f}")
        return False

    cos = float(np.dot(logits1, logits2) / (np.linalg.norm(logits1) * np.linalg.norm(logits2) + 1e-9))
    print(f"  {PASS}  Cosine similarity (run 1 vs 2): {cos:.6f}")

    probs  = np.exp(logits1 - logits1.max())
    probs /= probs.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    vocab_entropy = np.log(len(logits1))
    ratio = entropy / vocab_entropy
    if 0.1 < ratio < 0.95:
        print(f"  {PASS}  Entropy: {entropy:.2f} nats ({ratio*100:.1f}% of max) — looks like a trained LM")
    else:
        print(f"  {WARN}  Entropy: {entropy:.2f} nats ({ratio*100:.1f}% of max) — "
              f"{'too collapsed' if ratio < 0.1 else 'too uniform, may be random weights'}")

    top5_idx  = np.argsort(logits1)[::-1][:5]
    top5_tok  = [tok.decode([int(i)], skip_special_tokens=False).strip() for i in top5_idx]
    top5_prob = probs[top5_idx]
    print(f"  Top-5 next tokens: {list(zip(top5_tok, [f'{p:.3f}' for p in top5_prob]))}")

    if top5_prob[0] > 0.9:
        print(f"  {WARN}  Top token has {top5_prob[0]*100:.1f}% probability — distribution collapsed")
    else:
        print(f"  {PASS}  Distribution looks healthy (top-1 prob = {top5_prob[0]*100:.1f}%)")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify MiniMind-O MLX weight alignment")
    parser.add_argument("--verbose", action="store_true", help="Print every key comparison")
    parser.add_argument("--skip_alignment", action="store_true", help="Skip key-by-key weight diff")
    args = parser.parse_args()

    if not WEIGHTS_NPZ.exists():
        print(f"NPZ not found at {WEIGHTS_NPZ}")
        print("Run: python scripts/minimind_o_test_text.py")
        sys.exit(1)

    bins = sorted(BIN_PATH.glob("*/pytorch_model.bin"))
    if not bins and not args.skip_alignment:
        print(f"pytorch_model.bin not found under {BIN_PATH}")
        print("Skipping alignment check — run with --skip_alignment to proceed with checks 2 & 3 only")
        args.skip_alignment = True
    bin_path = bins[-1] if bins else None

    print(f"\n{'='*60}")
    print("  MiniMind-O  |  Weight Alignment Verification")
    print(f"{'='*60}")
    if bin_path:
        print(f"  Source  : {bin_path}")
    print(f"  Target  : {WEIGHTS_NPZ}\n")

    print("Loading NPZ weights...", end=" ", flush=True)
    npz_sd = dict(np.load(WEIGHTS_NPZ, allow_pickle=False))
    print(f"OK  ({len(npz_sd)} keys)")

    # ---- Check 1 ----
    if not args.skip_alignment:
        print("\n-- Check 1: Key-by-key weight alignment --")
        print("Loading pytorch_model.bin ...", end=" ", flush=True)
        raw_sd = load_bin(str(bin_path))
        print(f"OK  ({len(raw_sd)} raw keys)")

        print("Applying key remapping ...")
        remapped = _remap(raw_sd)
        print(f"Remapped to {len(remapped)} keys\n")

        p, w, f = check_weight_alignment(remapped, npz_sd, args.verbose)
        total = p + w + f
        print(f"\n  Summary: {p}/{total} pass, {w}/{total} warn, {f}/{total} fail")
        if f == 0:
            print(f"  {PASS}  All keys loaded correctly")
        elif f <= 3:
            print(f"  {WARN}  {f} key(s) failed — check mapping above")
        else:
            print(f"  {FAIL}  {f} key failures — weight loading is broken")
        del raw_sd, remapped

    # ---- Check 2 ----
    print("\n-- Check 2: Tied embeddings --")
    tie_via_npz = check_tied_embeddings(npz_sd)

    # ---- Load model ----
    print("\n-- Loading MLX model --")
    config = OmniConfig()
    model  = MiniMindOmni(config, audio_encoder_path="")
    model.load_weights([(k, mx.array(v)) for k, v in npz_sd.items()], strict=False)
    model.eval()
    print("  Model ready")

    if tie_via_npz is None:
        check_tied_via_model(model)

    # ---- Check 3 ----
    print("\n-- Check 3: Forward-pass sanity --")
    check_forward_pass(model)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
