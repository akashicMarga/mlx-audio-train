"""
Text-only smoke test for MiniMind-O (MLX).

Downloads weights from HF on first run and caches them locally.
Skips SenseVoice / Mimi — pure text in, text out.

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    python scripts/minimind_o_test_text.py
    python scripts/minimind_o_test_text.py --prompt "What is the capital of France?"
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from models.minimind_o import MiniMindOmni, OmniConfig

HF_REPO = "jingyaogong/minimind-3o"
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out", "minimind_3o",
)
WEIGHTS_PATH = os.path.join(CACHE_DIR, "weights.npz")


# ---------------------------------------------------------------------------
# PyTorch-free .bin loader (reads PyTorch ZIP format with custom unpickler)
# ---------------------------------------------------------------------------

def _bf16_to_f32(raw: bytes) -> "np.ndarray":
    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    f32_bits = u16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


_DTYPE_READERS = {
    "FloatStorage":    lambda b: np.frombuffer(b, dtype=np.float32).copy(),
    "HalfStorage":     lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32),
    "BFloat16Storage": _bf16_to_f32,
    "LongStorage":     lambda b: np.frombuffer(b, dtype=np.int64).copy(),
    "IntStorage":      lambda b: np.frombuffer(b, dtype=np.int32).copy(),
    "ByteStorage":     lambda b: np.frombuffer(b, dtype=np.uint8).copy(),
    "DoubleStorage":   lambda b: np.frombuffer(b, dtype=np.float64).astype(np.float32),
}


def _rebuild_tensor(storage, offset, size, stride, *_args):
    arr = storage
    total = 1
    for s in size:
        total *= s
    chunk = arr[offset: offset + total]
    if len(size) == 0:
        return chunk
    exp = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        exp[i] = exp[i + 1] * size[i + 1]
    if list(stride) == exp:
        return chunk.reshape(size)
    return np.lib.stride_tricks.as_strided(
        arr[offset:],
        shape=size,
        strides=[s * arr.dtype.itemsize for s in stride],
    ).copy()


import pickle as _pickle


class _TorchUnpickler(_pickle.Unpickler):
    def __init__(self, fileobj, zip_file, prefix):
        super().__init__(fileobj)
        self._zf = zip_file
        self._prefix = prefix
        self._storage: dict = {}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        return type(name, (), {"__name__": name})

    def persistent_load(self, pid):
        typename, storage_cls, key, _loc, numel = pid
        if key not in self._storage:
            reader = _DTYPE_READERS.get(storage_cls.__name__, _DTYPE_READERS["FloatStorage"])
            raw = self._zf.read(f"{self._prefix}/data/{key}")
            self._storage[key] = reader(raw)
        return self._storage[key]


def load_pytorch_bin_as_numpy(bin_path: str) -> dict:
    """Load a pytorch_model.bin (ZIP format) without PyTorch."""
    import zipfile

    zf = zipfile.ZipFile(bin_path)
    names = zf.namelist()
    prefix = names[0].split("/")[0]
    pkl_data = zf.open(f"{prefix}/data.pkl")
    state_dict = _TorchUnpickler(pkl_data, zf, prefix).load()
    zf.close()
    return state_dict


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------

def download_and_convert() -> str:
    """Download from HF, convert keys to MLX format, return tokenizer dir."""
    from huggingface_hub import snapshot_download

    os.makedirs(CACHE_DIR, exist_ok=True)

    hf_cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--jingyaogong--minimind-3o/snapshots/*/tokenizer.json"
    )
    cached = sorted(glob.glob(hf_cache_pattern))

    if os.path.exists(WEIGHTS_PATH):
        print(f"Weights already cached: {WEIGHTS_PATH}")
        tok_dir = os.path.dirname(cached[-1]) if cached else CACHE_DIR
        return tok_dir

    print(f"Downloading {HF_REPO} ...")
    local_dir = snapshot_download(HF_REPO)
    print(f"Downloaded to: {local_dir}")

    bin_files = sorted(glob.glob(os.path.join(local_dir, "*.bin")))
    safetensor_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    pth_files = sorted(glob.glob(os.path.join(local_dir, "*.pth")))

    if bin_files:
        print(f"Loading {bin_files[0]} via custom unpickler (no PyTorch needed) ...")
        state_dict = load_pytorch_bin_as_numpy(bin_files[0])
    elif safetensor_files:
        try:
            from safetensors import numpy as sf_np
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(sf_np.load_file(f))
            print(f"Loaded from safetensors ({len(safetensor_files)} shards)")
        except ImportError:
            raise RuntimeError("safetensors package not installed. Run: pip install safetensors")
    elif pth_files:
        import torch
        state_dict = torch.load(pth_files[0], map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights found in {local_dir}")

    print(f"Converting {len(state_dict)} keys ...")
    import re
    mlx_weights = {}
    for k, v in state_dict.items():
        if not isinstance(v, np.ndarray):
            continue
        new_k = k
        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.0\.", r"talker.\1.fc1.", new_k)
        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.2\.", r"talker.\1.fc2.", new_k)
        new_k = re.sub(r"talker\.(codec_proj|embed_proj)\.3\.", r"talker.\1.norm.", new_k)
        new_k = re.sub(r"talker\.embed_tokens\.adapters\.(\d+)\.0\.weight",
                       r"talker.embed_tokens.adapters.\1.emb.weight", new_k)
        new_k = re.sub(r"talker\.embed_tokens\.adapters\.(\d+)\.2\.weight",
                       r"talker.embed_tokens.adapters.\1.proj.weight", new_k)
        new_k = re.sub(r"talker\.lm_head\.adapters\.(\d+)\.0\.weight",
                       r"talker.lm_head.adapters.\1.fc1.weight", new_k)
        new_k = re.sub(r"talker\.lm_head\.adapters\.(\d+)\.2\.weight",
                       r"talker.lm_head.adapters.\1.fc2.weight", new_k)
        new_k = re.sub(r"audio_proj\.mlp\.0\.", "audio_proj.ln.", new_k)
        new_k = re.sub(r"audio_proj\.mlp\.1\.", "audio_proj.fc1.", new_k)
        new_k = re.sub(r"audio_proj\.mlp\.3\.", "audio_proj.fc2.", new_k)
        new_k = re.sub(r"vision_proj\.mlp\.0\.", "vision_proj.ln.", new_k)
        new_k = re.sub(r"vision_proj\.mlp\.1\.", "vision_proj.fc1.", new_k)
        new_k = re.sub(r"vision_proj\.mlp\.3\.", "vision_proj.fc2.", new_k)
        mlx_weights[new_k] = v.astype(np.float32)

    print(f"Saving {len(mlx_weights)} keys → {WEIGHTS_PATH}")
    np.savez(WEIGHTS_PATH, **mlx_weights)
    print("Conversion complete.")
    return local_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> MiniMindOmni:
    config = OmniConfig()
    model = MiniMindOmni(config, audio_encoder_path="")
    weights = list(np.load(WEIGHTS_PATH, allow_pickle=False).items())
    model.load_weights([(k, mx.array(v)) for k, v in weights], strict=False)
    return model


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(
    model: MiniMindOmni,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.75,
    top_p: float = 0.85,
) -> tuple[str, int, float]:
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        text = f"<s>user\n{prompt}</s><s>assistant\n"

    input_ids = mx.array(tokenizer(text).data["input_ids"], dtype=mx.int32)[None, :]

    response_tokens: list[int] = []
    t0 = time.time()

    for y, _ in model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        return_audio_codes=False,
    ):
        if y is None:
            break
        response_tokens = y[0].tolist()

    elapsed = time.time() - t0
    answer = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return answer, len(response_tokens), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "Please introduce yourself briefly.",
    "What is 7 multiplied by 8?",
    "Tell me a short joke.",
    "What are three tips for writing clean code?",
]


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O text-only test")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.85)
    args = parser.parse_args()

    tokenizer_dir = download_and_convert()

    print(f"\nLoading tokenizer from: {tokenizer_dir}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if not tokenizer.chat_template:
        jinja_path = os.path.join(tokenizer_dir, "chat_template.jinja")
        if os.path.exists(jinja_path):
            with open(jinja_path) as f:
                tokenizer.chat_template = f.read()
    print(f"Vocab size: {len(tokenizer)}")

    print("Loading MLX model ...")
    model = load_model()
    from mlx.utils import tree_flatten
    n_params = sum(
        p.size for name, p in tree_flatten(model.parameters())
        if "audio_encoder" not in name and "vision_encoder" not in name
    )
    print(f"Model ready — {n_params / 1e6:.1f}M params\n")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    sep = "=" * 60

    print(sep)
    print("MiniMind-O  |  text → text  (MLX, Apple Silicon)")
    print(sep)

    for prompt in prompts:
        print(f"\n[User]      {prompt}")
        answer, n_tok, elapsed = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        tok_per_sec = n_tok / elapsed if elapsed > 0 else 0
        print(f"[Assistant] {answer}")
        print(f"            -> {n_tok} tokens  {elapsed:.1f}s  {tok_per_sec:.1f} tok/s")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
