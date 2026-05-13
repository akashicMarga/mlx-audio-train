"""
Convert minimind-o PyTorch checkpoint → MLX-compatible .pth or npz.

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    # From local out/ directory
    python scripts/minimind_o_convert_weights.py --input out/sft_omni_768.pth --output out/sft_omni_768_mlx.pth

    # From HuggingFace (downloads and converts)
    python scripts/minimind_o_convert_weights.py --hf_repo jingyaogong/minimind-3o --output out/minimind_3o_mlx.pth
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_from_file(input_path: str, output_path: str) -> None:
    import torch
    from models.minimind_o.model import convert_pt_to_mlx

    print(f"Loading {input_path} ...")
    state_dict = torch.load(input_path, map_location="cpu")

    print("Converting keys ...")
    mlx_weights = convert_pt_to_mlx(state_dict)

    print(f"Keys: {len(state_dict)} → {len(mlx_weights)}")

    if output_path.endswith(".npz"):
        np.savez(output_path, **mlx_weights)
        print(f"Saved as npz: {output_path}")
    else:
        import torch
        torch.save({k: torch.tensor(v) for k, v in mlx_weights.items()}, output_path)
        print(f"Saved as pth: {output_path}")


def convert_from_hf(repo_id: str, output_path: str, revision: str = "main") -> None:
    from huggingface_hub import snapshot_download
    import torch
    from models.minimind_o.model import convert_pt_to_mlx

    print(f"Downloading {repo_id} ...")
    local_dir = snapshot_download(repo_id, revision=revision)
    print(f"Downloaded to {local_dir}")

    import glob
    pth_files = glob.glob(os.path.join(local_dir, "*.pth"))
    safetensor_files = glob.glob(os.path.join(local_dir, "*.safetensors"))

    if safetensor_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(f))
        print(f"Loaded from safetensors: {safetensor_files}")
    elif pth_files:
        state_dict = torch.load(pth_files[0], map_location="cpu")
        print(f"Loaded from pth: {pth_files[0]}")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(local_dir, trust_remote_code=True)
        state_dict = model.state_dict()
        print("Loaded from HF model")

    mlx_weights = convert_pt_to_mlx(state_dict)
    print(f"Keys converted: {len(mlx_weights)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".npz"):
        np.savez(output_path, **mlx_weights)
    else:
        torch.save({k: torch.tensor(v) for k, v in mlx_weights.items()}, output_path)
    print(f"Saved: {output_path}")
    print(f"Tokenizer at: {local_dir}")


def verify_weights(checkpoint_path: str) -> None:
    """Quick sanity check: instantiate model and load weights."""
    from models.minimind_o import MiniMindOmni, OmniConfig

    print("Verifying with model instantiation ...")
    model = MiniMindOmni(OmniConfig(), audio_encoder_path="")
    model.load_weights_from_pt(checkpoint_path, strict=False)

    total = sum(p.size for _, p in model.parameters().items())
    print(f"  Loaded. Total MLX params: {total/1e6:.2f}M")

    import mlx.core as mx
    text_ids = mx.ones((1, 4), dtype=mx.int32)
    out = model(text_ids)
    mx.eval(out["logits"])
    print(f"  Forward pass OK. logits shape: {out['logits'].shape}")
    print("  Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Convert minimind-o weights to MLX format")
    parser.add_argument("--input", type=str, default=None, help="Input .pth file")
    parser.add_argument("--hf_repo", type=str, default=None, help="HuggingFace repo ID")
    parser.add_argument("--output", type=str, required=True, help="Output file (.pth or .npz)")
    parser.add_argument("--verify", action="store_true", help="Run a forward-pass sanity check after conversion")
    args = parser.parse_args()

    if args.input:
        convert_from_file(args.input, args.output)
    elif args.hf_repo:
        convert_from_hf(args.hf_repo, args.output)
    else:
        parser.error("Provide either --input or --hf_repo")

    if args.verify:
        verify_weights(args.output)


if __name__ == "__main__":
    main()
