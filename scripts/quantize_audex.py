#!/usr/bin/env python3
"""
Save a pre-quantized Audex MLX checkpoint (smaller download, loads directly).

Quantizes the chosen components' Linear/Embedding layers and copies the rest
through unchanged. Writes quant.json so models.audex.load_model applies the
same quantization structure before loading the weights.

Default quantizes the LM only (the tested-safe, best quality/size tradeoff):
    python scripts/quantize_audex.py --src checkpoints/audex_mlx --out checkpoints/audex_mlx_4bit --bits 4

Add --decoder / --audio to quantize those too (smaller, but audio quality
becomes precision-sensitive — validate before publishing).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx.utils import tree_flatten


def _save_quantized(module, out_path, bits, group_size):
    from models.audex.generate import _quantize
    _quantize(module, bits, group_size)
    mx.eval(module.parameters())
    weights = dict(tree_flatten(module.parameters()))
    mx.save_safetensors(str(out_path), weights)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="bf16 checkpoint dir")
    ap.add_argument("--out", required=True, help="output quantized checkpoint dir")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--decoder", action="store_true", help="also quantize the speech decoder")
    ap.add_argument("--audio", action="store_true", help="also quantize the audio encoder")
    args = ap.parse_args()

    from models.audex.lm import LMConfig, NemotronDenseForCausalLM
    from models.audex.speech_decoder import SpeechDecoderConfig, AudexSpeechDecoder
    from models.audex.audio_encoder import AudioEncoderConfig, AudioTower

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # copy configs + tokenizer + everything not being quantized
    passthrough = ["lm_config.json", "speech_decoder_config.json", "audio_config.json",
                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for fn in passthrough:
        if (src / fn).exists():
            shutil.copy(src / fn, out / fn)

    targets = ["lm"]

    lm = NemotronDenseForCausalLM(LMConfig(**json.loads((src / "lm_config.json").read_text())))
    lm.load_weights(str(src / "lm.safetensors"))
    print(f"[quantize] LM -> {args.bits}-bit ...", flush=True)
    _save_quantized(lm, out / "lm.safetensors", args.bits, args.group_size)

    if args.decoder:
        targets.append("decoder")
        dec = AudexSpeechDecoder(SpeechDecoderConfig(**json.loads((src / "speech_decoder_config.json").read_text())))
        dec.load_weights(str(src / "speech_decoder.safetensors"))
        print(f"[quantize] speech decoder -> {args.bits}-bit ...", flush=True)
        _save_quantized(dec, out / "speech_decoder.safetensors", args.bits, args.group_size)
    else:
        shutil.copy(src / "speech_decoder.safetensors", out / "speech_decoder.safetensors")

    if (src / "audio.safetensors").exists():
        if args.audio:
            targets.append("audio")
            tower = AudioTower(AudioEncoderConfig(**json.loads((src / "audio_config.json").read_text())))
            tower.load_weights(str(src / "audio.safetensors"))
            print(f"[quantize] audio encoder -> {args.bits}-bit ...", flush=True)
            _save_quantized(tower, out / "audio.safetensors", args.bits, args.group_size)
        else:
            shutil.copy(src / "audio.safetensors", out / "audio.safetensors")

    (out / "quant.json").write_text(json.dumps(
        {"bits": args.bits, "group_size": args.group_size, "targets": targets}, indent=2))
    total = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"[quantize] done -> {out}  ({total:.2f} GB safetensors, targets={targets})")


if __name__ == "__main__":
    main()
