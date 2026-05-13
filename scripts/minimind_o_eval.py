"""
MiniMind-O inference script — MLX port of minimind-o/eval_omni.py.

Upstream model: https://github.com/jingyaogong/minimind-o

Usage:
    # Text → speech
    python scripts/minimind_o_eval.py --model_dir out/ --mode 0

    # Audio input → speech
    python scripts/minimind_o_eval.py --model_dir out/ --mode 2 --audio_dir dataset/eval_omni/

    # Voice cloning
    python scripts/minimind_o_eval.py --model_dir out/ --mode 3

    # Load from HuggingFace checkpoint directory
    python scripts/minimind_o_eval.py --model_dir /path/to/hf_checkpoint --mode 0
"""

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from models.minimind_o import MiniMindOmni, OmniConfig


# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------

def init_model(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_dir, trust_remote_code=True
    )

    config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindOmni(
        config,
        audio_encoder_path=args.audio_encoder_dir,
        vision_model_path=args.vision_dir or None,
    )
    model.load_weights_from_pt(args.checkpoint, strict=False)

    try:
        from transformers import MimiModel
        model.mimi_model = MimiModel.from_pretrained(args.mimi_dir).eval()
        print(f"Mimi decoder loaded from {args.mimi_dir}")
    except Exception as e:
        print(f"[warn] Mimi decoder not loaded: {e}. Audio will be codes only.")
        model.mimi_model = None

    trainable = sum(
        p.size for _, p in model.parameters().items()
        if "audio_encoder" not in _ and "vision_encoder" not in _
    )
    print(f"Model params: {trainable / 1e6:.2f}M")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Decode Mimi codes to audio
# ---------------------------------------------------------------------------

def decode_audio(model, audio_frames, output_path: str, device: str = "cpu") -> bool:
    if not audio_frames or model.mimi_model is None:
        return False
    try:
        import torch
        import soundfile as sf
        from pydub import AudioSegment

        codes = [f for f in audio_frames if f and len(f) == 8]
        if not codes:
            return False

        mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)  # (1, 8, frames)
        filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
        with torch.no_grad():
            audio = model.mimi_model.decode(filtered).audio_values

        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
        sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)
        if output_path.endswith(".mp3"):
            AudioSegment.from_wav(wav_path).export(output_path, format="mp3", bitrate="64k")
            os.remove(wav_path)
        print(f"  → {output_path}")
        return True
    except Exception as e:
        print(f"  [warn] Audio decode failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Single sample evaluation
# ---------------------------------------------------------------------------

def eval_sample(
    model, tokenizer, args, prompt: str, output_name: str,
    audio_inputs=None, audio_lens=None, pixel_values=None,
    history=None, ref_codes=None, spk_emb=None,
):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        open_thinking=bool(args.open_thinking),
    )
    input_ids = mx.array(
        tokenizer(text).data["input_ids"], dtype=mx.int32
    )[None, :]  # (1, T)

    audio_frames = []
    print("  [Thinker]: ", end="", flush=True)
    history_idx = 0

    kwargs = {}
    if audio_inputs is not None:
        kwargs["audio_inputs"] = audio_inputs
        kwargs["audio_lens"] = audio_lens
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values
    if ref_codes is not None:
        kwargs["ref_codes"] = ref_codes
    if spk_emb is not None:
        kwargs["spk_emb"] = spk_emb
    if args.open_thinking:
        kwargs["open_thinking"] = True

    for y, audio_frame in model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=True,
        return_audio_codes=True,
        **kwargs,
    ):
        if y is not None:
            answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
            if answer and not answer.endswith("?"):
                print(answer[history_idx:], end="", flush=True)
                history_idx = len(answer)
        if audio_frame:
            audio_frames.append(audio_frame)

    print()

    if audio_frames and args.decode_audio:
        out_path = os.path.join(args.output_dir, output_name)
        decode_audio(model, audio_frames, out_path)
    elif audio_frames:
        print(f"  [Talker]: {len(audio_frames)} frames (decode_audio=off)")


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def run_text_mode(model, tokenizer, args):
    print("\n====== text → {text, audio} ======")
    prompts = [
        "Tell me an interesting fact about space.",
        "How do I make a cup of coffee?",
        "Tell me a joke.",
        "Can you sing a song for me?",
        "Please introduce yourself.",
    ]
    for idx, prompt in enumerate(prompts):
        print(f"\n[text-{idx+1}]: {prompt}")
        eval_sample(model, tokenizer, args, prompt, f"text-{idx:02d}.mp3")


def run_audio_mode(model, tokenizer, args):
    print("\n====== audio → {text, audio} ======")
    from models.minimind_o.dataset import OmniDataset
    import torch

    audio_files = sorted([
        f for f in os.listdir(args.audio_dir)
        if f.lower().endswith((".mp3", ".wav"))
    ])
    for idx, af in enumerate(audio_files):
        print(f"\n[audio-{idx+1}]: {af}")
        mel, valid_len = OmniDataset.process_audio(
            os.path.join(args.audio_dir, af), model.audio_processor
        )
        audio_inputs = mel.unsqueeze(0)
        audio_lens = torch.tensor([valid_len])
        prompt = model.config.audio_special_token * (valid_len or 1)
        eval_sample(
            model, tokenizer, args, prompt, f"audio-{idx:02d}-{os.path.splitext(af)[0]}.mp3",
            audio_inputs=audio_inputs, audio_lens=audio_lens,
        )


def run_clone_mode(model, tokenizer, args):
    print("\n====== voice clone → {text, audio} ======")
    import torch

    prompts = [
        "Hello, please introduce yourself.",
        "What's the weather like today?",
        "Tell me a joke.",
    ]
    voices_pt = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "speaker", "voices_unseen.pt",
    )
    voices = [("default", None, None)]
    if os.path.exists(voices_pt):
        voice_data = torch.load(voices_pt, map_location="cpu")
        for speaker, v in sorted(voice_data.items()):
            rc = v["ref_codes"].unsqueeze(0)
            se = v["spk_emb"].half().unsqueeze(0) if "spk_emb" in v else None
            rc_mx = mx.array(rc.numpy())
            se_mx = mx.array(se.float().numpy()) if se is not None else None
            voices.append((speaker, rc_mx, se_mx))

    history = [{"role": "system", "content": "You are a professional voice assistant. Answer in the given voice style."}]
    for speaker, rc, se in voices:
        print(f"\n[clone: {speaker}]")
        for idx, prompt in enumerate(prompts):
            print(f"  [{idx+1}]: {prompt}")
            eval_sample(
                model, tokenizer, args, prompt,
                f"clone-{speaker}-{idx:02d}.mp3",
                ref_codes=rc, spk_emb=se, history=history,
            )


def run_multi_mode(model, tokenizer, args):
    print("\n====== multi-turn → {text, audio} ======")
    tests = [
        {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
            ],
            "prompt": "I want to find something to do. Any suggestions?",
        },
        {
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "I want to find something to do. Any suggestions?"},
                {"role": "assistant", "content": "You can listen to music or read a book."},
            ],
            "prompt": "Thanks, I'll try that.",
        },
    ]
    for idx, test in enumerate(tests):
        print(f"\n[multi-{idx+1}]")
        eval_sample(
            model, tokenizer, args, test["prompt"],
            f"multi-{idx:02d}.mp3", history=test["history"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MiniMind-O inference (MLX)")
    parser.add_argument("--model_dir", default="out", type=str)
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to .pth checkpoint (auto-detected if None)")
    parser.add_argument("--tokenizer_path", default=None, type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.75, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--output_dir", default="./output_audio", type=str)
    parser.add_argument("--audio_dir", default="./dataset/eval_omni", type=str)
    parser.add_argument("--audio_encoder_dir", default="./model/SenseVoiceSmall", type=str)
    parser.add_argument("--vision_dir", default=None, type=str)
    parser.add_argument("--mimi_dir", default="./model/mimi", type=str)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument("--decode_audio", default=1, type=int)
    parser.add_argument("--mode", default="0", type=str, help="0=text 1=multi 2=audio 3=clone or comma-separated")
    args = parser.parse_args()

    if args.checkpoint is None:
        import glob
        ptfiles = glob.glob(os.path.join(args.model_dir, "*.pth"))
        if not ptfiles:
            raise FileNotFoundError(f"No .pth file in {args.model_dir}")
        args.checkpoint = sorted(ptfiles)[-1]
        print(f"Auto-detected checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(int(time.time()) % 31415926)

    model, tokenizer = init_model(args)
    modes = set(args.mode.replace(",", "").replace("-1", "0123"))

    if "0" in modes:
        run_text_mode(model, tokenizer, args)
    if "1" in modes:
        run_multi_mode(model, tokenizer, args)
    if "2" in modes:
        run_audio_mode(model, tokenizer, args)
    if "3" in modes:
        run_clone_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
