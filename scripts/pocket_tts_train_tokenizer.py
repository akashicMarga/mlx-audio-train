#!/usr/bin/env python3
"""
Fit a SentencePiece tokenizer for a new Pocket TTS language (e.g. Hindi).

Same spec as the released tokenizers (pocket-tts training/scripts/train_tokenizer.py):
unigram, 4000 pieces (= lookup_table.n_bins), pieces <= 6 chars, digits split,
byte fallback, pad id 3, and NO normalisation, so text reaches the model as written.
Apply the same --normalize-text here, in training (data.normalize_text) and at inference.

Input: JSONL with "text" (this repo) or "transcript" (official manifests), or plain text.

Usage:
    python scripts/pocket_tts_train_tokenizer.py --out data/hindi_tokenizer \\
        --normalize-text hindi data/hindi/train.jsonl
    # -> data/hindi_tokenizer.model; point flow_lm.lookup_table.tokenizer_path at it
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pocket_tts.text import normalizer  # noqa: E402


def iter_texts(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    line = row.get("text") or row.get("transcript") or ""
                if line:
                    yield line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True, help="output prefix: writes <out>.model and <out>.vocab")
    p.add_argument("--vocab-size", type=int, default=4000, help="must equal lookup_table.n_bins")
    p.add_argument("--character-coverage", type=float, default=0.9995,
                   help="Devanagari is a small alphabet: 0.9995 keeps all regular characters; "
                        "1.0 also keeps every rare symbol")
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe", "char"])
    p.add_argument("--normalize-text", default=None, help="e.g. hindi (see models/pocket_tts/text.py)")
    args = p.parse_args()

    import sentencepiece as spm

    norm = normalizer(args.normalize_text) or (lambda s: s)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        n = 0
        for text in iter_texts(args.inputs):
            tmp.write(norm(text).replace("\n", " ") + "\n")
            n += 1
    print(f"[tokenizer] training on {n} utterances")
    spm.SentencePieceTrainer.train(
        input=tmp.name, model_prefix=args.out, vocab_size=args.vocab_size,
        character_coverage=args.character_coverage, model_type=args.model_type,
        normalization_rule_name="identity", remove_extra_whitespaces=False,
        max_sentencepiece_length=6, allow_whitespace_only_pieces=True, split_digits=True,
        byte_fallback=True, pad_id=3, input_sentence_size=10_000_000, shuffle_input_sentence=True,
    )
    Path(tmp.name).unlink()
    sp = spm.SentencePieceProcessor(model_file=args.out + ".model")
    print(f"[tokenizer] wrote {args.out}.model (vocab {sp.get_piece_size()})")


if __name__ == "__main__":
    main()
