#!/usr/bin/env python3
"""
add_lang_code.py — Stamp a lang_code field onto every sample in a JSONL file.

Run once per language dataset before training. Samples that already have a
lang_code field are left unchanged (use --overwrite to force-update them).

Supported language codes:
    hi  Hindi       ta  Tamil       te  Telugu      bn  Bengali
    kn  Kannada     ml  Malayalam   mr  Marathi     gu  Gujarati
    pa  Punjabi
    (plus any code from the 10 pretrained languages: en, zh, de, es, fr, etc.)

Usage:
    # Stamp all Hindi samples (in-place):
    python scripts/add_lang_code.py --input data/hindi/train_codes.jsonl --lang hi

    # Write to a new file:
    python scripts/add_lang_code.py --input data/hindi/train_codes.jsonl --lang hi \\
        --output data/hindi/train_codes_hi.jsonl

    # Force-overwrite existing lang_code fields:
    python scripts/add_lang_code.py --input data/hindi/train_codes.jsonl --lang hi --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_CODES = {
    # Indian languages (new tokens)
    "hi", "ta", "te", "bn", "kn", "ml", "mr", "gu", "pa",
    # Pretrained languages
    "en", "zh", "de", "es", "fr", "it", "pt", "ja", "ko", "ru",
    # Fallback
    "auto",
}


def main():
    parser = argparse.ArgumentParser(description="Stamp lang_code onto a JSONL dataset")
    parser.add_argument("--input",     required=True, help="Input JSONL file")
    parser.add_argument("--lang",      required=True, help="Language code to stamp (e.g. hi, ta, te)")
    parser.add_argument("--output",    default=None,  help="Output JSONL file (default: overwrite input)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite lang_code even if already present")
    args = parser.parse_args()

    if args.lang not in SUPPORTED_CODES:
        print(f"Warning: '{args.lang}' is not in the known code list {sorted(SUPPORTED_CODES)}.")
        print("Proceeding anyway — make sure you add it to INDIAN_LANG_IDS in scripts/train.py.")

    src = Path(args.input)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        sys.exit(1)

    dst = Path(args.output) if args.output else src
    tmp = dst.with_suffix(".tmp")

    updated = skipped = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "lang_code" not in obj or args.overwrite:
                obj["lang_code"] = args.lang
                updated += 1
            else:
                skipped += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp.replace(dst)
    print(f"Done. {updated} samples stamped with lang_code='{args.lang}', {skipped} already had a code.")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
