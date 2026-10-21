"""
Text normalisation applied identically when fitting a tokenizer, during training and
at inference. Pocket TTS reads raw text through a SentencePiece table (no phonemes),
so anything not written the way it is spoken — digits above all — must be spelled out.

    normalizer("hindi")("कल 25 लोग आए, कुल ₹1,50,000 खर्च हुए।")
    -> "कल पच्चीस लोग आए, कुल एक लाख पचास हज़ार रुपये खर्च हुए।"
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

# 0-99 are irregular in Hindi.
_HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानबे बानबे तिरानबे चौरानबे पचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे"
).split()
assert len(_HI_0_99) == 100

# Indian grouping: crore (10^7), lakh (10^5), thousand, hundred.
_HI_SCALES = [(10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")]
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_CURRENCY = {"₹": "रुपये", "$": "डॉलर", "%": "प्रतिशत"}


def hindi_number(n: int) -> str:
    if n < 100:
        return _HI_0_99[n]
    words = []
    for value, name in _HI_SCALES:
        if n >= value:
            q, n = divmod(n, value)
            words += [hindi_number(q), name]
    if n:
        words.append(_HI_0_99[n])
    return " ".join(words)


def _hindi_digits(m: re.Match) -> str:
    s = m.group(0).replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{hindi_number(int(whole or 0))} दशमलव " + " ".join(_HI_0_99[int(d)] for d in frac)
    if len(s) > 1 and s.startswith("0"):  # phone numbers, codes: digit by digit
        return " ".join(_HI_0_99[int(d)] for d in s)
    return hindi_number(int(s))


def normalize_hindi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_DEVANAGARI_DIGITS)
    # ₹500 -> 500 रुपये ; 20% -> 20 प्रतिशत
    text = re.sub(r"([₹$])\s*([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(2)} {_CURRENCY[m.group(1)]}", text)
    text = re.sub(r"(\d)\s*%", lambda m: f"{m.group(1)} {_CURRENCY['%']}", text)
    text = re.sub(r"\d[\d,]*(?:\.\d+)?", _hindi_digits, text)
    return re.sub(r"[ \t]+", " ", text).strip()


NORMALIZERS: dict[str, Callable[[str], str]] = {"hindi": normalize_hindi}


def normalizer(name: str | None) -> Callable[[str], str] | None:
    if not name:
        return None
    if name not in NORMALIZERS:
        raise ValueError(f"unknown normalize_text {name!r}; have {sorted(NORMALIZERS)}")
    return NORMALIZERS[name]
