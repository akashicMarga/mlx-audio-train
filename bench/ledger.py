"""
Results ledger: one CSV row per benchmark run, rejected attempts included.

COLUMNS must stay identical to the blog's `_data/qwen3tts_mlx_ledger.csv`
(the chart script there reads it by column name). If the target file exists with
a different header, appending refuses rather than writing a misaligned row.
"""

import csv
import os
from pathlib import Path
from typing import Dict, Optional

COLUMNS = [
    "date", "device", "variant", "config_id", "parent_config_id", "lever", "change",
    "model_id", "precision", "batch_size", "mlx_version", "mlx_audio_version",
    "n_prompts", "n_seeds", "audio_s", "rtf", "ttfp_ms", "talker_ms_step",
    "mtp_ms_step", "codec_ms", "snr_db", "peak_mem_gb", "cer_capped", "wilcoxon_p",
    "dnsmos_ovrl", "spk_sim", "decision", "evidence_path", "notes",
]

DECISIONS = {"baseline", "accepted", "rejected", "no-benefit", ""}

DEFAULT_LEDGER = Path(os.environ.get("BENCH_LEDGER", "bench_runs/ledger.csv"))


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def validate_row(row: Dict) -> Dict[str, str]:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown ledger columns: {sorted(unknown)}")
    for key in ("device", "variant", "config_id", "lever"):
        if not row.get(key):
            raise ValueError(f"ledger row needs '{key}'")
    if row.get("decision", "") not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}, got {row['decision']!r}")
    return {c: _fmt(row.get(c)) for c in COLUMNS}


def read_header(path: Path) -> Optional[list]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as f:
        return next(csv.reader(f), None)


def append_row(row: Dict, path: Path = DEFAULT_LEDGER) -> Path:
    path = Path(os.path.expanduser(str(path)))
    out = validate_row(row)
    header = read_header(path)
    if header is not None and header != COLUMNS:
        raise ValueError(
            f"{path} has a different header than bench.ledger.COLUMNS; "
            "update one of them so they match before appending.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if header is None:
            w.writeheader()
        w.writerow(out)
    return path


def config_ids(path: Path = DEFAULT_LEDGER) -> set:
    path = Path(os.path.expanduser(str(path)))
    if read_header(path) is None:
        return set()
    with path.open(newline="") as f:
        return {r["config_id"] for r in csv.DictReader(f)}
