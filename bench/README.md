# TTS inference benchmark harness

Measures Qwen3-TTS inference on Apple Silicon, one change at a time, for Part 4
of the speech series. **Baseline = stock [mlx-audio](https://github.com/Blaizzy/mlx-audio)
`model.generate()`**; every optimisation is a lever applied on top of it.
Each run appends one row to a ledger whose columns match the blog's
`_data/qwen3tts_mlx_ledger.csv`, so the post's charts redraw from it directly.

## Quick start

```bash
export BENCH_LEDGER=~/Documents/akashicMarga.github.io/_data/qwen3tts_mlx_ledger.csv

# validate everything, load nothing
python scripts/bench_sweep.py bench/sweeps/m5pro_baseline.yaml --dry-run

# baseline rows (each configuration in a fresh process; resumes if interrupted)
python scripts/bench_sweep.py bench/sweeps/m5pro_baseline.yaml

# redraw the post's charts
python ~/Documents/akashicMarga.github.io/assets/charts/qwen3tts_mlx_figures.py
```

`python scripts/bench_tts.py --list-levers` prints the lever ladder and which
levers are implemented.

## What one run measures

| Pass | Mode | Gives |
|---|---|---|
| warm-up | one generation, discarded | — |
| speed | uninstrumented `generate()`, prompts × seeds, ≥ `--min-audio-s` audio | `rtf` (wall ÷ audio), `peak_mem_gb`, the audio scored for quality |
| ttfp | `generate(stream=True)` on the first `--ttfp-prompts` | `ttfp_ms` (median time to first chunk) |
| instrument | eval-bracketed timers on Talker / code predictor / codec decode | `talker_ms_step`, `mtp_ms_step` (all 15 passes per frame), `codec_ms` (ms per audio-second), "other" per frame |
| quality | whisper CER capped at 1.0, DNSMOS, paired Wilcoxon + bit-exactness vs `--parent` | `cer_capped`, `dnsmos_ovrl`, `wilcoxon_p`, proposed `decision` |

Notes:

- **Instrumented numbers are only for the breakdown.** Forcing evaluation at
  component boundaries adds syncs, so RTF and TTFP always come from the
  uninstrumented passes.
- **Talker ms/step is the forward only.** Sampling lands in "other".
- **`model.eval()` is always called.** MLX modules default to training mode, which
  leaves LoRA dropout active at inference; the ledger notes record this.
- **Bit-exactness** compares a sha1 of each item's float32 waveform with the
  parent's (same seed and prompt). Waveforms themselves are kept only with
  `--save-wavs`.

## Decisions (the quality gate)

A run is compared with `--parent`, the configuration it changes one thing from.

- **accepted** — faster by at least `--min-speedup` (default 1.03×), and either
  bit-identical or CER not significantly worse (Wilcoxon, `--gate-alpha`).
- **rejected** — CER significantly worse, or DNSMOS / speaker similarity dropped
  more than `--max-dnsmos-drop` (default 0.05) / `--max-spk-drop` (off unless set).
- **no-benefit** — passes the gate but isn't measurably faster.
- **baseline** — no parent.

The decision is a proposal: `--decision` overrides it, and the reason is
written to the row's notes either way.

## Adding a lever

Implement it in `bench/levers.py` with `@register(name, post_section, description,
expect_bit_exact)`. The function receives `(model, gen_kwargs)` and mutates them
in place. Each lever must be one change, composable with the others. Replace
the matching `plan(...)` line.

Precision, streaming chunk size and batch size are configuration, not levers:
use `--model-id`, `--stream-interval` and the batch sweep, and set `--lever` /
`--change` so the ledger records them.

## Files

```
bench/ledger.py        ledger schema (must match the blog CSV) + append
bench/env.py           device label, versions, git state
bench/instrument.py    component timers (class-level patch, restored on exit)
bench/quality.py       scoring, Wilcoxon, bit-exactness, gate proposal
bench/levers.py        lever registry (planned ladder from the post)
bench/sweeps/          sweep definitions
bench/prompts/         prompt sets (hi_smoke.txt is a 5-sentence pipeline check)
scripts/bench_tts.py   one configuration → one ledger row
scripts/bench_sweep.py run a sweep, one fresh process per configuration
tests/test_bench.py    tests that need no model download
bench_runs/<config_id> evidence: summary.json, items.jsonl, env.json (gitignored)
```
