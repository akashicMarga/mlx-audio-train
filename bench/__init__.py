"""
TTS inference benchmark harness (Part 4 of the speech series).

Baseline = stock mlx-audio `generate()` (Blaizzy/mlx-audio). Every optimisation
is a registered lever in `bench/levers.py`, applied on top of that baseline, one
change per run. Each run appends one row to a results ledger whose columns match
the blog's `_data/qwen3tts_mlx_ledger.csv` exactly (see `bench/ledger.py`).

Entry points:
    scripts/bench_tts.py     one configuration, one fresh process, one ledger row
    scripts/bench_sweep.py   run a YAML list of configurations, each in its own process
"""
