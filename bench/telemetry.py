"""
Hardware telemetry during a timed pass: GPU utilisation and clock, power,
temperatures and swap, sampled with macmon (sudo-free Apple Silicon monitor,
`macmon pipe` emits one JSON object per sample).

    with Telemetry() as tel:          # start early: macmon needs ~2.4 s to emit its first sample
        ...warm-up...
        tel.mark_start(); ...timed work...; tel.mark_end()
    tel.summary()   # only samples that arrived inside the marked window

The binary is $BENCH_MACMON, else ~/Documents/macmon/target/release/macmon, else
`macmon` on PATH. If none works, telemetry is reported as unavailable and the
benchmark carries on — it is evidence, not a dependency.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from statistics import mean, median

CANDIDATES = [os.environ.get("BENCH_MACMON"),
              os.path.expanduser("~/Documents/macmon/target/release/macmon"),
              shutil.which("macmon")]


def _binary():
    for c in CANDIDATES:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


class Telemetry:
    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.samples = []                  # (arrival time, sample dict)
        self.error = None
        self.t_start = None
        self.t_end = None
        self._proc = None
        self._thread = None

    def __enter__(self):
        exe = _binary()
        if exe is None:
            self.error = "macmon not found"
            return self
        try:
            self._proc = subprocess.Popen([exe, "pipe", "-i", str(self.interval_ms)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:  # noqa: BLE001
            self.error = f"macmon failed to start: {e}"
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            try:
                self.samples.append((time.time(), json.loads(line)))
            except json.JSONDecodeError:
                continue

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if not self.samples:
                err = (self._proc.stderr.read() or "").strip().splitlines()
                self.error = "macmon produced no samples" + (f": {err[-1]}" if err else "")
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False

    def mark_start(self):
        self.t_start = time.time()

    def mark_end(self):
        self.t_end = time.time()

    def summary(self) -> dict:
        lo = self.t_start or float("-inf")
        hi = self.t_end or float("inf")
        s = [x for t, x in self.samples if lo <= t <= hi]
        if not s:
            reason = self.error or ("no samples in the timed window" if self.samples else "no samples")
            return {"available": False, "reason": reason}

        def col(f):
            out = []
            for x in s:
                try:
                    out.append(float(f(x)))
                except (KeyError, TypeError, IndexError, ValueError):
                    pass
            return out

        util = col(lambda x: x["gpu_usage"][1])
        freq = col(lambda x: x["gpu_usage"][0])
        gpu_w = col(lambda x: x["gpu_power"])
        sys_w = col(lambda x: x["sys_power"])
        gpu_t = col(lambda x: x["temp"]["gpu_temp_avg"])
        cpu_t = col(lambda x: x["temp"]["cpu_temp_avg"])
        swap = col(lambda x: x["memory"]["swap_usage"])

        def stat(v, f):
            return round(f(v), 3) if v else None

        return {
            "available": True, "samples": len(s), "interval_ms": self.interval_ms,
            "gpu_util_mean": stat(util, mean), "gpu_util_median": stat(util, median),
            "gpu_freq_mhz_mean": stat(freq, mean), "gpu_freq_mhz_max": stat(freq, max),
            "gpu_power_w_mean": stat(gpu_w, mean), "sys_power_w_mean": stat(sys_w, mean),
            "gpu_temp_c_start": round(gpu_t[0], 1) if gpu_t else None,
            "gpu_temp_c_end": round(gpu_t[-1], 1) if gpu_t else None,
            "gpu_temp_c_max": stat(gpu_t, max), "cpu_temp_c_max": stat(cpu_t, max),
            "swap_gb_delta": round((swap[-1] - swap[0]) / 1e9, 3) if len(swap) > 1 else None,
            "swap_gb_end": round(swap[-1] / 1e9, 2) if swap else None,
        }


def note(t: dict) -> str:
    """One-line ledger note."""
    if not t.get("available"):
        return f"telemetry unavailable ({t.get('reason')})"
    return (f"GPU util {100 * t['gpu_util_mean']:.0f}% @ {t['gpu_freq_mhz_mean']:.0f} MHz "
            f"(max {t['gpu_freq_mhz_max']:.0f}), GPU {t['gpu_power_w_mean']:.1f} W, "
            f"sys {t['sys_power_w_mean']:.1f} W, GPU {t['gpu_temp_c_start']}→{t['gpu_temp_c_end']} °C "
            f"(max {t['gpu_temp_c_max']}), swap Δ{t['swap_gb_delta']} GB")
