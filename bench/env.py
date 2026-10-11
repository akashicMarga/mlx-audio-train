"""Run environment: device label, framework versions, git state — recorded with every run."""

import platform
import subprocess
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def device_label() -> str:
    """'Apple M5 Pro' -> 'M5 Pro' (the label the blog ledger uses)."""
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        brand = platform.processor() or "unknown"
    return brand.removeprefix("Apple ").strip()


def memory_gb() -> float:
    try:
        b = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, check=True).stdout)
        return b / 2**30
    except Exception:
        return 0.0


def versions() -> dict:
    def v(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    return {"mlx": v("mlx"), "mlx_audio": v("mlx-audio"), "mlx_whisper": v("mlx-whisper"),
            "python": platform.python_version()}


def git_state() -> dict:
    def g(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"sha": g("rev-parse", "--short", "HEAD"),
            "dirty": bool(g("status", "--porcelain")),
            "branch": g("branch", "--show-current")}


def busy_processes(threshold: float = 40.0) -> list:
    """Other processes using ≥ threshold % CPU right now (e.g. macOS storage scans),
    which make timings noisy. Excludes this process."""
    import os
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True,
                             text=True, check=True).stdout
    except Exception:
        return []
    busy = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[0]) != os.getpid():
            pcpu = float(parts[1])
            if pcpu >= threshold:
                busy.append({"pid": int(parts[0]), "pcpu": pcpu, "comm": parts[2].rsplit("/", 1)[-1]})
    return sorted(busy, key=lambda b: -b["pcpu"])


def wait_until_idle(threshold: float = 40.0, timeout_s: float = 600.0, poll_s: float = 10.0) -> list:
    """Block until no other process is above threshold (or timeout). Returns what was still busy."""
    import time
    deadline = time.time() + timeout_s
    busy = busy_processes(threshold)
    while busy and time.time() < deadline:
        print(f"[bench] waiting for idle: {', '.join(f'{b['comm']} {b['pcpu']:.0f}%' for b in busy[:3])}")
        time.sleep(poll_s)
        busy = busy_processes(threshold)
    return busy


def snapshot() -> dict:
    return {"device": device_label(), "memory_gb": round(memory_gb(), 1),
            **versions(), "git": git_state()}
