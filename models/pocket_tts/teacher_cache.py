"""
Precomputed teacher targets for distillation: run the frozen teacher ONCE over the
data and store its guidance-combined z per sample, so each distillation step only
runs the student. The teacher is ~70% of a 24L->6L step, so this is ~3x faster per
step; the price is disk (z_width x 2 bytes per frame: ~18 GB per 100 h at z 1024)
and fixed prompt cuts (pass --epochs N to store N differently-cut copies).

Layout: <dir>/index.jsonl + <dir>/shards/NNNNNNN.npz with latents, tokens, voice, target.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from models.pocket_tts.data import batches
from models.pocket_tts.distill import teacher_target
from models.pocket_tts.train_utils import normalize


def write_cache(teacher, ds, out_dir: str | Path, cfg_coef: float, batch_size: int = 16,
                epochs: int = 1, log_every: int = 50) -> int:
    out_dir = Path(out_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "index.jsonl", "w") as index:
        for epoch in range(epochs):
            ds.rng.seed(epoch)  # a different prompt cut per copy
            for bi, batch in enumerate(batches(ds, batch_size, 1, shuffle=False)):
                target = teacher_target(teacher, normalize(batch, teacher), cfg_coef)
                mx.eval(target)
                target = np.array(target.astype(mx.float16))
                lat, mask = np.array(batch["latents"]), np.array(batch["mask"])
                voice = np.array(batch["voice"])
                for b in range(lat.shape[0]):
                    T = int(mask[b].sum())
                    name = f"shards/{n:07d}.npz"
                    np.savez(out_dir / name, latents=lat[b, :T].astype(np.float16),
                             tokens=np.asarray(batch["text_tokens"][b], dtype=np.int32),
                             voice=voice[b, : int(batch["voice_lens"][b])].astype(np.float16),
                             target=target[b, : T + 1])  # + the EOS frame
                    index.write(json.dumps({"shard": name, "frames": T}) + "\n")
                    n += 1
                if bi % log_every == 0:
                    print(f"  [cache] epoch {epoch} batch {bi}: {n} rows", flush=True)
    meta = {"rows": n, "cfg_coef": cfg_coef, "z_width": teacher.args.z_width,
            "teacher_layers": teacher.args.num_layers}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return n


class CachedDataset:
    """Rows of a teacher cache; same interface as LatentDataset (sample() adds the target)."""

    def __init__(self, cache_dir: str | Path, expect_cfg_coef: float | None = None):
        self.root = Path(cache_dir)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if expect_cfg_coef is not None and abs(self.meta["cfg_coef"] - expect_cfg_coef) > 1e-6:
            raise ValueError(f"cache was written with cfg_coef {self.meta['cfg_coef']}, config says {expect_cfg_coef}")
        with open(self.root / "index.jsonl") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, i: int):
        d = np.load(self.root / self.entries[i]["shard"])
        return (d["latents"].astype(np.float32), d["tokens"], d["voice"].astype(np.float32),
                d["target"].astype(np.float32))
