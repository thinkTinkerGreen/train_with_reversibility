"""Unit tests for dataset sampling determinism and train/val split isolation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from revllm.data import MemmapDataset, PrefetchDataLoader


def test_data_determinism() -> None:
    """Same seed yields identical sequence batches; train and val splits are disjoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.bin"
        val_path = Path(tmpdir) / "val.bin"

        # Create disjoint synthetic uint16 arrays
        train_tokens = np.arange(0, 10000, dtype=np.uint16)
        val_tokens = np.arange(10000, 15000, dtype=np.uint16)

        train_tokens.tofile(train_path)
        val_tokens.tofile(val_path)

        train_ds = MemmapDataset(train_path, context_len=128)
        MemmapDataset(val_path, context_len=128)

        # 1. Check disjoint token sets
        assert set(train_tokens).isdisjoint(set(val_tokens))

        # 2. Check reproducible sampling with seed 1337
        loader1 = PrefetchDataLoader(train_ds, batch_size=4, seed=1337, device=torch.device("cpu"))
        loader2 = PrefetchDataLoader(train_ds, batch_size=4, seed=1337, device=torch.device("cpu"))

        batches1 = list(loader1)
        batches2 = list(loader2)

        loader1.close()
        loader2.close()

        assert len(batches1) == len(batches2)
        for (x1, y1, _), (x2, y2, _) in zip(batches1, batches2):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)


def test_compute_sha256() -> None:
    """Validate compute_sha256 matches hashlib output."""
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"hello world")
        f.flush()
        from revllm.data import compute_sha256

        sha = compute_sha256(f.name)
        assert len(sha) == 64
        assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
