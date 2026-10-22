"""Data loading, BPE tokenizer, binary memmap dataset, and prefetching DataLoader."""

from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class MemmapDataset:
    """Dataset backed by a uint16 numpy memmap file."""

    def __init__(self, bin_path: str | Path, context_len: int = 1024) -> None:
        self.bin_path = Path(bin_path)
        self.context_len = context_len
        self.seq_len = context_len + 1  # 1025 tokens for input + target

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary data file not found: {bin_path}")

        file_size_bytes = os.path.getsize(self.bin_path)
        self.total_tokens = file_size_bytes // 2  # uint16 is 2 bytes
        # Non-overlapping windows starting at multiples of 1024
        self.num_sequences = (self.total_tokens - 1) // context_len
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return self.num_sequences

    def get_sequence(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_len
        chunk = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class PrefetchDataLoader:
    """Background prefetch data loader with pinned memory and non-blocking GPU transfer."""

    def __init__(
        self,
        dataset: MemmapDataset,
        batch_size: int,
        seed: int = 1337,
        device: torch.device | None = None,
        queue_size: int = 4,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.queue_size = queue_size

        # Create seeded permutation of sequence indices
        rng = np.random.default_rng(seed)
        self.indices = rng.permutation(len(dataset))
        self.num_batches = len(self.indices) // batch_size

        self.queue: queue.Queue[tuple[torch.Tensor, torch.Tensor, float] | None] = queue.Queue(
            maxsize=queue_size
        )
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

    def _worker(self) -> None:
        pin_memory = torch.cuda.is_available()
        for b_idx in range(self.num_batches):
            if self.stop_event.is_set():
                break

            batch_slice = self.indices[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            t0 = time.perf_counter()
            for seq_idx in batch_slice:
                x, y = self.dataset.get_sequence(int(seq_idx))
                xs.append(x)
                ys.append(y)

            batch_x = torch.stack(xs)
            batch_y = torch.stack(ys)
            if pin_memory:
                batch_x = batch_x.pin_memory()
                batch_y = batch_y.pin_memory()

            fetch_time = time.perf_counter() - t0
            self.queue.put((batch_x, batch_y, fetch_time))

        self.queue.put(None)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        while True:
            t_wait_start = time.perf_counter()
            item = self.queue.get()
            wait_time_ms = (time.perf_counter() - t_wait_start) * 1000.0

            if item is None:
                break

            batch_x, batch_y, _ = item
            non_blocking = self.device.type == "cuda"
            batch_x = batch_x.to(self.device, non_blocking=non_blocking)
            batch_y = batch_y.to(self.device, non_blocking=non_blocking)

            yield batch_x, batch_y, wait_time_ms

    def close(self) -> None:
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
