"""Background GPU utilization and memory monitor via nvidia-smi."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


class GPUMonitor:
    """Samples nvidia-smi every interval_ms in a background thread."""

    def __init__(self, output_csv_path: str | Path, interval_ms: int = 500) -> None:
        self.output_csv_path = Path(output_csv_path)
        self.interval_ms = interval_ms
        self.interval_sec = interval_ms / 1000.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.records: list[tuple[float, float, float]] = []  # (timestamp, util_gpu, mem_used_mb)

    def _monitor_loop(self) -> None:
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "utilization_gpu_pct", "memory_used_mb"])

            while not self.stop_event.is_set():
                t_now = time.time()
                util = 0.0
                mem_mb = 0.0
                try:
                    res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=1.0,
                    )
                    if res.returncode == 0:
                        line = res.stdout.strip().split("\n")[0]
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            util = float(parts[0])
                            mem_mb = float(parts[1])
                except Exception:
                    pass

                self.records.append((t_now, util, mem_mb))
                writer.writerow([f"{t_now:.3f}", util, mem_mb])
                f.flush()

                time.sleep(self.interval_sec)

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop monitor and return (mean_utilization_pct, peak_memory_gb)."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if not self.records:
            return 0.0, 0.0

        utils = [r[1] for r in self.records]
        mems = [r[2] for r in self.records]

        mean_util = float(sum(utils) / len(utils))
        peak_mem_gb = float(max(mems) / 1024.0)
        return mean_util, peak_mem_gb
