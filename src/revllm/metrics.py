"""Metrics logging, JSONL buffer, and final.json generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class StepMetric:
    step: int
    train_loss: float
    lr: float
    tokens_per_sec: float
    step_time_ms: float
    data_wait_ms: float
    grad_norm: float
    scaler_scale: float
    tokens_seen: int


@dataclass
class EvalMetric:
    step: int
    val_loss: float
    val_ppl: float
    val_loss_sampled_a: float | None = None
    tokens_seen: int = 0


class MetricsTracker:
    """Tracks training and evaluation metrics, writing to metrics.jsonl."""

    def __init__(self, log_path: str | Path, flush_every: int = 10) -> None:
        self.log_path = Path(log_path)
        self.flush_every = flush_every
        self.step_metrics: list[StepMetric] = []
        self.eval_metrics: list[EvalMetric] = []
        self.step_throughputs: list[float] = []

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Touch or clear file
        with open(self.log_path, "a"):
            pass

    def log_step(self, metric: StepMetric) -> None:
        self.step_metrics.append(metric)
        self.step_throughputs.append(metric.tokens_per_sec)

        if len(self.step_metrics) % self.flush_every == 0:
            self.flush()

    def log_eval(self, metric: EvalMetric) -> None:
        self.eval_metrics.append(metric)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"type": "eval", **asdict(metric)}) + "\n")

    def flush(self) -> None:
        if not self.step_metrics:
            return
        with open(self.log_path, "a") as f:
            for m in self.step_metrics[-self.flush_every :]:
                f.write(json.dumps({"type": "step", **asdict(m)}) + "\n")

    def get_steady_state_throughput(self) -> float:
        """Calculate median throughput excluding initial warmup steps (max(5, 5% of steps))."""
        if not self.step_throughputs:
            return 0.0
        n = len(self.step_throughputs)
        warmup = max(5, int(0.05 * n))
        steady = self.step_throughputs[warmup:] if n > warmup else self.step_throughputs
        return float(np.median(steady))

    def get_final_train_loss(self, last_k: int = 20) -> float:
        if not self.step_metrics:
            return 0.0
        k = min(last_k, len(self.step_metrics))
        losses = [m.train_loss for m in self.step_metrics[-k:]]
        return float(np.mean(losses))

    def get_mean_data_wait_ms(self) -> float:
        if not self.step_metrics:
            return 0.0
        return float(np.mean([m.data_wait_ms for m in self.step_metrics]))
