"""Configuration dataclasses and loaders for revllm."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    layers: int = 10
    hidden_size: int = 384
    num_heads: int = 6
    head_dim: int = 64
    mlp_hidden_size: int = 1024
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    qk_norm: bool = True
    context_len: int = 1024
    vocab_size: int = 8192
    tie_embeddings: bool = True
    dropout: float = 0.0
    residual_dtype: str = "float32"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = True
    weight_decay: float = 0.1


@dataclass
class LRScheduleConfig:
    base_lr: float = 1e-3
    max_lr: float = 3e-3
    ref_tokens_per_step: int = 65536
    warmup_ratio: float = 0.03
    min_warmup_steps: int = 5
    min_lr_ratio: float = 0.1


@dataclass
class TrainingParams:
    total_tokens: int = 50_000_000
    grad_clip: float = 1.0
    precision: str = "fp16"
    loss_chunk_size: int = 4096
    seed: int = 1337
    grad_scaler_init: int = 16384
    eval_points: int = 20
    log_every: int = 10
    checkpoint_points: int = 10


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    training: TrainingParams = field(default_factory=TrainingParams)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            lr_schedule=LRScheduleConfig(**data.get("lr_schedule", {})),
            training=TrainingParams(**data.get("training", {})),
        )


@dataclass
class RunConfig:
    run_id: str
    arch: str
    h: float | None = None
    batch_size: int = 32
    seed: int = 1337
    compile: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class DataConfig:
    dataset: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
