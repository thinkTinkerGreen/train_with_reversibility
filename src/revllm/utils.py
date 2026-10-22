"""Utility functions: determinism, RNG streams, device detection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> None:
    """Set random seed across all libraries and enable deterministic settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class ThetaRNG:
    """Dedicated deterministic RNG stream for midpoint_theta coefficient generation."""

    def __init__(self, seed: int = 1337 + 17) -> None:
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample_layer_theta(self, num_layers: int) -> list[float]:
        """Sample a_l for all layers: s in {+1, -1} uniform, a_l = s + 0.5 * U(-1, 1)."""
        thetas: list[float] = []
        for _ in range(num_layers):
            # Sample sign s in {-1, 1}
            s = 1.0 if torch.randint(0, 2, (1,), generator=self.generator).item() == 1 else -1.0
            # Sample u in [-1, 1]
            u = (torch.rand((1,), generator=self.generator).item() * 2.0) - 1.0
            a_l = s + 0.5 * u
            thetas.append(float(a_l))
        return thetas

    def get_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)


def get_device() -> torch.device:
    """Return cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_mfu(
    num_params: int,
    tokens_per_sec: float,
    device_name: str,
    layers: int = 10,
    hidden_size: int = 384,
    context_len: int = 1024,
) -> float:
    """Estimate Model FLOPs Utilization (MFU).

    Peak T4 FP16 Tensor Core = 65 TFLOPs.
    """
    # 6 * N * tokens_per_sec + attention term
    flops_per_token = 6 * num_params + 12 * layers * 6 * 64 * context_len
    flops_achieved = flops_per_token * tokens_per_sec

    # Standard peak throughput estimates (TFLOPs in fp16/bf16)
    peak_tflops = 65.0  # Default: NVIDIA Tesla T4
    dev_lower = device_name.lower()
    if "a100" in dev_lower:
        peak_tflops = 312.0
    elif "v100" in dev_lower:
        peak_tflops = 125.0
    elif "l4" in dev_lower:
        peak_tflops = 120.0
    elif "t4" in dev_lower:
        peak_tflops = 65.0

    peak_flops = peak_tflops * 1e12
    return float((flops_achieved / peak_flops) * 100.0)
