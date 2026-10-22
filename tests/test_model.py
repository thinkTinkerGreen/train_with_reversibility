"""Unit tests for model architecture, parameter count, forward shapes, and config diffs."""

from __future__ import annotations

import pytest
import torch
import yaml

from revllm.config import ModelConfig
from revllm.model import TransformerLM


def test_param_count() -> None:
    """Assert full-size model parameter count equals exactly 20,848,512 and is in [19.0M, 21.0M]."""
    config = ModelConfig()
    model = TransformerLM(config, arch="baseline")
    total_params = model.count_parameters()

    assert total_params == 20_848_512, f"Expected 20,848,512 params, got {total_params}"
    assert 19_000_000 <= total_params <= 21_000_000


@pytest.mark.parametrize(
    "arch", ["baseline", "midpoint", "midpoint_theta", "leapfrog", "hamiltonian"]
)
def test_arch_forward_shapes(arch: str) -> None:
    """Every architecture maps [B, T] token IDs to [B, T, D] hidden states and computes loss without NaN."""
    # Use 2-layer miniature config for fast unit testing
    config = ModelConfig(
        layers=2, hidden_size=64, num_heads=2, head_dim=32, mlp_hidden_size=128, vocab_size=256
    )
    model = TransformerLM(config, arch=arch)

    B, T = 2, 16
    tokens = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    thetas = [0.8, -0.5] if arch == "midpoint_theta" else None

    y, loss = model(tokens, targets, thetas=thetas)

    assert y.shape == (B, T, config.hidden_size)
    assert not torch.isnan(y).any()
    assert loss is not None
    assert not torch.isnan(loss).any()
    assert loss.item() > 0


def test_config_a_vs_b_differ_only_in_arch() -> None:
    """Diff of A.yaml and B.yaml must be exactly the 'arch' (and optional 'h') keys."""
    with open("configs/runs/A.yaml") as f:
        cfg_a = yaml.safe_load(f)
    with open("configs/runs/B.yaml") as f:
        cfg_b = yaml.safe_load(f)

    # Remove run_id, arch, and h
    filtered_a = {k: v for k, v in cfg_a.items() if k not in ["run_id", "arch", "h"]}
    filtered_b = {k: v for k, v in cfg_b.items() if k not in ["run_id", "arch", "h"]}

    assert filtered_a == filtered_b, (
        f"Config A and B differ in keys beyond arch/h/run_id: {filtered_a} != {filtered_b}"
    )
