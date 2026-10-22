"""Unit and property tests for reversible dynamics, reconstruction accuracy, and gradient equivalence."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from revllm.blocks import TransformerBlock, precompute_rope_freqs_cis
from revllm.config import ModelConfig
from revllm.reversible import ReversibleFunction, verify_reconstruction
from revllm.utils import ThetaRNG, set_seed


@pytest.mark.parametrize("arch", ["midpoint", "midpoint_theta", "leapfrog"])
@pytest.mark.parametrize("num_layers", [2, 8, 32])
def test_reconstruction_exact(arch: str, num_layers: int) -> None:
    """In fp32, backward reconstructed states match forward states within 1e-5 across depths."""
    set_seed(1337)
    cfg = ModelConfig(
        layers=num_layers,
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        mlp_hidden_size=128,
        vocab_size=256,
        context_len=64,
    )
    blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(num_layers)])
    freqs_cis = precompute_rope_freqs_cis(cfg.head_dim, cfg.context_len)
    x0 = torch.randn(2, cfg.context_len, cfg.hidden_size, dtype=torch.float32)

    theta_rng = ThetaRNG(seed=2024)
    thetas = theta_rng.sample_layer_theta(num_layers) if arch == "midpoint_theta" else None
    h = 0.5 if arch == "midpoint" else 1.0

    max_abs_err, rel_l2_err = verify_reconstruction(
        blocks,
        x0,
        freqs_cis,
        arch=arch,
        h=h,
        thetas=thetas,
        autocast_enabled=False,
    )

    assert max_abs_err <= 1e-5, (
        f"Reconstruction max abs error {max_abs_err} exceeded 1e-5 for {arch} at L={num_layers}"
    )
    assert rel_l2_err <= 1e-5


@pytest.mark.parametrize("arch", ["midpoint", "midpoint_theta", "leapfrog"])
def test_grad_equivalence(arch: str) -> None:
    """Custom reversible autograd backward matches plain autograd within cosine >= 0.9999 and rel-L2 <= 1e-4."""
    set_seed(42)
    cfg = ModelConfig(
        layers=2,
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        mlp_hidden_size=128,
        vocab_size=256,
        context_len=32,
    )

    blocks_plain = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
    blocks_rev = copy.deepcopy(blocks_plain)

    freqs_cis = precompute_rope_freqs_cis(cfg.head_dim, cfg.context_len)
    x0 = torch.randn(2, cfg.context_len, cfg.hidden_size, dtype=torch.float32, requires_grad=True)
    x0_plain = x0.detach().clone().requires_grad_(True)

    thetas = [0.85, -0.6] if arch == "midpoint_theta" else None
    h = 0.5 if arch == "midpoint" else 1.0

    # 1. Plain autograd
    p_prev = x0_plain
    p_curr = x0_plain
    for l_idx, block in enumerate(blocks_plain):
        f_l = block(p_curr, freqs_cis).to(torch.float32)
        if arch == "midpoint":
            p_next = p_prev + (2.0 * h) * f_l
        elif arch == "midpoint_theta":
            a_l = thetas[l_idx] if thetas is not None else 0.0
            p_next = a_l * p_prev + (1.0 - a_l) * p_curr + h * f_l
        elif arch == "leapfrog":
            p_next = 2.0 * p_curr - p_prev + (h * h) * f_l
        p_prev = p_curr
        p_curr = p_next

    loss_plain = (p_curr**2).mean()
    loss_plain.backward()

    # 2. Reversible custom backward
    all_params = [p for p in blocks_rev.parameters() if p.requires_grad]
    p_out_rev = ReversibleFunction.apply(
        x0,
        freqs_cis,
        arch,
        h,
        thetas,
        False,
        torch.float32,
        blocks_rev,
        *all_params,
    )
    loss_rev = (p_out_rev**2).mean()
    loss_rev.backward()

    # Compare gradients
    plain_grads: list[torch.Tensor] = []
    rev_grads: list[torch.Tensor] = []

    for p_p, p_r in zip(blocks_plain.parameters(), blocks_rev.parameters()):
        if p_p.grad is not None and p_r.grad is not None:
            plain_grads.append(p_p.grad.flatten())
            rev_grads.append(p_r.grad.flatten())

    vec_plain = torch.cat(plain_grads)
    vec_rev = torch.cat(rev_grads)

    cosine = float(
        torch.dot(vec_plain, vec_rev) / (torch.norm(vec_plain) * torch.norm(vec_rev) + 1e-12)
    )
    rel_l2 = float(torch.norm(vec_plain - vec_rev) / (torch.norm(vec_plain) + 1e-12))

    assert cosine >= 0.9999, f"Gradient cosine similarity {cosine} < 0.9999 for {arch}"
    assert rel_l2 <= 1e-4, f"Gradient relative L2 error {rel_l2} > 1e-4 for {arch}"


def test_theta_rng_reproducible() -> None:
    """Same seed produces identical sequence of a_l values."""
    rng1 = ThetaRNG(seed=1354)
    rng2 = ThetaRNG(seed=1354)

    seq1 = rng1.sample_layer_theta(10)
    seq2 = rng2.sample_layer_theta(10)

    assert seq1 == seq2
    assert len(seq1) == 10
    # Values must be within [-1.5, -0.5] or [0.5, 1.5]
    for val in seq1:
        assert (0.5 <= val <= 1.5) or (-1.5 <= val <= -0.5)
