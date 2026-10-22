"""Unit tests for chunked cross-entropy loss equivalence and gradient matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from revllm.loss import chunked_cross_entropy


def test_chunked_ce_matches_full() -> None:
    """Chunked cross-entropy loss and gradients match unchunked cross-entropy within 1e-5."""
    B, T, D, V = 4, 64, 32, 128
    torch.manual_seed(1337)

    # Hidden states and tied weight
    hidden_full = torch.randn(B, T, D, requires_grad=True)
    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)

    weight_full = torch.randn(V, D, requires_grad=True)
    weight_chunked = weight_full.detach().clone().requires_grad_(True)

    targets = torch.randint(0, V, (B, T))

    # 1. Full unchunked CE
    logits = F.linear(hidden_full, weight_full)  # [B, T, V]
    loss_full = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    loss_full.backward()

    # 2. Chunked CE with chunk size smaller than total tokens (B*T = 256, chunk=64)
    loss_chunked = chunked_cross_entropy(
        hidden_states=hidden_chunked,
        targets=targets,
        lm_head_weight=weight_chunked,
        chunk_size=64,
    )
    loss_chunked.backward()

    # Verify loss equality
    assert torch.isclose(loss_full, loss_chunked, atol=1e-5)

    # Verify hidden gradient equality
    assert hidden_full.grad is not None and hidden_chunked.grad is not None
    assert torch.allclose(hidden_full.grad, hidden_chunked.grad, atol=1e-5)

    # Verify weight gradient equality
    assert weight_full.grad is not None and weight_chunked.grad is not None
    assert torch.allclose(weight_full.grad, weight_chunked.grad, atol=1e-5)
