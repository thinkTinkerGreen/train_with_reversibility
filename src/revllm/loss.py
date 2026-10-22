"""Chunked Cross-Entropy Loss to prevent vocabulary logits from dominating VRAM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce_step(
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Project hidden states to vocabulary logits and compute cross entropy for one chunk."""
    logits = F.linear(hidden_chunk, lm_head_weight)  # [Chunk, Vocab]
    loss = F.cross_entropy(logits, target_chunk, reduction="sum")
    return loss


def chunked_cross_entropy(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute cross-entropy loss over flattened (B*T) tokens in chunks wrapped in checkpointing.

    Args:
        hidden_states: [B, T, D] tensor of hidden states from final RMSNorm
        targets: [B, T] tensor of target token IDs
        lm_head_weight: [Vocab, D] tied embedding weights
        chunk_size: Number of tokens per projection chunk (default: 4096)

    Returns:
        Scalar mean cross-entropy loss.
    """
    B, T, D = hidden_states.shape
    total_tokens = B * T

    flat_hidden = hidden_states.reshape(total_tokens, D)
    flat_targets = targets.reshape(total_tokens)

    # If small number of tokens or smoke test, chunk_size can be clamped
    actual_chunk_size = min(chunk_size, total_tokens)
    if actual_chunk_size <= 0:
        actual_chunk_size = total_tokens

    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

    for start_idx in range(0, total_tokens, actual_chunk_size):
        end_idx = min(start_idx + actual_chunk_size, total_tokens)
        h_chunk = flat_hidden[start_idx:end_idx]
        t_chunk = flat_targets[start_idx:end_idx]

        chunk_loss = checkpoint(
            _chunk_ce_step,
            h_chunk,
            t_chunk,
            lm_head_weight,
            use_reentrant=False,
        )
        total_loss = total_loss + chunk_loss

    return total_loss / total_tokens
