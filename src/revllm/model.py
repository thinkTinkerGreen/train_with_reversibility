"""Decoder-only Transformer LM with baseline and reversible dynamics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from revllm.blocks import RMSNorm, TransformerBlock, precompute_rope_freqs_cis
from revllm.config import ModelConfig
from revllm.loss import chunked_cross_entropy
from revllm.reversible import ReversibleFunction


class TransformerLM(nn.Module):
    """Decoder-only Transformer supporting baseline (Euler) and reversible residual dynamics."""

    def __init__(self, config: ModelConfig, arch: str = "baseline", h: float | None = None) -> None:
        super().__init__()
        self.config = config
        self.arch = arch
        self.h = h if h is not None else (0.5 if arch == "midpoint" else 1.0)

        # Token embedding (no positional embeddings; RoPE handles positions)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])

        # Final RMSNorm
        self.norm_final = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Precomputed RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs_cis(
                dim=config.head_dim,
                max_seq_len=config.context_len,
                theta=config.rope_base,
            ),
            persistent=False,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights to Normal(0, 0.02).

        Output projections scaled by 1/sqrt(2L).
        """
        L = self.config.layers
        std_init = 0.02
        scaled_std = std_init / math.sqrt(2 * L)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "out_proj" in name or "down_proj" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std_init)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_init)

    def count_parameters(self) -> int:
        """Count total unique parameters (tied LM head counted once)."""
        unique_params = set(self.parameters())
        return sum(p.numel() for p in unique_params)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        thetas: Sequence[float] | None = None,
        chunk_size: int = 4096,
        autocast_enabled: bool = False,
        autocast_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            tokens: [B, T] tensor of input token IDs
            targets: [B, T] tensor of target token IDs (optional)
            thetas: Sequence of layer-wise random coefficients for midpoint_theta
            chunk_size: Token chunk size for chunked cross entropy
            autocast_enabled: Whether to run block compute under autocast
            autocast_dtype: Autocast floating-point type (fp16 on T4)

        Returns:
            (hidden_states_or_logits, loss)
        """
        B, T = tokens.shape
        x0 = self.tok_embeddings(tokens).to(torch.float32)
        freqs_cis = self.freqs_cis[:T]

        if self.arch == "baseline":
            p = x0
            for block in self.blocks:
                device_type = "cuda" if p.is_cuda else "cpu"
                with torch.amp.autocast(
                    device_type=device_type, enabled=autocast_enabled, dtype=autocast_dtype
                ):
                    f_out = block(p, freqs_cis).to(torch.float32)
                p = p + f_out
            final_hidden = p
        elif self.arch in ["midpoint", "midpoint_theta", "leapfrog"]:
            all_params = [p for p in self.parameters() if p.requires_grad]
            final_hidden = ReversibleFunction.apply(
                x0,
                freqs_cis,
                self.arch,
                self.h,
                thetas,
                autocast_enabled,
                autocast_dtype,
                self.blocks,
                *all_params,
            )
        elif self.arch == "hamiltonian":
            p = x0
            q = x0.clone()
            device_type = "cuda" if p.is_cuda else "cpu"
            for block in self.blocks:
                with torch.amp.autocast(
                    device_type=device_type, enabled=autocast_enabled, dtype=autocast_dtype
                ):
                    # q(l) = q(l-1) + Attn(LN1(p(l-1)))
                    q = q + block.attn(block.norm1(p), freqs_cis).to(torch.float32)
                    # p(l) = p(l-1) + MLP(LN2(q(l)))
                    p = p + block.mlp(block.norm2(q)).to(torch.float32)
            final_hidden = 0.5 * (p + q)
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

        y = self.norm_final(final_hidden)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = chunked_cross_entropy(
                hidden_states=y,
                targets=targets,
                lm_head_weight=self.tok_embeddings.weight,
                chunk_size=chunk_size,
            )

        return y, loss
