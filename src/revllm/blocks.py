"""Transformer building blocks: RMSNorm, RoPE, Attention, SwiGLU MLP, and Block update."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from revllm.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


def apply_qk_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Non-parametric RMS normalization applied to Q and K per head."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def precompute_rope_freqs_cis(
    dim: int,
    max_seq_len: int = 1024,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE)."""
    # dim must be even (head_dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to tensor x of shape [B, num_heads, T, head_dim]."""
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(1)  # [1, 1, T, D // 2]
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_rotated.type_as(x)


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with QK-norm and RoPE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = apply_qk_norm(q)
            k = apply_qk_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block implementing f_l(p):

    f_l(p) = Attn_l(RMSNorm1(p)) + MLP_l(RMSNorm2(p + Attn_l(RMSNorm1(p))))
    This is the pure block update excluding the identity residual term.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.mlp_hidden_size)

    def forward(self, p: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(p), freqs_cis)
        mid = p + attn_out
        mlp_out = self.mlp(self.norm2(mid))
        return attn_out + mlp_out
