"""Reversible residual stream dynamics and custom PyTorch autograd functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from revllm.blocks import TransformerBlock


def run_block_forward(
    block: TransformerBlock,
    p: torch.Tensor,
    freqs_cis: torch.Tensor,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run block update f_l(p) with optional autocast, keeping result in float32."""
    device_type = "cuda" if p.is_cuda else "cpu"
    with torch.amp.autocast(
        device_type=device_type, enabled=autocast_enabled, dtype=autocast_dtype
    ):
        out = block(p, freqs_cis)
    return out.to(torch.float32)


class ReversibleFunction(torch.autograd.Function):
    """Memory-efficient reversible autograd function.

    Forward caches only the boundary states (p(L) and p(L-1)) and reconstructs
    all intermediate layer activations backward during reverse propagation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x0: torch.Tensor,
        freqs_cis: torch.Tensor,
        arch: str,
        h: float,
        thetas: Sequence[float] | None,
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
        blocks: Sequence[TransformerBlock],
        *all_params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.arch = arch
        ctx.h = h
        ctx.thetas = list(thetas) if thetas is not None else None
        ctx.autocast_enabled = autocast_enabled
        ctx.autocast_dtype = autocast_dtype
        ctx.blocks = blocks
        ctx.freqs_cis = freqs_cis
        ctx.num_layers = len(blocks)

        L = len(blocks)
        p_prev = x0.to(torch.float32)  # p(-1) = x0
        p_curr = x0.to(torch.float32)  # p(0) = x0

        with torch.no_grad():
            for l_idx in range(L):
                f_l = run_block_forward(
                    blocks[l_idx],
                    p_curr,
                    freqs_cis,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )
                if arch == "midpoint":
                    p_next = p_prev + (2.0 * h) * f_l
                elif arch == "midpoint_theta":
                    a_l = ctx.thetas[l_idx] if ctx.thetas is not None else 0.0
                    p_next = a_l * p_prev + (1.0 - a_l) * p_curr + h * f_l
                elif arch == "leapfrog":
                    p_next = 2.0 * p_curr - p_prev + (h * h) * f_l
                else:
                    raise ValueError(f"Unsupported reversible arch: {arch}")

                p_prev = p_curr
                p_curr = p_next

        # Save boundary states for backward reconstruction
        ctx.save_for_backward(p_curr, p_prev, x0)
        return p_curr

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        arch: str = ctx.arch
        h: float = ctx.h
        thetas: list[float] | None = ctx.thetas
        autocast_enabled: bool = ctx.autocast_enabled
        autocast_dtype: torch.dtype = ctx.autocast_dtype
        blocks: Sequence[TransformerBlock] = ctx.blocks
        freqs_cis: torch.Tensor = ctx.freqs_cis
        num_layers: int = ctx.num_layers

        p_curr, p_prev, x0 = ctx.saved_tensors

        # grad_output is dLoss / dp(L)
        grad_curr = grad_output.to(torch.float32)
        grad_prev = torch.zeros_like(grad_curr)

        # Iterate backward through layers L-1 down to 0
        for l_idx in range(num_layers - 1, -1, -1):
            block = blocks[l_idx]

            # Reconstruct p(l-1) using p(l+1) [p_curr] and p(l) [p_prev]
            # p_curr corresponds to p(l+1), p_prev corresponds to p(l)
            # We need f_l(p(l)) with gradients enabled for backward
            p_l = p_prev.detach().requires_grad_(True)
            with torch.enable_grad():
                f_l = run_block_forward(
                    block,
                    p_l,
                    freqs_cis,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )

            # 1. Reconstruct p(l-1)
            with torch.no_grad():
                if arch == "midpoint":
                    # p(l+1) = p(l-1) + 2h*f(p(l)) => p(l-1) = p(l+1) - 2h*f(p(l))
                    p_before = p_curr - (2.0 * h) * f_l
                    grad_f = (2.0 * h) * grad_curr
                    grad_p_before = grad_curr
                    grad_p_l_direct = grad_prev
                elif arch == "midpoint_theta":
                    a_l = thetas[l_idx] if thetas is not None else 0.0
                    # p(l+1) = a_l*p(l-1) + (1-a_l)*p(l) + h*f_l =>
                    # p(l-1) = (p(l+1) - (1-a_l)*p(l) - h*f_l) / a_l
                    p_before = (p_curr - (1.0 - a_l) * p_prev - h * f_l) / a_l
                    grad_f = h * grad_curr
                    grad_p_before = a_l * grad_curr
                    grad_p_l_direct = grad_prev + (1.0 - a_l) * grad_curr
                elif arch == "leapfrog":
                    # p(l+1) = 2p(l) - p(l-1) + h^2*f_l => p(l-1) = 2p(l) - p(l+1) + h^2*f_l
                    p_before = 2.0 * p_prev - p_curr + (h * h) * f_l
                    grad_f = (h * h) * grad_curr
                    grad_p_before = -grad_curr
                    grad_p_l_direct = grad_prev + 2.0 * grad_curr
                else:
                    raise ValueError(f"Unknown arch: {arch}")

            # 2. Backpropagate through f_l to accumulate parameter gradients in block
            params = [p for p in block.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                outputs=f_l,
                inputs=[p_l] + params,
                grad_outputs=grad_f,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            grad_p_l_from_f = grads[0]
            param_grads = grads[1:]
            for p, g in zip(params, param_grads):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach().clone()
                    else:
                        p.grad.add_(g.detach())

            # Update state gradients for the next iteration (layer l-1)
            grad_curr = grad_p_l_direct + (grad_p_l_from_f if grad_p_l_from_f is not None else 0.0)
            grad_prev = grad_p_before

            # Shift state pointers
            p_curr = p_prev
            p_prev = p_before

        # At l = 0, p(-1) and p(0) are both x0
        grad_x0 = grad_curr + grad_prev

        # Match outputs to forward arguments
        # x0, freqs_cis, arch, h, thetas, autocast_enabled, autocast_dtype, blocks, *all_params
        param_grads_list: list[torch.Tensor | None] = []
        for block in blocks:
            for p in block.parameters():
                param_grads_list.append(p.grad if p.requires_grad else None)

        return (
            grad_x0,
            None,  # freqs_cis
            None,  # arch
            None,  # h
            None,  # thetas
            None,  # autocast_enabled
            None,  # autocast_dtype
            None,  # blocks
            *param_grads_list,
        )


def verify_reconstruction(
    blocks: Sequence[TransformerBlock],
    x0: torch.Tensor,
    freqs_cis: torch.Tensor,
    arch: str,
    h: float,
    thetas: Sequence[float] | None = None,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    """Perform forward propagation storing all states, then backward reconstruction,

    and return (max_abs_error, rel_l2_error).
    """
    L = len(blocks)
    forward_states: list[torch.Tensor] = [x0.to(torch.float32)]  # p(0)
    p_prev = x0.to(torch.float32)  # p(-1)
    p_curr = x0.to(torch.float32)  # p(0)

    with torch.no_grad():
        for l_idx in range(L):
            f_l = run_block_forward(
                blocks[l_idx],
                p_curr,
                freqs_cis,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            if arch == "midpoint":
                p_next = p_prev + (2.0 * h) * f_l
            elif arch == "midpoint_theta":
                a_l = thetas[l_idx] if thetas is not None else 0.0
                p_next = a_l * p_prev + (1.0 - a_l) * p_curr + h * f_l
            elif arch == "leapfrog":
                p_next = 2.0 * p_curr - p_prev + (h * h) * f_l
            else:
                raise ValueError(f"Unknown arch: {arch}")

            p_prev = p_curr
            p_curr = p_next
            forward_states.append(p_curr.clone())

    # Reconstruct backwards
    reconstructed_states: list[torch.Tensor] = [p_curr.clone()]
    with torch.no_grad():
        for l_idx in range(L - 1, -1, -1):
            f_l = run_block_forward(
                blocks[l_idx],
                p_prev,
                freqs_cis,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            if arch == "midpoint":
                p_before = p_curr - (2.0 * h) * f_l
            elif arch == "midpoint_theta":
                a_l = thetas[l_idx] if thetas is not None else 0.0
                p_before = (p_curr - (1.0 - a_l) * p_prev - h * f_l) / a_l
            elif arch == "leapfrog":
                p_before = 2.0 * p_prev - p_curr + (h * h) * f_l
            else:
                raise ValueError(f"Unknown arch: {arch}")

            p_curr = p_prev
            p_prev = p_before
            reconstructed_states.append(p_curr.clone())

    # reconstructed_states has [p(L), p(L-1), ..., p(0)]
    reconstructed_states.reverse()

    max_abs_err = 0.0
    rel_l2_err = 0.0
    for fwd_p, rec_p in zip(forward_states, reconstructed_states):
        diff = torch.abs(fwd_p - rec_p)
        max_abs_err = max(max_abs_err, float(diff.max().item()))
        rel = torch.norm(fwd_p - rec_p) / (torch.norm(fwd_p) + 1e-8)
        rel_l2_err = max(rel_l2_err, float(rel.item()))

    return max_abs_err, rel_l2_err
