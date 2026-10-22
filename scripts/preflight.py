"""Colab Preflight Verification Script.

Tests forward/backward across architectures, compiles, reconstruction accuracy,
and gradient equivalence vs standard autograd.
Outputs results/preflight.json.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from revllm.blocks import TransformerBlock, precompute_rope_freqs_cis
from revllm.config import ModelConfig
from revllm.model import TransformerLM
from revllm.reversible import ReversibleFunction, verify_reconstruction
from revllm.utils import ThetaRNG, get_device, set_seed


def test_grad_equivalence(
    model_cfg: ModelConfig,
    arch: str,
    h: float = 0.5,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Test custom reversible backward gradients vs plain autograd of identical math."""
    device = device or get_device()
    set_seed(42)

    # 2-layer miniature model for clean gradient check
    cfg = copy.deepcopy(model_cfg)
    cfg.layers = 2
    cfg.context_len = 128
    cfg.hidden_size = 64
    cfg.num_heads = 2
    cfg.head_dim = 32
    cfg.mlp_hidden_size = 128

    blocks_plain = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)]).to(device)
    blocks_rev = copy.deepcopy(blocks_plain)

    freqs_cis = precompute_rope_freqs_cis(cfg.head_dim, cfg.context_len, device=device)
    x0 = torch.randn(
        2, cfg.context_len, cfg.hidden_size, device=device, dtype=torch.float32, requires_grad=True
    )
    x0_plain = x0.detach().clone().requires_grad_(True)

    thetas = [0.85, -0.6] if arch == "midpoint_theta" else None

    # 1. Plain autograd forward & backward
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
        else:
            raise ValueError(f"Unknown arch: {arch}")
        p_prev = p_curr
        p_curr = p_next

    loss_plain = (p_curr**2).mean()
    loss_plain.backward()

    # 2. Reversible custom forward & backward
    all_params = [p for p in blocks_rev.parameters() if p.requires_grad]
    p_out_rev = ReversibleFunction.apply(
        x0,
        freqs_cis,
        arch,
        h,
        thetas,
        False,  # autocast
        torch.float32,
        blocks_rev,
        *all_params,
    )
    loss_rev = (p_out_rev**2).mean()
    loss_rev.backward()

    # Compare gradients across all parameters
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

    return cosine, rel_l2


def run_preflight() -> dict:
    device = get_device()
    print(f"Running preflight on device: {device}")

    results = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "arch_checks": {},
        "reconstruction_checks": {},
        "gradient_equivalence": {},
        "compile_status": False,
    }

    model_cfg = ModelConfig()

    # (a) Check forward & backward on all architectures
    archs = ["baseline", "midpoint", "midpoint_theta", "leapfrog", "hamiltonian"]
    tokens = torch.randint(0, 100, (2, 64), device=device)
    targets = torch.randint(0, 100, (2, 64), device=device)

    theta_rng = ThetaRNG(seed=1354)

    for arch in archs:
        print(f"Checking arch: {arch}...")
        model = TransformerLM(model_cfg, arch=arch).to(device)
        thetas = (
            theta_rng.sample_layer_theta(model_cfg.layers) if arch == "midpoint_theta" else None
        )

        # Eager fwd + bwd
        _, loss_eager = model(tokens, targets, thetas=thetas)
        assert loss_eager is not None
        loss_eager.backward()

        results["arch_checks"][arch] = {
            "eager_loss": float(loss_eager.item()),
            "status": "PASS",
        }

    # (b) Reconstruction tests
    print("Testing reconstruction precision...")
    test_archs = ["midpoint", "midpoint_theta", "leapfrog"]
    dummy_x0 = torch.randn(2, 128, model_cfg.hidden_size, device=device, dtype=torch.float32)
    dummy_freqs = precompute_rope_freqs_cis(model_cfg.head_dim, 128, device=device)

    for arch in test_archs:
        test_model = TransformerLM(model_cfg, arch=arch).to(device)
        thetas = (
            theta_rng.sample_layer_theta(model_cfg.layers) if arch == "midpoint_theta" else None
        )
        h_val = 0.5 if arch == "midpoint" else 1.0

        # FP32 check
        max_abs_fp32, rel_l2_fp32 = verify_reconstruction(
            test_model.blocks,
            dummy_x0,
            dummy_freqs,
            arch=arch,
            h=h_val,
            thetas=thetas,
            autocast_enabled=False,
        )

        # FP16 check if cuda
        max_abs_fp16, rel_l2_fp16 = verify_reconstruction(
            test_model.blocks,
            dummy_x0,
            dummy_freqs,
            arch=arch,
            h=h_val,
            thetas=thetas,
            autocast_enabled=device.type == "cuda",
            autocast_dtype=torch.float16,
        )

        results["reconstruction_checks"][arch] = {
            "fp32_max_abs": max_abs_fp32,
            "fp32_rel_l2": rel_l2_fp32,
            "fp16_max_abs": max_abs_fp16,
            "fp16_rel_l2": rel_l2_fp16,
            "pass_fp32": bool(max_abs_fp32 <= 1e-4),
            "pass_fp16": bool(max_abs_fp16 <= 5e-3),
        }

    # (c) Gradient Equivalence tests
    print("Testing gradient equivalence against standard autograd...")
    for arch in ["midpoint", "midpoint_theta", "leapfrog"]:
        cosine, rel_l2 = test_grad_equivalence(model_cfg, arch=arch, device=device)
        results["gradient_equivalence"][arch] = {
            "cosine_similarity": cosine,
            "relative_l2_error": rel_l2,
            "pass": bool(cosine >= 0.999 and rel_l2 <= 1e-2),
        }
        print(f"  {arch}: Cosine={cosine:.6f}, Rel-L2={rel_l2:.6f}")

    # (d) Check compile support
    print("Testing torch.compile...")
    can_compile = False
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            m_comp = TransformerLM(model_cfg, arch="baseline").to(device)
            m_comp.blocks[0] = torch.compile(m_comp.blocks[0])  # type: ignore
            _, loss_comp = m_comp(tokens, targets)
            assert loss_comp is not None
            loss_comp.backward()
            can_compile = True
        except Exception as e:
            print(f"Compile preflight failed or unsupported: {e}")
            can_compile = False

    results["compile_status"] = can_compile
    print(f"Compile decision: {can_compile}")

    output_path = Path("results/preflight.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Preflight finished successfully. Output written to {output_path}")
    return results


if __name__ == "__main__":
    run_preflight()
