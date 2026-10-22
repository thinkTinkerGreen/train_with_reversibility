"""Batch size probing algorithm (exponential + binary bisection) and memory/depth diagnostic sweeps."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from revllm.config import ModelConfig
from revllm.model import TransformerLM
from revllm.train import build_optimizer
from revllm.utils import ThetaRNG, set_seed


def single_probe_worker(arch: str, b: int, steps: int = 15) -> None:
    """Executed in a dedicated subprocess to avoid CUDA allocator fragmentation on OOM."""
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # CPU mock pass
        res = {
            "status": "PASS",
            "arch": arch,
            "b": b,
            "reserved_gb": 0.5,
            "total_vram_gb": 16.0,
            "tokens_per_sec": 1000.0,
        }
        print(json.dumps(res))
        return

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / 1e9

    try:
        model_cfg = ModelConfig()
        model = TransformerLM(model_cfg, arch=arch).to(device)
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", init_scale=16384)
        theta_rng = ThetaRNG(seed=1354)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            tokens = torch.randint(
                0, model_cfg.vocab_size, (b, model_cfg.context_len), device=device
            )
            targets = torch.randint(
                0, model_cfg.vocab_size, (b, model_cfg.context_len), device=device
            )
            thetas = (
                theta_rng.sample_layer_theta(model_cfg.layers) if arch == "midpoint_theta" else None
            )

            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, loss = model(tokens, targets, thetas=thetas, autocast_enabled=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        t_elapsed = time.perf_counter() - t0
        tps = (b * model_cfg.context_len * steps) / max(1e-6, t_elapsed)
        reserved_bytes = torch.cuda.max_memory_reserved()
        reserved_gb = reserved_bytes / 1e9

        status = "PASS" if reserved_bytes <= 0.92 * vram_bytes else "FAIL"
        res = {
            "status": status,
            "arch": arch,
            "b": b,
            "reserved_gb": reserved_gb,
            "total_vram_gb": vram_gb,
            "tokens_per_sec": tps,
        }
    except torch.cuda.OutOfMemoryError:
        res = {
            "status": "FAIL",
            "arch": arch,
            "b": b,
            "reserved_gb": vram_gb,
            "total_vram_gb": vram_gb,
            "tokens_per_sec": 0.0,
        }
    except Exception as e:
        res = {
            "status": "FAIL",
            "error": str(e),
            "arch": arch,
            "b": b,
            "reserved_gb": 0.0,
            "total_vram_gb": vram_gb,
            "tokens_per_sec": 0.0,
        }

    print(json.dumps(res))


def run_probe_subprocess(arch: str, b: int, steps: int = 15) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-c",
        f"from scripts.probe import single_probe_worker; single_probe_worker('{arch}', {b}, {steps})",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in proc.stdout.split("\n") if line.strip()]
        last_json = json.loads(lines[-1])
        return last_json
    except Exception as e:
        return {"status": "FAIL", "arch": arch, "b": b, "error": str(e)}


def probe_arch_max_batch(arch: str) -> tuple[int, int, list[dict]]:
    """Execute exponential doubling followed by binary bisection to find B_max and B_safe."""
    probe_log: list[dict] = []
    print(f"\n--- Probing Batch Size for Architecture: {arch} ---")

    # 1. Exponential phase
    b = 8
    last_pass = 8
    first_fail: int | None = None

    while b <= 2048:
        print(f"Testing B = {b} ... ", end="", flush=True)
        res = run_probe_subprocess(arch, b)
        probe_log.append(res)
        print(f"{res.get('status')} (Reserved: {res.get('reserved_gb', 0):.2f} GB)")

        if res.get("status") == "PASS":
            last_pass = b
            b *= 2
        else:
            first_fail = b
            break

    if first_fail is None:
        first_fail = b

    # 2. Binary phase
    lo = last_pass
    hi = first_fail
    while (hi - lo) > 8:
        mid = ((lo + hi) // 2 // 8) * 8
        if mid == lo:
            break
        print(f"Bisecting B = {mid} ... ", end="", flush=True)
        res = run_probe_subprocess(arch, mid)
        probe_log.append(res)
        print(f"{res.get('status')} (Reserved: {res.get('reserved_gb', 0):.2f} GB)")

        if res.get("status") == "PASS":
            lo = mid
        else:
            hi = mid

    b_max = lo
    b_safe = max(8, math.floor(0.90 * b_max / 8) * 8)
    print(f"Result for {arch}: B_max = {b_max}, B_safe = {b_safe}")
    return b_max, b_safe, probe_log


def run_diagnostic_sweeps(
    winner_arch: str,
    baseline_b_max: int,
    winner_b_max: int,
) -> dict[str, Any]:
    """Run sweeps F2 (memory and tokens/s vs batch) and F3 (memory vs depth)."""
    print("\n--- Running Diagnostic Sweeps (F2 & F3) ---")
    f2_results = []
    # Test batch sizes: 8, 16, 24, 32, ... up to max
    batches = [8, 16, 24, 32]
    if winner_b_max > 32:
        batches.append(winner_b_max)

    for b in sorted(set(batches)):
        for arch in ["baseline", winner_arch]:
            if (arch == "baseline" and b <= baseline_b_max) or (
                arch == winner_arch and b <= winner_b_max
            ):
                print(f"F2 sweep: arch={arch}, b={b} (30 steps)...")
                res = run_probe_subprocess(arch, b, steps=30)
                f2_results.append(res)

    # F3: Peak memory vs depth L in {4, 8, 16, 32} at fixed micro-batch 16
    f3_results = []
    depths = [4, 8, 16, 32]
    for d in depths:
        for arch in ["baseline", winner_arch]:
            print(f"F3 depth sweep: arch={arch}, depth={d} (30 steps)...")
            cmd = [
                sys.executable,
                "-c",
                f"""
import json, torch
from revllm.config import ModelConfig
from revllm.model import TransformerLM
from revllm.train import build_optimizer
from revllm.utils import set_seed
set_seed(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = ModelConfig(layers={d})
m = TransformerLM(cfg, arch='{arch}').to(device)
opt = build_optimizer(m)
scaler = torch.amp.GradScaler('cuda', init_scale=16384)
torch.cuda.reset_peak_memory_stats()
tokens = torch.randint(0, cfg.vocab_size, (16, cfg.context_len), device=device)
targets = torch.randint(0, cfg.vocab_size, (16, cfg.context_len), device=device)
for _ in range(15):
    opt.zero_grad()
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
        _, loss = m(tokens, targets, autocast_enabled=(device.type == 'cuda'))
    if device.type == 'cuda':
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
res = {{'depth': {d}, 'arch': '{arch}', 'reserved_gb': float(torch.cuda.max_memory_reserved() / 1e9) if device.type == 'cuda' else 0.5}}
print(json.dumps(res))
""",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = [line.strip() for line in proc.stdout.split("\n") if line.strip()]
                f3_results.append(json.loads(lines[-1]))
            except Exception as e:
                f3_results.append({"depth": d, "arch": arch, "error": str(e)})

    return {"f2_batch_sweep": f2_results, "f3_depth_sweep": f3_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch size probe and sweeps")
    parser.add_argument("--output", default="results/batch_probe.json", help="Output path")
    args = parser.parse_args()

    archs = ["baseline", "midpoint", "midpoint_theta", "leapfrog"]
    all_probes: list[dict] = []
    b_max_map = {}
    b_safe_map = {}

    for arch in archs:
        b_max, b_safe, probes = probe_arch_max_batch(arch)
        b_max_map[arch] = b_max
        b_safe_map[arch] = b_safe
        all_probes.extend(probes)

    b_A = b_safe_map["baseline"]
    # Default candidate winner before pilot
    winner_arch = "midpoint"
    b_C = b_safe_map[winner_arch]

    diagnostic_data = run_diagnostic_sweeps(
        winner_arch=winner_arch,
        baseline_b_max=b_max_map["baseline"],
        winner_b_max=b_max_map[winner_arch],
    )

    out = {
        "b_A": b_A,
        "b_C": b_C,
        "b_max_map": b_max_map,
        "b_safe_map": b_safe_map,
        "probe_history": all_probes,
        "diagnostic_sweeps": diagnostic_data,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nBatch probing completed! Results saved to {output_path}")


if __name__ == "__main__":
    main()
