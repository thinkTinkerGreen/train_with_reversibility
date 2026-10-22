"""Full training loop, optimizer configuration, LR scheduling, and validation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from revllm.config import ModelConfig, RunConfig, TrainConfig
from revllm.data import MemmapDataset, PrefetchDataLoader
from revllm.drive import CheckpointManager
from revllm.gpu_monitor import GPUMonitor
from revllm.metrics import EvalMetric, MetricsTracker, StepMetric
from revllm.model import TransformerLM
from revllm.reversible import verify_reconstruction
from revllm.utils import ThetaRNG, calculate_mfu, get_device, set_seed


def build_optimizer(model: TransformerLM, weight_decay: float = 0.1, lr: float = 1e-3) -> AdamW:
    """Configure AdamW with fused=True (if cuda) and weight decay only on >=2-D weights."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay on 1D weights (norms) or embeddings
        if param.ndim >= 2 and "tok_embeddings" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()
    return AdamW(
        optim_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )


def get_lr_for_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return peak_lr * min_lr_ratio

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine_decay


def evaluate(
    model: TransformerLM,
    val_dataset: MemmapDataset,
    batch_size: int,
    device: torch.device,
    autocast_enabled: bool,
    thetas: list[float] | None = None,
    max_eval_batches: int | None = None,
) -> tuple[float, float]:
    """Run evaluation over the validation set, returning (mean_val_loss, perplexity)."""
    model.eval()
    loader = PrefetchDataLoader(val_dataset, batch_size=batch_size, device=device)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for b_idx, (batch_x, batch_y, _) in enumerate(loader):
            if max_eval_batches is not None and b_idx >= max_eval_batches:
                break
            _, loss = model(
                tokens=batch_x,
                targets=batch_y,
                thetas=thetas,
                autocast_enabled=autocast_enabled,
            )
            if loss is not None:
                total_loss += float(loss.item())
                count += 1
    loader.close()

    mean_loss = total_loss / max(1, count)
    ppl = math.exp(min(20.0, mean_loss))
    return mean_loss, ppl


def train_run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_cfg: RunConfig,
    train_data_path: str | Path,
    val_data_path: str | Path,
    local_output_dir: str | Path,
    drive_output_dir: str | Path | None = None,
    smoke_mode: bool = False,
) -> dict[str, Any]:
    """Execute complete training run with checkpointing, GPU monitoring, and evaluation."""
    set_seed(run_cfg.seed)
    device = get_device()
    use_cuda = device.type == "cuda"

    local_dir = Path(local_output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = local_dir / "final.json"

    # Idempotent return if already finished
    if final_json_path.exists():
        with open(final_json_path) as f:
            existing: dict[str, Any] = json.load(f)
            return existing

    ckpt_mgr = CheckpointManager(local_dir, drive_output_dir)
    metrics_tracker = MetricsTracker(local_dir / "metrics.jsonl")

    # Calculate step budgets
    batch_size = 2 if smoke_mode else run_cfg.batch_size
    tokens_per_step = batch_size * model_cfg.context_len
    target_tokens = 5000 if smoke_mode else train_cfg.training.total_tokens
    total_steps = math.ceil(target_tokens / tokens_per_step)

    peak_lr = min(
        train_cfg.lr_schedule.max_lr,
        train_cfg.lr_schedule.base_lr
        * math.sqrt(tokens_per_step / train_cfg.lr_schedule.ref_tokens_per_step),
    )
    warmup_steps = max(
        train_cfg.lr_schedule.min_warmup_steps,
        round(train_cfg.lr_schedule.warmup_ratio * total_steps),
    )

    eval_every = max(1, math.ceil(total_steps / train_cfg.training.eval_points))
    ckpt_every = max(1, math.ceil(total_steps / train_cfg.training.checkpoint_points))

    # Initialize model
    model = TransformerLM(model_cfg, arch=run_cfg.arch, h=run_cfg.h)
    model.to(device)

    # Optional torch.compile on individual blocks
    if run_cfg.compile and hasattr(torch, "compile") and use_cuda:
        try:
            for i in range(len(model.blocks)):
                model.blocks[i] = torch.compile(model.blocks[i])  # type: ignore
        except Exception as e:
            print(f"[Warning] torch.compile fallback: {e}")

    optimizer = build_optimizer(model, weight_decay=train_cfg.optimizer.weight_decay, lr=peak_lr)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_cuda, init_scale=train_cfg.training.grad_scaler_init
    )
    theta_rng = ThetaRNG(seed=run_cfg.seed + 17)

    # Load datasets
    train_dataset = MemmapDataset(train_data_path, context_len=model_cfg.context_len)
    val_dataset = MemmapDataset(val_data_path, context_len=model_cfg.context_len)

    # GPU monitor
    gpu_monitor = GPUMonitor(local_dir / "gpu_util.csv")
    if use_cuda:
        gpu_monitor.start()

    # Resume state
    start_step = 0
    tokens_seen = 0
    cum_train_time = 0.0
    resumed_steps: list[int] = []

    latest_ckpt = ckpt_mgr.get_latest_checkpoint()
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optim_state"])
        scaler.load_state_dict(ckpt_data["scaler_state"])
        theta_rng.set_state(ckpt_data["theta_rng_state"])
        start_step = ckpt_data["step"] + 1
        tokens_seen = ckpt_data["tokens_seen"]
        cum_train_time = ckpt_data["cum_train_time"]
        resumed_steps.append(start_step)

    # Reset memory stats
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    train_loader = PrefetchDataLoader(
        train_dataset, batch_size=batch_size, seed=run_cfg.seed, device=device
    )
    loader_iter = iter(train_loader)

    skipped_scaler_steps = 0
    model.train()

    t_loop_start = time.perf_counter()

    for step in range(start_step, total_steps):
        t_step_start = time.perf_counter()

        # Fetch batch
        try:
            batch_x, batch_y, data_wait_ms = next(loader_iter)
        except StopIteration:
            train_loader.close()
            train_loader = PrefetchDataLoader(
                train_dataset, batch_size=batch_size, seed=run_cfg.seed + step, device=device
            )
            loader_iter = iter(train_loader)
            batch_x, batch_y, data_wait_ms = next(loader_iter)

        # Update learning rate
        current_lr = get_lr_for_step(step, total_steps, warmup_steps, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Theta coefficients for midpoint_theta
        thetas: list[float] | None = None
        if run_cfg.arch == "midpoint_theta":
            thetas = theta_rng.sample_layer_theta(model_cfg.layers)

        optimizer.zero_grad(set_to_none=True)

        # Forward + backward with mixed precision
        with torch.amp.autocast(
            device_type="cuda" if use_cuda else "cpu", enabled=use_cuda, dtype=torch.float16
        ):
            _, loss = model(
                tokens=batch_x,
                targets=batch_y,
                thetas=thetas,
                chunk_size=train_cfg.training.loss_chunk_size,
                autocast_enabled=use_cuda,
            )

        if loss is None:
            raise RuntimeError("Loss returned None during training step")

        loss_val = float(loss.item())

        if use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.training.grad_clip)
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_scaler_steps += 1
        else:
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.training.grad_clip)
            )
            optimizer.step()

        tokens_seen += tokens_per_step
        step_duration = time.perf_counter() - t_step_start
        cum_train_time += step_duration
        tokens_per_sec = tokens_per_step / max(1e-6, step_duration)

        metrics_tracker.log_step(
            StepMetric(
                step=step,
                train_loss=loss_val,
                lr=current_lr,
                tokens_per_sec=tokens_per_sec,
                step_time_ms=step_duration * 1000.0,
                data_wait_ms=data_wait_ms,
                grad_norm=grad_norm,
                scaler_scale=float(scaler.get_scale()) if use_cuda else 1.0,
                tokens_seen=tokens_seen,
            )
        )

        # Periodic evaluation
        if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
            eval_loss, eval_ppl = evaluate(
                model=model,
                val_dataset=val_dataset,
                batch_size=batch_size,
                device=device,
                autocast_enabled=use_cuda,
                thetas=None if run_cfg.arch != "midpoint_theta" else [0.0] * model_cfg.layers,
                max_eval_batches=5 if smoke_mode else None,
            )
            sampled_a_loss: float | None = None
            if run_cfg.arch == "midpoint_theta":
                sampled_thetas = theta_rng.sample_layer_theta(model_cfg.layers)
                sampled_a_loss, _ = evaluate(
                    model=model,
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    device=device,
                    autocast_enabled=use_cuda,
                    thetas=sampled_thetas,
                    max_eval_batches=5 if smoke_mode else None,
                )
            metrics_tracker.log_eval(
                EvalMetric(
                    step=step,
                    val_loss=eval_loss,
                    val_ppl=eval_ppl,
                    val_loss_sampled_a=sampled_a_loss,
                    tokens_seen=tokens_seen,
                )
            )
            ckpt_mgr.sync_file_to_drive(local_dir / "metrics.jsonl")
            model.train()

        # Periodic checkpoint
        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            payload = {
                "step": step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "theta_rng_state": theta_rng.get_state(),
                "tokens_seen": tokens_seen,
                "cum_train_time": cum_train_time,
            }
            ckpt_mgr.save_checkpoint(step, payload)

    train_loader.close()

    total_wall_time = time.perf_counter() - t_loop_start
    mean_gpu_util, peak_smi_gb = gpu_monitor.stop() if use_cuda else (0.0, 0.0)

    # Final reconstruction check
    recon_max_abs: float | None = None
    recon_rel: float | None = None
    if run_cfg.arch in ["midpoint", "midpoint_theta", "leapfrog"]:
        dummy_x0 = torch.randn(2, model_cfg.context_len, model_cfg.hidden_size, device=device)
        dummy_freqs = model.freqs_cis[: model_cfg.context_len]
        recon_max_abs, recon_rel = verify_reconstruction(
            model.blocks,
            dummy_x0,
            dummy_freqs,
            arch=run_cfg.arch,
            h=model.h,
            thetas=[0.0] * model_cfg.layers if run_cfg.arch == "midpoint_theta" else None,
            autocast_enabled=use_cuda,
        )

    # Final validation evaluation
    final_val_loss, final_val_ppl = evaluate(
        model=model,
        val_dataset=val_dataset,
        batch_size=batch_size,
        device=device,
        autocast_enabled=use_cuda,
        thetas=[0.0] * model_cfg.layers if run_cfg.arch == "midpoint_theta" else None,
        max_eval_batches=5 if smoke_mode else None,
    )
    sampled_a_val: float | None = None
    if run_cfg.arch == "midpoint_theta":
        sampled_thetas = theta_rng.sample_layer_theta(model_cfg.layers)
        sampled_a_val, _ = evaluate(
            model=model,
            val_dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            autocast_enabled=use_cuda,
            thetas=sampled_thetas,
            max_eval_batches=5 if smoke_mode else None,
        )

    # Memory metrics
    peak_alloc_gb = float(torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0
    peak_res_gb = float(torch.cuda.max_memory_reserved() / 1e9) if use_cuda else 0.0

    # Throughput
    steady_tps = metrics_tracker.get_steady_state_throughput()
    e2e_tps = float(tokens_seen / max(1e-6, total_wall_time))
    device_name = torch.cuda.get_device_name() if use_cuda else "CPU"
    mfu = calculate_mfu(
        num_params=model.count_parameters(),
        tokens_per_sec=steady_tps,
        device_name=device_name,
        layers=model_cfg.layers,
        hidden_size=model_cfg.hidden_size,
        context_len=model_cfg.context_len,
    )

    result_payload: dict[str, Any] = {
        "run_id": run_cfg.run_id,
        "arch": run_cfg.arch,
        "h": run_cfg.h,
        "seed": run_cfg.seed,
        "params": model.count_parameters(),
        "micro_batch": batch_size,
        "tokens_per_step": tokens_per_step,
        "steps": total_steps,
        "tokens_seen": tokens_seen,
        "lr_peak": peak_lr,
        "compile": run_cfg.compile,
        "final_train_loss": metrics_tracker.get_final_train_loss(),
        "final_val_loss": final_val_loss,
        "final_val_loss_sampled_a": sampled_a_val,
        "val_ppl": final_val_ppl,
        "tokens_per_s_median": steady_tps,
        "tokens_per_s_end_to_end": e2e_tps,
        "train_time_s": cum_train_time,
        "peak_mem_allocated_gb": peak_alloc_gb,
        "peak_mem_reserved_gb": peak_res_gb,
        "peak_mem_nvidia_smi_gb": peak_smi_gb,
        "gpu_util_mean_pct": mean_gpu_util,
        "data_wait_mean_ms": metrics_tracker.get_mean_data_wait_ms(),
        "mfu_est_pct": mfu,
        "recon_err_max_abs": recon_max_abs,
        "recon_err_rel": recon_rel,
        "grad_scaler_skipped_steps": skipped_scaler_steps,
        "resumed_at_step": resumed_steps,
        "env": {
            "gpu": device_name,
            "vram_gb": float(torch.cuda.get_device_properties(0).total_memory / 1e9)
            if use_cuda
            else 0.0,
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.version.cuda else "cpu",
            "python": sys.version.split()[0],
        },
    }

    with open(final_json_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    ckpt_mgr.sync_file_to_drive(final_json_path)
    ckpt_mgr.sync_file_to_drive(local_dir / "gpu_util.csv")

    return result_payload
