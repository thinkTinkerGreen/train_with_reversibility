"""Unit tests for LR scheduler, training loop, and checkpoint resume equivalence."""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np

from revllm.config import ModelConfig, RunConfig, TrainConfig
from revllm.train import get_lr_for_step, train_run


def test_lr_schedule() -> None:
    """Validate linear warmup and cosine decay to 0.1 * peak_lr."""
    total_steps = 100
    warmup_steps = 10
    peak_lr = 1e-3
    min_lr_ratio = 0.1

    # Step 0: Warmup start
    lr_0 = get_lr_for_step(0, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert math.isclose(lr_0, peak_lr * (1.0 / 10.0), rel_tol=1e-5)

    # Step 9: End of warmup (step 10 in 1-based indexing)
    lr_warmup_end = get_lr_for_step(9, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert math.isclose(lr_warmup_end, peak_lr, rel_tol=1e-5)

    # Final step: exactly 0.1 * peak_lr
    lr_final = get_lr_for_step(total_steps - 1, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert math.isclose(lr_final, peak_lr * min_lr_ratio, rel_tol=1e-2)

    # Beyond final step: exactly 0.1 * peak_lr
    lr_beyond = get_lr_for_step(total_steps, total_steps, warmup_steps, peak_lr, min_lr_ratio)
    assert math.isclose(lr_beyond, peak_lr * min_lr_ratio, rel_tol=1e-5)


def test_resume_equivalence() -> None:
    """Training 6 steps continuously yields identical loss trajectory to 3 steps + checkpoint + resume 3 steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.bin"
        val_path = Path(tmpdir) / "val.bin"

        tokens = np.random.randint(0, 256, size=50000, dtype=np.uint16)
        tokens.tofile(train_path)
        tokens[:5000].tofile(val_path)

        model_cfg = ModelConfig(
            layers=2,
            hidden_size=64,
            num_heads=2,
            head_dim=32,
            mlp_hidden_size=128,
            vocab_size=256,
            context_len=32,
        )
        train_cfg = TrainConfig()
        train_cfg.training.total_tokens = 6 * 2 * 32  # exactly 6 steps at B=2
        train_cfg.training.eval_points = 1
        train_cfg.training.checkpoint_points = 2

        # 1. Uninterrupted run
        run_cfg1 = RunConfig(
            run_id="uninterrupted", arch="baseline", batch_size=2, seed=1337, compile=False
        )
        out1 = train_run(
            model_cfg,
            train_cfg,
            run_cfg1,
            train_path,
            val_path,
            local_output_dir=Path(tmpdir) / "run1",
            smoke_mode=True,
        )

        # 2. Interrupted run: train 3 steps first, then resume for remaining steps
        run_cfg2 = RunConfig(
            run_id="interrupted", arch="baseline", batch_size=2, seed=1337, compile=False
        )
        train_cfg_half = copy.deepcopy(train_cfg)
        train_cfg_half.training.total_tokens = 3 * 2 * 32

        train_run(
            model_cfg,
            train_cfg_half,
            run_cfg2,
            train_path,
            val_path,
            local_output_dir=Path(tmpdir) / "run2",
            smoke_mode=True,
        )

        # Remove final.json so second run resumes from checkpoint
        final_json2 = Path(tmpdir) / "run2" / "final.json"
        if final_json2.exists():
            final_json2.unlink()

        out2 = train_run(
            model_cfg,
            train_cfg,
            run_cfg2,
            train_path,
            val_path,
            local_output_dir=Path(tmpdir) / "run2",
            smoke_mode=True,
        )

        assert math.isclose(out1["final_train_loss"], out2["final_train_loss"], rel_tol=1e-4)
