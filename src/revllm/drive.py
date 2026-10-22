"""Google Drive persistence, atomic checkpointing, and resume logic."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manages atomic checkpoints locally and mirrors them to Google Drive."""

    def __init__(self, local_run_dir: str | Path, drive_run_dir: str | Path | None = None) -> None:
        self.local_run_dir = Path(local_run_dir)
        self.drive_run_dir = Path(drive_run_dir) if drive_run_dir is not None else None

        self.local_ckpt_dir = self.local_run_dir / "ckpt"
        self.local_ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.drive_ckpt_dir: Path | None
        if self.drive_run_dir:
            self.drive_ckpt_dir = self.drive_run_dir / "ckpt"
            self.drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.drive_ckpt_dir = None

    def save_checkpoint(
        self,
        step: int,
        payload: dict[str, Any],
        keep_last: int = 2,
    ) -> Path:
        """Atomically save checkpoint to *.tmp, rename to step_XXXXXX.pt, update LATEST,

        and copy to Drive.
        """
        ckpt_filename = f"step_{step:06d}.pt"
        tmp_path = self.local_ckpt_dir / f"{ckpt_filename}.tmp"
        final_path = self.local_ckpt_dir / ckpt_filename

        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)

        # Update LATEST symlink or text pointer
        latest_file = self.local_ckpt_dir / "LATEST"
        with open(latest_file, "w") as f:
            f.write(ckpt_filename)

        # Copy to Drive if enabled
        if self.drive_ckpt_dir:
            try:
                drive_final = self.drive_ckpt_dir / ckpt_filename
                shutil.copy2(final_path, drive_final)
                drive_latest = self.drive_ckpt_dir / "LATEST"
                shutil.copy2(latest_file, drive_latest)
            except Exception as e:
                print(f"[Warning] Failed to sync checkpoint to Drive: {e}")

        # Clean old checkpoints keeping only last `keep_last`
        self._prune_checkpoints(self.local_ckpt_dir, keep_last)
        if self.drive_ckpt_dir:
            try:
                self._prune_checkpoints(self.drive_ckpt_dir, keep_last)
            except Exception:
                pass

        return final_path

    def _prune_checkpoints(self, directory: Path, keep: int) -> None:
        ckpts = sorted(directory.glob("step_*.pt"))
        if len(ckpts) > keep:
            for old_ckpt in ckpts[:-keep]:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass

    def get_latest_checkpoint(self) -> Path | None:
        """Find latest checkpoint from Drive or local."""
        # Check Drive first for resume across Colab runtime restarts
        if self.drive_ckpt_dir:
            drive_latest = self.drive_ckpt_dir / "LATEST"
            if drive_latest.exists():
                ckpt_name = drive_latest.read_text().strip()
                target = self.drive_ckpt_dir / ckpt_name
                if target.exists():
                    return target

        local_latest = self.local_ckpt_dir / "LATEST"
        if local_latest.exists():
            ckpt_name = local_latest.read_text().strip()
            target = self.local_ckpt_dir / ckpt_name
            if target.exists():
                return target

        return None

    def sync_file_to_drive(self, local_file: str | Path) -> None:
        """Sync a file (e.g. metrics.jsonl, config.yaml, final.json) to Drive."""
        if not self.drive_run_dir:
            return
        local_path = Path(local_file)
        if not local_path.exists():
            return
        rel_path = local_path.relative_to(self.local_run_dir)
        dest_path = self.drive_run_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(local_path, dest_path)
        except Exception as e:
            print(f"[Warning] Failed to sync {local_path} to Drive: {e}")
