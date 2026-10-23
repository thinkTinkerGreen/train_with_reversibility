"""Build FineWeb-Edu BPE tokenizer and binary token datasets (train.bin, val.bin)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def compute_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def train_bpe_tokenizer(
    text_iterator,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    output_path: str | Path = "data/tokenizer.json",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and tokenizer")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--manifest-path", default="results/data_manifest.json", help="Manifest output"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Generate tiny synthetic data for testing"
    )

    if argv is None:
        if "ipykernel" in sys.modules or "google.colab" in sys.modules:
            argv = []
    args, _ = parser.parse_known_args(argv)

    smoke = args.smoke or (os.environ.get("REVLLM_SMOKE", "0") == "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_size = cfg["tokenizer"]["vocab_size"]
    train_tokens_target = 100_000 if smoke else cfg["splits"]["train_tokens"]
    val_tokens_target = 20_000 if smoke else cfg["splits"]["val_tokens"]

    tokenizer_path = output_dir / "tokenizer.json"
    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"

    if smoke:
        print("[Smoke Mode] Generating synthetic token dataset for fast verification...")
        # Train tiny tokenizer
        sample_texts = ["The quick brown fox jumps over the lazy dog. " * 50 for _ in range(200)]
        tokenizer = train_bpe_tokenizer(
            sample_texts, vocab_size=vocab_size, output_path=tokenizer_path
        )

        train_data = np.random.randint(0, vocab_size, size=train_tokens_target, dtype=np.uint16)
        val_data = np.random.randint(0, vocab_size, size=val_tokens_target, dtype=np.uint16)
        train_data.tofile(train_bin_path)
        val_data.tofile(val_bin_path)
    else:
        print(
            f"Loading FineWeb-Edu dataset ({cfg['dataset']['name']}, {cfg['dataset']['config']})..."
        )
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            cfg["dataset"]["name"],
            name=cfg["dataset"]["config"],
            split="train",
            streaming=True,
        )

        # 1. Collect texts for tokenizer training
        print(f"Collecting first {cfg['tokenizer']['train_docs']} docs for BPE training...")
        train_docs: list[str] = []
        for _i, row in enumerate(ds):
            train_docs.append(row[cfg["dataset"]["text_field"]])
            if len(train_docs) >= cfg["tokenizer"]["train_docs"]:
                break

        print("Training BPE tokenizer...")
        tokenizer = train_bpe_tokenizer(
            train_docs,
            vocab_size=vocab_size,
            min_frequency=cfg["tokenizer"]["min_frequency"],
            output_path=tokenizer_path,
        )

        eot_id = tokenizer.token_to_id("<|endoftext|>")

        # 2. Tokenize into train.bin and val.bin
        print("Tokenizing train split...")
        train_tokens: list[int] = []
        val_tokens: list[int] = []

        for row in ds:
            text = row[cfg["dataset"]["text_field"]]
            ids = tokenizer.encode(text).ids
            ids.append(eot_id)

            if len(train_tokens) < train_tokens_target:
                train_tokens.extend(ids)
            elif len(val_tokens) < val_tokens_target:
                val_tokens.extend(ids)
            else:
                break

        train_arr = np.array(train_tokens[:train_tokens_target], dtype=np.uint16)
        val_arr = np.array(val_tokens[:val_tokens_target], dtype=np.uint16)
        train_arr.tofile(train_bin_path)
        val_arr.tofile(val_bin_path)

    manifest = {
        "dataset_name": cfg["dataset"]["name"],
        "dataset_config": cfg["dataset"]["config"],
        "dataset_revision": cfg["dataset"]["revision"],
        "tokenizer_sha256": compute_sha256(tokenizer_path),
        "train_bin": {
            "path": str(train_bin_path),
            "tokens": int(train_tokens_target),
            "sha256": compute_sha256(train_bin_path),
        },
        "val_bin": {
            "path": str(val_bin_path),
            "tokens": int(val_tokens_target),
            "sha256": compute_sha256(val_bin_path),
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset successfully built! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
