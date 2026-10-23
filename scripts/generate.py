"""Sample greedy text generation from trained checkpoints for finding F10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from revllm.config import ModelConfig
from revllm.model import TransformerLM
from revllm.utils import get_device

FIXED_PROMPTS = [
    "The nature of intelligence is",
    "In modern science, the primary goal of",
    "Deep learning models achieve high efficiency when",
    "The historical development of mathematics demonstrates that",
    "To solve complex optimization problems, algorithms must",
]


def generate_greedy(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: torch.device | None = None,
) -> str:
    """Generate greedy completions token by token."""
    device = device or get_device()
    model.eval()

    input_ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Context length clamp
            ctx_tokens = tokens[:, -model.config.context_len :]
            y, _ = model(ctx_tokens)
            # Final token hidden state projected through tied embeddings
            last_hidden = y[:, -1, :]  # [1, D]
            logits = F.linear(last_hidden, model.tok_embeddings.weight)  # [1, V]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

    generated_ids = tokens[0].tolist()
    return tokenizer.decode(generated_ids)


def main(*args, **kwargs) -> None:
    parser = argparse.ArgumentParser(description="Generate text completions")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--output", default="results/generations.json", help="Output JSON path")

    argv = args[0] if args else kwargs.get("argv", None)
    if argv is None:
        if "ipykernel" in sys.modules or "google.colab" in sys.modules:
            argv = []
    parsed_args, _ = parser.parse_known_args(argv)

    device = get_device()
    model_cfg = ModelConfig()
    model = TransformerLM(model_cfg, arch=parsed_args.arch).to(device)

    ckpt_data = torch.load(parsed_args.ckpt, map_location=device)
    state_dict = ckpt_data["model_state"] if "model_state" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)

    tokenizer = Tokenizer.from_file(parsed_args.tokenizer)

    results: list[dict[str, str]] = []
    print(f"\n--- Generating greedy samples for arch: {parsed_args.arch} ---")
    for prompt in FIXED_PROMPTS:
        text = generate_greedy(model, tokenizer, prompt, max_new_tokens=64, device=device)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {text}")
        results.append({"prompt": prompt, "completion": text})

    out_path = Path(parsed_args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"arch": args.arch, "ckpt": args.ckpt, "samples": results}, f, indent=2)


if __name__ == "__main__":
    main()
