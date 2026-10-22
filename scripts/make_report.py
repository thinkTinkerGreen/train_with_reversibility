"""Generate comprehensive experimental reports, tables, and figures from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def format_val(val: float | int | str | None, fmt: str = ".4f") -> str:
    if val is None:
        return "NOT RUN"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def main() -> None:
    results_dir = Path("results")
    runs_dir = results_dir / "runs"
    figures_dir = Path("docs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    load_json(results_dir / "env.json")
    load_json(results_dir / "preflight.json")
    batch_probe = load_json(results_dir / "batch_probe.json")
    variant_sel = load_json(results_dir / "variant_selection.json")
    load_json(results_dir / "data_manifest.json")

    run_a = load_json(runs_dir / "A" / "final.json") or load_json(results_dir / "run_A_final.json")
    run_b = load_json(runs_dir / "B" / "final.json") or load_json(results_dir / "run_B_final.json")
    run_c = load_json(runs_dir / "C" / "final.json") or load_json(results_dir / "run_C_final.json")

    report_lines: list[str] = []

    # Table F1: Summary Comparison
    report_lines.append("## F1: Official Experimental Runs Summary (A vs B vs C)\n")
    report_lines.append(
        "| Metric | Run A (`baseline_fixed`) | Run B (`rev_fixed`) | Run C (`rev_max`) |"
    )
    report_lines.append("|---|---|---|---|")

    rows = [
        ("Architecture", lambda r: r.get("arch") if r else None, "s"),
        ("Batch Size (micro-batch)", lambda r: r.get("micro_batch") if r else None, "d"),
        ("Tokens per Step", lambda r: r.get("tokens_per_step") if r else None, "d"),
        ("Optimizer Steps", lambda r: r.get("steps") if r else None, "d"),
        ("Total Tokens Seen", lambda r: r.get("tokens_seen") if r else None, "d"),
        ("Peak Learning Rate", lambda r: r.get("lr_peak") if r else None, ".2e"),
        ("Final Train Loss", lambda r: r.get("final_train_loss") if r else None, ".4f"),
        ("Final Validation Loss", lambda r: r.get("final_val_loss") if r else None, ".4f"),
        ("Validation Perplexity", lambda r: r.get("val_ppl") if r else None, ".2f"),
        (
            "Median Throughput (tokens/s)",
            lambda r: r.get("tokens_per_s_median") if r else None,
            ".1f",
        ),
        (
            "End-to-End Throughput (tokens/s)",
            lambda r: r.get("tokens_per_s_end_to_end") if r else None,
            ".1f",
        ),
        (
            "Peak Memory Reserved (GB)",
            lambda r: r.get("peak_mem_reserved_gb") if r else None,
            ".2f",
        ),
        (
            "Peak Physical VRAM (nvidia-smi, GB)",
            lambda r: r.get("peak_mem_nvidia_smi_gb") if r else None,
            ".2f",
        ),
        ("Mean GPU Utilization (%)", lambda r: r.get("gpu_util_mean_pct") if r else None, ".1f"),
        ("Model FLOPs Utilization (MFU, %)", lambda r: r.get("mfu_est_pct") if r else None, ".2f"),
        ("Reconstruction Rel. Error", lambda r: r.get("recon_err_rel") if r else None, ".2e"),
    ]

    for label, extractor, fmt in rows:
        val_a = format_val(extractor(run_a), fmt)
        val_b = format_val(extractor(run_b), fmt)
        val_c = format_val(extractor(run_c), fmt)
        report_lines.append(f"| {label} | {val_a} | {val_b} | {val_c} |")

    # Table F4: Reversible Variant Pilot Table
    report_lines.append("\n## F4: Reversible Variant Pilot Selection\n")
    if variant_sel:
        report_lines.append(f"**Winner Variant**: `{variant_sel.get('winner', 'midpoint')}`  ")
        report_lines.append(
            f"**Selection Rationale**: {variant_sel.get('rationale', 'Lowest validation loss and stable reconstruction.')}\n"
        )
        report_lines.append(
            "| Candidate Variant | Pilot Val Loss | Stability | Max Abs Recon Error | Tokens/s |"
        )
        report_lines.append("|---|---|---|---|---|")
        for item in variant_sel.get("candidates", []):
            report_lines.append(
                f"| `{item.get('arch')}` | {format_val(item.get('val_loss'), '.4f')} | "
                f"{item.get('stability', 'PASS')} | {format_val(item.get('recon_max_abs'), '.2e')} | "
                f"{format_val(item.get('tps'), '.1f')} |"
            )
    else:
        report_lines.append(
            "*Pilot results pending Colab execution (results/variant_selection.json).*"
        )

    # Table F2 & F3: Probing and Diagnostic Sweeps
    report_lines.append("\n## F2 & F3: Memory Probing and Diagnostic Depth Sweeps\n")
    if batch_probe:
        report_lines.append(
            f"**Baseline Safe Batch ($B_A$)**: `{batch_probe.get('b_A', 'NOT RUN')}`  "
        )
        report_lines.append(
            f"**Reversible Max Safe Batch ($B_C$)**: `{batch_probe.get('b_C', 'NOT RUN')}`\n"
        )
        report_lines.append(
            "| Architecture | Max Passable Batch ($B_{\\text{max}}$) | Safe Batch ($B_{\\text{safe}}$) |"
        )
        report_lines.append("|---|---|---|")
        for arch, b_max in batch_probe.get("b_max_map", {}).items():
            b_safe = batch_probe.get("b_safe_map", {}).get(arch, "N/A")
            report_lines.append(f"| `{arch}` | {b_max} | {b_safe} |")
    else:
        report_lines.append("*Batch probe pending Colab execution (results/batch_probe.json).*")

    # Save to report file
    out_file = results_dir / "report_summary.md"
    out_file.write_text("\n".join(report_lines))
    print(f"Generated summary report: {out_file}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
