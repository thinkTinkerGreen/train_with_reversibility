## F1: Official Experimental Runs Summary (A vs B vs C)

| Metric | Run A (`baseline_fixed`) | Run B (`rev_fixed`) | Run C (`rev_max`) |
|---|---|---|---|
| Architecture | baseline | midpoint | midpoint |
| Batch Size (micro-batch) | 32 | 32 | 64 |
| Tokens per Step | 32768 | 32768 | 65536 |
| Optimizer Steps | 1526 | 1526 | 763 |
| Total Tokens Seen | 50003968 | 50003968 | 50003968 |
| Peak Learning Rate | 7.07e-04 | 7.07e-04 | 1.00e-03 |
| Final Train Loss | 3.8989 | 3.9215 | 4.3315 |
| Final Validation Loss | 3.9160 | 3.9385 | 4.3398 |
| Validation Perplexity | 50.20 | 51.34 | 76.69 |
| Median Throughput (tokens/s) | 54670.4 | 47150.3 | 39457.3 |
| End-to-End Throughput (tokens/s) | 39740.1 | 36130.2 | 30765.4 |
| Peak Memory Reserved (GB) | 6.16 | 1.94 | 3.15 |
| Peak Physical VRAM (nvidia-smi, GB) | 5.92 | 1.99 | 3.12 |
| Mean GPU Utilization (%) | 96.1 | 96.0 | 97.3 |
| Model FLOPs Utilization (MFU, %) | 14.49 | 12.50 | 10.46 |
| Reconstruction Rel. Error | NOT RUN | 7.40e-05 | 6.38e-05 |

## F4: Reversible Variant Pilot Selection

**Winner Variant**: `midpoint`  
**Selection Rationale**: Surviving candidate with lowest validation loss (5.1661)

| Candidate Variant | Pilot Val Loss | Stability | Max Abs Recon Error | Tokens/s |
|---|---|---|---|---|
| `midpoint` | 5.1661 | PASS | 2.33e-04 | 27845.6 |
| `midpoint_theta` | 6.5838 | PASS | 0.00e+00 | 27074.4 |
| `leapfrog` | 5.0095 | PASS | 1.59e-03 | 27031.0 |

## F2 & F3: Memory Probing and Diagnostic Depth Sweeps

*Batch probe pending Colab execution (results/batch_probe.json).*