# Metrics Specification & Estimation Formulas

This document outlines all metrics tracked across runs A, B, and C, as recorded in `results/final.json` and `results/metrics.jsonl`.

## 1. Throughput & Wall Time
- **`tokens_per_s_median`**: Median training throughput (tokens/s) during steady-state, excluding initial compile warmup steps ($\max(5, 0.05 \times \text{steps})$) and evaluation pauses.
- **`tokens_per_s_end_to_end`**: Total tokens seen divided by overall wall time (including eval, checkpointing, and I/O).
- **`train_time_s`**: Cumulative training time in seconds.

## 2. Memory Measurements
- **`peak_mem_allocated_gb`**: Peak memory actively allocated by PyTorch (`torch.cuda.max_memory_allocated() / 1e9`), reset after model setup and warm-up step.
- **`peak_mem_reserved_gb`**: Peak memory reserved by the PyTorch caching allocator (`torch.cuda.max_memory_reserved() / 1e9`).
- **`peak_mem_nvidia_smi_gb`**: Peak physical GPU memory measured via periodic `nvidia-smi` sampling.

## 3. Hardware & Operational Metrics
- **`gpu_util_mean_pct`**: Mean GPU core utilization (`nvidia-smi --query-gpu=utilization.gpu`), target $\ge 50\%$.
- **`data_wait_mean_ms`**: Mean time per step the GPU thread spent waiting for data prefetch from host memory.

## 4. Model FLOPs Utilization (MFU)
The theoretical FLOP count per token for a decoder Transformer is approximately:
$$\text{FLOPs/token} \approx 6N + 12 L H D_{\text{head}} T$$
Where:
- $N$ = non-embedding parameter count ($\approx 17,702,784$)
- $L = 10$, $H = 6$, $D_{\text{head}} = 64$, $T = 1024$

The Model FLOPs Utilization (MFU) is:
$$\text{MFU} = \frac{\text{FLOPs/token} \times \text{tokens\_per\_s\_median}}{\text{Peak Device FP16 TFLOPs} \times 10^{12}}$$
On NVIDIA Tesla T4 (FP16 Tensor Core peak = 65 TFLOPs):
$$\text{Peak T4} = 65.0 \times 10^{12} \text{ FLOP/s}$$

## 5. Reconstruction & Gradient Equivalence
- **`recon_err_max_abs`**: Maximum absolute difference between reconstructed activation state and forward-saved activation:
  $$\max |p_{\text{recon}} - p_{\text{forward}}|$$
- **`recon_err_rel`**: Relative L2 reconstruction error:
  $$\frac{\|p_{\text{recon}} - p_{\text{forward}}\|_2}{\|p_{\text{forward}}\|_2 + 10^{-8}}$$
- **Gradient Cosine Similarity**:
  $$\text{Cosine}(\nabla W_{\text{rev}}, \nabla W_{\text{autograd}}) \ge 0.9999$$
