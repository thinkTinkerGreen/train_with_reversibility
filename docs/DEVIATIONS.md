# Deviations and Engineering Assumptions

This document lists all design choices, paper ambiguities resolved, and engineering deviations relative to Gal et al., *Reversing LLMs for Efficient Training and Fine-Tuning* (arXiv:2512.02056).

| ID | Topic | Paper Description | Implementation in revllm | Rationale / Resolution |
|---|---|---|---|---|
| D1 | Midpoint-theta parameter $a_l$ | Eq. 3.7 specifies random coefficient drawn from two intervals around $+1$ and $-1$. | For each layer $l$ and step, draw sign $s \in \{+1, -1\}$ with $p=0.5$, then $a_l = s + 0.5 \cdot U(-1, 1)$. Seeded via dedicated RNG stream `rev_theta`. | Makes recurrence strictly deterministic and reproducible. |
| D2 | Evaluation of Midpoint-theta | Mentions expectation behaves like forward Euler. | Set $a_l = 0$ at evaluation time; optionally report sampled $a$ evaluation. | Avoids evaluation noise and preserves expected deterministic inference. |
| D3 | Step size $h$ | Left open or varied in paper. | Fixed to $h=0.5$ for midpoint (effective step $2h=1$), $h=1.0$ for leapfrog and midpoint_theta. | Matches the baseline residual stream scale ($1 \cdot f_l(p)$) directly. |
| D4 | Residual State Initialization | $p(-1)$ boundary condition not specified. | $p(-1) = p(0) = x_0$ where $x_0 = \text{Embedding}(\text{tokens})$. | Natural 2-step initialization from embedding output. |
| D5 | Chunked Cross-Entropy | Often full logits calculated in typical training scripts. | Chunked cross-entropy with chunk size 4096 tokens wrapped in checkpointing. | Prevents 8192-vocab logit tensor ($B \times T \times V$) from dominating VRAM and masking reversible memory savings. |
| D6 | State Precision | Paper tests FP32/BF16 on 24–80GB GPUs. | FP32 residual stream states; block compute under FP16 autocast with GradScaler. | Google Colab free tier GPU (NVIDIA T4) lacks native BF16 hardware support; FP32 states prevent numerical drift during reverse reconstruction. |
| D7 | Reversible Candidate Code Diff | Checked arXiv:2512.02056 (checked Oct 2026). | No official standalone PyTorch package repository is linked yet in the preprint. | Implemented clean, self-contained, mathematically verified PyTorch autograd Function. |
