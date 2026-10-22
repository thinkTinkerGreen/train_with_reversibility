# System Design & Architecture

## 1. Overview

`revllm` implements a modern ~20.85M parameter decoder-only Transformer with reversible residual dynamics. In standard Transformers (forward-Euler residual stream), activations at every layer must be saved in GPU VRAM during forward propagation to compute gradients during backpropagation:

$$p(l) = p(l-1) + f_l(p(l-1))$$

In contrast, reversible architectures invert the state transition backwards during the backward pass:

- **Midpoint**: $p(l-1) = p(l+1) - 2h \cdot f_l(p(l))$
- **Midpoint-$\theta$**: $p(l-1) = \frac{1}{a_l} \left[ p(l+1) - (1-a_l)p(l) - h \cdot f_l(p(l)) \right]$
- **Leapfrog**: $p(l-1) = 2p(l) - p(l+1) + h^2 \cdot f_l(p(l))$
- **Hamiltonian**: 2-stream split update

By recomputing intermediate states backward on the fly, memory consumption of layer activations scales as $O(1)$ with respect to depth $L$, instead of $O(L)$.

## 2. Model Architecture (20.85M)

- **Layers ($L$)**: 10
- **Hidden dimension ($d$)**: 384
- **Attention heads**: 6 (head dimension: 64)
- **MLP**: SwiGLU, hidden dimension 1024 (gate, up, down matrices without bias)
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-6}$), learnable scale
- **Positional Encoding**: RoPE (base 10,000) applied to Q and K
- **QK-Norm**: RMSNorm on Q and K per head
- **Vocabulary**: 8,192 (custom BPE trained on FineWeb-Edu)
- **Tied Embeddings**: Input token embedding shared with LM head projection
- **Parameter Count**:
  - Per layer: $4 \times 384^2 + 3 \times 384 \times 1024 + 2 \times 384 = 1,770,240$
  - 10 layers: $17,702,400$
  - Embedding: $8192 \times 384 = 3,145,728$
  - Final norm: $384$
  - **Total**: $20,848,512$ parameters (within [19.0M, 21.0M])

## 3. Chunked Cross-Entropy Loss

Standard cross-entropy creates a full logit tensor of shape $[B, T, V]$. For batch size 64 and context length 1024 with vocab 8192, this tensor alone requires:
$64 \times 1024 \times 8192 \times 2 \text{ bytes} \approx 1.07 \text{ GB}$ of memory.
To prevent the loss computation from masking the memory savings of reversibility, `revllm` projects the final hidden states and computes cross-entropy in chunks of 4096 tokens, discarding intermediate logits per chunk.

## 4. Reversible Autograd Implementation

`ReversibleTransformerFunction` inherits from `torch.autograd.Function`. During the forward pass:
1. Only the final states ($p(L)$ and $p(L-1)$) and the initial embedding state $x_0$ are retained.
2. During `backward()`, the layer transitions are walked in reverse from $l = L-1$ down to $0$.
3. At each layer $l$, $p(l-1)$ is mathematically reconstructed from $p(l+1)$ and $f_l(p(l))$.
4. Gradients through $f_l$ are computed with standard autograd using `torch.autograd.grad` with `p(l)` enabled for gradients.
5. In FP32 residual precision, this exact reversal avoids storing intermediate layer tensors in VRAM.
