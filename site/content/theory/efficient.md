+++
title = "Efficient Attention"
weight = 7
+++

## The Quadratic Problem

Standard attention has:
- **Time complexity:** $O(n^2 d)$
- **Memory complexity:** $O(n^2)$ (storing attention matrix)

For long sequences ($n > 10000$), this becomes prohibitive.

## Sparse Attention Patterns

### Local Window Attention

Only attend to tokens within a fixed window:

$$M^{ij} = \begin{cases}
0 & \text{if } |i - j| \leq w \\
-\infty & \text{otherwise}
\end{cases}$$

**Complexity:** $O(nwd)$ time, $O(nw)$ memory

### Strided Attention

Attend to every $k$-th token:

$$M^{ij} = \begin{cases}
0 & \text{if } j \mod k = 0 \\
-\infty & \text{otherwise}
\end{cases}$$

Combines with local attention for global receptive field.

### Block-Sparse Attention

Divide sequence into blocks, attend within and across select blocks:

```
Block pattern:
[■ ■ □ □ ■]    ■ = attend
[■ ■ ■ □ □]    □ = mask
[□ ■ ■ ■ □]
[□ □ ■ ■ ■]
[■ □ □ ■ ■]
```

### Longformer Pattern

Combine local attention with global tokens:

$$A^{ij} = \text{local}(i, j) + \text{global}(i) + \text{global}(j)$$

- **Local:** Sliding window of size $w$
- **Global:** Selected tokens (for example, [CLS]) attend to/from all positions

### BigBird Pattern

Longformer + random attention:

$$A = A_{local} + A_{global} + A_{random}$$

Random edges ensure any two tokens connect with high probability.

## Sparse Attention in Index Notation

For a mask $M^{ij} \in \{0, -\infty\}$:

$$S^{ij}_{masked} = S^{ij} + M^{ij}$$

The softmax naturally zeros out masked positions:

$$A^{ij} = \frac{\exp(S^{ij} + M^{ij})}{\sum_{j'} \exp(S^{ij'} + M^{ij'})}$$

When $M^{ij} = -\infty$, the term vanishes.

## Flash Attention

### The Memory Bottleneck

Standard attention:
1. Compute $S = QK^T/\sqrt{d_k}$ → Store $O(n^2)$
2. Compute $A = \text{softmax}(S)$ → Store $O(n^2)$
3. Compute $O = AV$ → Store $O(nd)$

The $n^2$ intermediate storage limits sequence length.

### Key Insight: Online Softmax

Softmax can be computed incrementally:

$$\text{softmax}(x_1, \ldots, x_n) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Online algorithm:**
1. Maintain running max $m$ and sum of exponentials $\ell$
2. Process blocks, updating $m$ and $\ell$
3. Correct for max changes using: $e^{x - m_{new}} = e^{x - m_{old}} \cdot e^{m_{old} - m_{new}}$

### Block-wise Computation

Divide $Q, K, V$ into blocks of size $B$:

$$Q = [Q_1, Q_2, \ldots, Q_{n/B}]$$

For each query block $Q_i$:
1. Initialize output $O_i = 0$, log-sum-exp $\ell_i = -\infty$
2. For each key-value block $(K_j, V_j)$:
   - Compute block scores $S_{ij} = Q_i K_j^T / \sqrt{d_k}$
   - Update running softmax statistics
   - Accumulate contribution to $O_i$

### Algorithm Pseudocode

```python
def flash_attention(Q, K, V, block_size=64):
    n, d = Q.shape
    O = zeros_like(Q)
    
    for i in range(0, n, block_size):
        Q_block = Q[i:i+block_size]
        m_i = full(-inf, block_size)  # running max
        l_i = zeros(block_size)        # running sum
        O_i = zeros((block_size, d))
        
        for j in range(0, n, block_size):
            K_block = K[j:j+block_size]
            V_block = V[j:j+block_size]
            
            # Compute block attention scores
            S_ij = Q_block @ K_block.T / sqrt(d)
            
            # Update running max
            m_ij = max(S_ij, axis=-1)
            m_new = maximum(m_i, m_ij)
            
            # Rescale previous contributions
            alpha = exp(m_i - m_new)
            beta = exp(m_ij - m_new)
            
            # Update running sum and output
            l_i = alpha * l_i + beta * sum(exp(S_ij - m_ij), axis=-1)
            O_i = alpha * O_i + beta * exp(S_ij - m_ij) @ V_block
            m_i = m_new
        
        O[i:i+block_size] = O_i / l_i
    
    return O
```

### Backward Pass

Key insight: avoid storing attention matrix. Recompute during backward.

**Forward**: store only $O$, $\ell$ (log-sum-exp), $m$ (max)

**Backward**: recompute $S$ and $A$ block-by-block while computing gradients.

### Gradient Computation

Standard gradient through softmax:

$$\frac{\partial L}{\partial S^{ij}} = A^{ij} \left( \frac{\partial L}{\partial A^{ij}} - \sum_{j'} A^{ij'} \frac{\partial L}{\partial A^{ij'}} \right)$$

Define $D^i = \sum_j A^{ij} \frac{\partial L}{\partial A^{ij}}$ (computed as $O \odot dO$ summed).

Then:

$$\frac{\partial L}{\partial S^{ij}} = A^{ij} (dA^{ij} - D^i)$$

Block-wise:

$$dS_{ij} = A_{ij} \odot (dA_{ij} - D_i)$$
$$dQ_i = \sum_j dS_{ij} K_j / \sqrt{d_k}$$
$$dK_j = \sum_i dS_{ij}^T Q_i / \sqrt{d_k}$$
$$dV_j = \sum_i A_{ij}^T dO_i$$

### Complexity Analysis

| Method | Time | Memory | IO |
|--------|------|--------|-----|
| Standard | $O(n^2 d)$ | $O(n^2)$ | $O(n^2 + nd)$ |
| Flash | $O(n^2 d)$ | $O(n)$ | $O(n^2 d / M)$ |

where $M$ is SRAM size.

Flash Attention is **IO-aware**: minimizes data movement between GPU SRAM and HBM.

## Linear Attention

### Kernel Trick

Standard attention:

$$O_i = \frac{\sum_j \exp(q_i^T k_j) v_j}{\sum_j \exp(q_i^T k_j)}$$

If $\exp(q^T k) \approx \phi(q)^T \phi(k)$ approximates:

$$O_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j}{\sum_j \phi(q_i)^T \phi(k_j)}$$

$$= \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

### Complexity

Precompute:
- $KV = \sum_j \phi(k_j) v_j^T$—shape $(d_\phi, d_v)$
- $K_{sum} = \sum_j \phi(k_j)$—shape $(d_\phi,)$

Then each query costs $O(d_\phi d)$ instead of $O(nd)$.

**Total**: $O(n d_\phi d)$—linear in sequence length

### Feature Maps

Common choices for $\phi$:

1. **Random Fourier Features:**
   $$\phi(x) = \exp(Wx) / \sqrt{d}$$
   
2. **ELU + 1:**
   $$\phi(x) = \text{ELU}(x) + 1$$
   
3. **Positive Random Features (Performers):**
   $$\phi(x) = \exp\left(x^T \omega - \frac{\|x\|^2}{2}\right)$$

### Causal Linear Attention

For autoregressive models, accumulate incrementally:

$$KV_i = KV_{i-1} + \phi(k_i) v_i^T$$
$$O_i = \frac{\phi(q_i)^T KV_i}{\phi(q_i)^T K_{sum,i}}$$

This is an RNN with hidden state $(KV, K_{sum})$.

## Multi-Query and Grouped-Query Attention

### Multi-Query Attention (MQA)

Share keys and values across all heads:

$$Q^{hia}: \text{per-head}$$
$$K^{ja}, V^{jb}: \text{shared across heads}$$

**Savings**: parameters and KV-cache reduced by factor of $H$.

### Grouped-Query Attention (GQA)

Compromise: group heads, share K/V within groups.

With $G$ groups and $H$ heads:
- Each group has $H/G$ heads
- Each group shares one K and one V

**GQA-1 = MQA, GQA-H = MHA**

## Code Example

```python
from attn_tensors.efficient import (
    flash_attention,
    linear_attention,
    local_attention,
    create_local_mask,
    create_strided_mask,
)

# Standard attention (for comparison)
O_standard = scaled_dot_product_attention(Q, K, V)

# Flash attention
O_flash = flash_attention(Q, K, V, block_size=64)

# Linear attention with ELU features
O_linear = linear_attention(Q, K, V, feature_map='elu')

# Local window attention
mask = create_local_mask(seq_len, window_size=128)
O_local = scaled_dot_product_attention(Q, K, V, mask=mask)
```

## When to Use What

| Method | Best For | Tradeoff |
|--------|----------|----------|
| Standard | Short sequences (<512) | Simple, exact |
| Flash | Medium sequences (512-8K) | Exact, memory efficient |
| Sparse | Long sequences (8K+) | Approximate, task-dependent |
| Linear | Long sequences | Approximate, loses expressivity |
| MQA/GQA | Inference | Reduced KV-cache |
