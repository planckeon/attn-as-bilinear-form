+++
title = "Attention Mechanism"
weight = 2
+++

## The Attention Formula

The scaled dot-product attention from "Attention Is All You Need":

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Let's decompose this in index notation.

## Step-by-Step Breakdown

### Step 1: Score Computation

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

- $Q^{ia}$: Query at position $i$, feature dimension $a$
- $K^{ja}$: Key at position $j$, feature dimension $a$
- Contraction over $a$ gives similarity between query $i$ and key $j$

### Step 2: Softmax Normalization

$$A^{ij} = \frac{\exp(S^{ij})}{\sum_{j'} \exp(S^{ij'})}$$

- Normalizes along the key dimension for each query
- $A^{ij}$ is the attention weight from query $i$ to key $j$
- Each row sums to 1: $\sum_j A^{ij} = 1$

### Step 3: Value Aggregation

$$O^{ib} = A^{ij} V^{jb}$$

- $V^{jb}$: Value at position $j$, feature dimension $b$
- Weighted sum of values based on attention weights
- Output $O^{ib}$ has same shape as queries: $(n_q, d_v)$

## Tensor Shapes

| Tensor | Shape | Indices |
|--------|-------|---------|
| Queries $Q$ | $(n_q, d_k)$ | $Q^{ia}$ |
| Keys $K$ | $(n_k, d_k)$ | $K^{ja}$ |
| Values $V$ | $(n_k, d_v)$ | $V^{jb}$ |
| Scores $S$ | $(n_q, n_k)$ | $S^{ij}$ |
| Attention $A$ | $(n_q, n_k)$ | $A^{ij}$ |
| Output $O$ | $(n_q, d_v)$ | $O^{ib}$ |

## Self-Attention vs Cross-Attention

**Self-attention**: $Q$, $K$, $V$ all come from the same sequence
- $n_q = n_k$, typically denoted just $n$

**Cross-attention**: $Q$ from one sequence, $K$, $V$ from another
- $n_q \neq n_k$ in general
- Example: decoder attending to encoder outputs

## Causal (Masked) Attention

For autoregressive models, we mask future positions:

$$S^{ij}&#95;{\text{masked}} = \begin{cases}
S^{ij} & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}$$

This ensures each position can only attend to earlier positions.

## Multi-Head Attention

Introduce a head index $h$:

$$Q^{hia} = X^{ib} W_Q^{hba}$$
$$K^{hja} = X^{jb} W_K^{hba}$$
$$V^{hjb} = X^{jc} W_V^{hcb}$$

Each head computes attention independently:

$$O^{hib} = A^{hij} V^{hjb}$$

Then concatenate and project:

$$\text{Output}^{ic} = O^{hib} W_O^{hbc}$$

## Code Example

```python
from attn_tensors import scaled_dot_product_attention
from attn_tensors.multihead import multihead_attention

# Single-head attention
Q = jnp.randn(10, 64)  # 10 queries, 64 dims
K = jnp.randn(20, 64)  # 20 keys
V = jnp.randn(20, 64)  # 20 values

output = scaled_dot_product_attention(Q, K, V)
# output.shape = (10, 64)

# Get attention weights too
output, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
# weights.shape = (10, 20)

# Multi-head attention
output = multihead_attention(Q, K, V, num_heads=8)
```
