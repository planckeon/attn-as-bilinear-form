+++
title = "Positional Encodings"
weight = 6
+++

## Why Positional Information?

Attention is permutation-equivariant: swapping input positions swaps outputs identically. But language has order! "Dog bites man" ≠ "Man bites dog".

We need to inject positional information.

## Absolute Positional Encodings

### Sinusoidal (Original Transformer)

Vaswani et al. (2017) used fixed sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Properties:**
- Each dimension oscillates at different frequency
- Can extrapolate to longer sequences (in theory)
- $PE_{pos+k}$ can be represented as linear function of $PE_{pos}$

### Learned Positional Embeddings

Simply learn a lookup table:

$$X^{ia}&#95;{\text{input}} = X^{ia}&#95;{\text{token}} + P^{ia}$$

where $P \in \mathbb{R}^{L_{max} \times d}$ is learned.

**Tradeoff:**
- More flexible than sinusoidal
- Cannot extrapolate beyond training length

## Relative Positional Encodings

### The Problem with Absolute

Absolute encodings conflate content with position. The model must learn that "word at position 5" attending to "word at position 3" is similar to "position 10 attending to position 8".

Relative encodings directly encode the offset.

### T5-Style Relative Bias

Add a learned bias based on relative position:

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja} + b_{i-j}$$

where $b_k$ is a learned bias for relative position $k$.

**Bucketing:** T5 uses logarithmic bucketing for large offsets:
- Exact positions for $|k| \leq 8$
- Bucketed for larger offsets

### Transformer-XL Style

Decompose attention into content and position terms:

$$S^{ij} = \underbrace{Q^{ia} K^{ja}}&#95;{\text{content-content}} + \underbrace{Q^{ia} R^{(i-j)a}}&#95;{\text{content-position}} + \underbrace{u^a K^{ja}}&#95;{\text{global content}} + \underbrace{v^a R^{(i-j)a}}&#95;{\text{global position}}$$

where:
- $R^{ka}$: Relative position embeddings
- $u^a, v^a$: Learned global query vectors

### ALiBi (Attention with Linear Biases)

Press et al. (2022) proposed a simple approach:

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja} - m \cdot |i - j|$$

where $m$ is a head-specific slope.

**Key insight:** No learned positional parameters! Just a linear penalty for distance.

**Slopes:** Different heads use different slopes: $m_h = 2^{-8h/H}$

| Head | Slope | Effect |
|------|-------|--------|
| 1 | Large | Very local attention |
| H | Small | Global attention |

**Extrapolation:** ALiBi extrapolates well to longer sequences than trained on.

## Rotary Position Embeddings (RoPE)

### Core Idea

Su et al. (2021): Encode position by rotating the query/key vectors.

For position $m$, rotate by angle $m\theta$:

$$f(x, m) = R_m x$$

where $R_m$ is a rotation matrix.

### Complex Number Formulation

For 2D, think of $(q_1, q_2)$ as complex number $q_1 + iq_2$:

$$f(q, m) = q \cdot e^{im\theta} = (q_1 + iq_2)(\cos m\theta + i\sin m\theta)$$

Expanding:
$$\text{Re}[f(q,m)] = q_1 \cos m\theta - q_2 \sin m\theta$$
$$\text{Im}[f(q,m)] = q_1 \sin m\theta + q_2 \cos m\theta$$

### Block-Diagonal Rotation

For $d$-dimensional vectors, apply 2D rotations to pairs:

$$R_m = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & & & \\\\
\sin m\theta_1 & \cos m\theta_1 & & & \\\\
& & \cos m\theta_2 & -\sin m\theta_2 & \\\\
& & \sin m\theta_2 & \cos m\theta_2 & \\\\
& & & & \ddots
\end{pmatrix}$$

Different frequencies: $\theta_i = 10000^{-2i/d}$

### Key Property

The dot product depends only on relative position:

$$(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k$$

This is because rotations compose: $R_m^T R_n = R_{n-m}$.

### In Index Notation

For query $Q^{ia}$ at position $i$:

$$\tilde{Q}^{ia} = R^{ab}(i) Q^{ib}$$

Score computation:

$$S^{ij} = \frac{1}{\sqrt{d_k}} \tilde{Q}^{ia} \tilde{K}^{ja} = \frac{1}{\sqrt{d_k}} R^{ab}(i) Q^{ib} R^{ac}(j) K^{jc}$$

Using $R^T(i) R(j) = R(j-i)$:

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} R^{ab}(j-i) K^{jb}$$

### Efficient Implementation

No explicit matrix multiplication needed! Use:

$$\tilde{q} = q \odot \cos(m\theta) + \text{rotate\_half}(q) \odot \sin(m\theta)$$

where `rotate_half` swaps and negates pairs:
```
rotate_half([q1, q2, q3, q4, ...]) = [-q2, q1, -q4, q3, ...]
```

### Gradients for RoPE

Since RoPE is just multiplication by rotation matrices:

$$\frac{\partial L}{\partial Q^{ia}} = R^{ab}(i) \frac{\partial L}{\partial \tilde{Q}^{ib}}$$

The rotation is its own transpose (orthogonal), so gradients just rotate back.

## Comparison Table

| Method | Learnable | Extrapolation | Relative | Memory |
|--------|-----------|---------------|----------|--------|
| Sinusoidal | No | Moderate | No | O(L·d) |
| Learned | Yes | Poor | No | O(L·d) |
| T5 Bias | Yes | Moderate | Yes | O(L²) |
| ALiBi | No | Excellent | Yes | O(1) |
| RoPE | No | Good | Yes | O(L·d) |

## Code Example

```python
from attn_tensors.positional import (
    sinusoidal_encoding,
    rotary_embedding,
    apply_rope,
    alibi_bias,
)

seq_len, d_model = 100, 64

# Sinusoidal
pos_enc = sinusoidal_encoding(seq_len, d_model)
X = X + pos_enc

# RoPE
Q_rotated = apply_rope(Q, positions)
K_rotated = apply_rope(K, positions)
scores = Q_rotated @ K_rotated.T / jnp.sqrt(d_k)

# ALiBi
scores = Q @ K.T / jnp.sqrt(d_k)
scores = scores + alibi_bias(seq_len, num_heads)
```

## RoPE: Worked Example

**Setup:** $d = 4$, position $m = 2$, $\theta = [1.0, 0.1]$

**Query:** $q = [1, 0, 1, 0]$

**Rotation angles:** $m\theta = [2.0, 0.2]$

**Apply rotation to pairs:**

Pair 1: $(q_1, q_2) = (1, 0)$
$$\tilde{q}_1 = 1 \cdot \cos(2) - 0 \cdot \sin(2) = \cos(2) \approx -0.42$$
$$\tilde{q}_2 = 1 \cdot \sin(2) + 0 \cdot \cos(2) = \sin(2) \approx 0.91$$

Pair 2: $(q_3, q_4) = (1, 0)$
$$\tilde{q}_3 = 1 \cdot \cos(0.2) - 0 \cdot \sin(0.2) \approx 0.98$$
$$\tilde{q}_4 = 1 \cdot \sin(0.2) + 0 \cdot \cos(0.2) \approx 0.20$$

**Result:** $\tilde{q} \approx [-0.42, 0.91, 0.98, 0.20]$
