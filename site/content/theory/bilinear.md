+++
title = "Bilinear Forms and Metric Tensors"
weight = 1
+++

## Mathematical Foundation

A **bilinear form** is a map $B\colon V \times W \to \mathbb{R}$ that is linear in both arguments.

In index notation:

$$B(u, v) = u^a g_{ab} v^b$$

where $g_{ab}$ is a **metric tensor**.

## Index Conventions

Physics conventions are used throughout:

| Notation | Meaning |
|----------|---------|
| $u^a$ | Contravariant vector (upper index) |
| $u_a$ | Covariant vector (lower index) |
| $g_{ab}$ | metric tensor (lowers indices) |
| $g^{ab}$ | inverse metric (raises indices) |

**Einstein summation**: repeated indices (one up, one down) sum over:

$$u^a v_a = \sum_{a=1}^{d} u^a v_a$$

## Metric Tensors

### Euclidean Metric

$$g_{ab} = \delta_{ab}$$

This gives the standard dot product:

$$\langle u, v \rangle = u^a \delta_{ab} v^b = u^a v^a$$

### Scaled Euclidean Metric

$$g_{ab} = \frac{1}{\sqrt{d}} \delta_{ab}$$

This is the metric used in standard attention (Vaswani et al., 2017). The $1/\sqrt{d_k}$ scaling prevents dot products from growing too large in high dimensions.

### Learned Metric

$$g_{ab} = (W^T W)_{ab}$$

For a weight matrix $W$, this ensures the metric is **positive semi-definite**.

## Index Raising and Lowering

**Lowering an index** (contravariant → covariant):

$$v_a = g_{ab} v^b$$

**Raising an index** (covariant → contravariant):

$$v^a = g^{ab} v_b$$

where $g^{ab}$ is the inverse metric satisfying:

$$g^{ac} g_{cb} = \delta^a_b$$

## Properties of Valid Metrics

A valid metric tensor must be:

1. **Symmetric**: $g_{ab} = g_{ba}$
2. **Positive definite**: $v^a g_{ab} v^b > 0$ for all $v \neq 0$

The eigenvalues of $g$ must all be positive.

## Connection to Attention

In attention, the score between query $i$ and key $j$ is:

$$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

For standard attention:

$$g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$$

This is why the formula has $\frac{QK^T}{\sqrt{d_k}}$—the metric tensor embeds in the scaling

## Code Example

```python
from attn_tensors.bilinear import (
    euclidean_metric,
    scaled_euclidean_metric,
    bilinear_form,
    bilinear_form_batch,
)

# Create a metric
d = 64
g = scaled_euclidean_metric(d)  # shape: (64, 64)

# Compute bilinear form for a single pair
u = jnp.ones(d)
v = jnp.ones(d)
result = bilinear_form(u, v, g)  # scalar

# Batch computation for attention scores
Q = jnp.randn(10, d)  # 10 queries
K = jnp.randn(20, d)  # 20 keys
scores = bilinear_form_batch(Q, K, g)  # shape: (10, 20)
```

## Worked Example: Computing Bilinear Forms

Compute attention scores step-by-step with a small example.

**Setup:**
- Dimension $d = 3$
- Query: $q = [1, 2, 1]$
- Key: $k = [2, 1, 0]$
- Metric: $g = \frac{1}{\sqrt{3}} I_3$ (scaled identity)

**Step 1: write out the metric tensor**

$$g_{ab} = \frac{1}{\sqrt{3}} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

**Step 2: compute the bilinear form**

$$B(q, k) = q^a g_{ab} k^b = \frac{1}{\sqrt{3}} \sum_{a=1}^{3} q^a k^a$$

$$= \frac{1}{\sqrt{3}} (1 \cdot 2 + 2 \cdot 1 + 1 \cdot 0)$$

$$= \frac{1}{\sqrt{3}} (2 + 2 + 0) = \frac{4}{\sqrt{3}} \approx 2.31$$

**Interpretation**: this is the attention score between this query-key pair.

## Generalized Metrics: Learning Similarity

### Mahalanobis Distance

A learned metric $g_{ab} = (W^T W)_{ab}$ gives:

$$B(q, k) = q^T W^T W k = (Wq)^T (Wk)$$

This computes dot product in a transformed space

### Asymmetric Bilinear Forms

Non-symmetric matrices are also used:

$$B(q, k) = q^T M k$$

where $M$ is not necessarily symmetric. This allows different "meanings" for queries vs keys.

### Connection to Kernel Methods

The bilinear form defines a kernel:

$$K(q, k) = \exp(q^T g k)$$

This is a valid Mercer kernel when $g$ is positive definite.

## Differential Geometry Perspective

### Tangent Vectors and Cotangent Vectors

In differential geometry:
- **Contravariant vectors** $v^a$: Tangent vectors (directions)
- **Covariant vectors** $u_a$: Cotangent vectors (linear functionals)

The metric converts between them:
- Lower: $v_a = g_{ab} v^b$
- Raise: $v^a = g^{ab} v_b$

### Musical Isomorphisms

In differential geometry notation:
- $\flat$ (flat): Lowers index, $v^\flat = g(v, \cdot)$
- $\sharp$ (sharp): Raises index, $\omega^\sharp = g^{-1}(\omega, \cdot)$

### Inner Product Structure

The metric defines an inner product on the tangent space:

$$\langle u, v \rangle_g = u^a g_{ab} v^b$$

This measures "lengths" and "angles" in feature space.
