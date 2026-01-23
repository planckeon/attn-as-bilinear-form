+++
title = "Attention as Bilinear Form"
description = "A Physicist's Guide to Transformer Attention"
sort_by = "weight"
template = "index.html"
+++

# Attention as Bilinear Form

*A Physicist's Guide to Transformer Attention using Tensor Calculus*

---

## The Core Insight

The attention mechanism in transformers can be understood through the lens of **tensor calculus** and **differential geometry**. This perspective reveals deep connections to physics and provides a rigorous mathematical foundation.

**Standard attention formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**In index notation:**

$$O^{ib} = A^{ij} V^{jb}, \quad A^{ij} = \frac{\exp(S^{ij})}{\sum_k \exp(S^{ik})}, \quad S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

---

## Key Perspectives

### 1. Bilinear Forms and Metric Tensors

The score computation is a **bilinear form**:

$$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

where $g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$ is the **metric tensor**. This gives us:

- A geometric interpretation of similarity
- A framework for learned metrics
- Connection to Riemannian geometry

### 2. Softmax as Gibbs Distribution

The attention weights form a **Gibbs distribution** from statistical mechanics:

$$A^{ij} = \frac{e^{\beta S^{ij}}}{Z^i}, \quad Z^i = \sum_j e^{\beta S^{ij}}$$

where $\beta = 1$ is the inverse temperature. This reveals:

- **High temperature** ($\beta \to 0$): Uniform attention
- **Low temperature** ($\beta \to \infty$): Hard attention (argmax)
- **Entropy** measures attention concentration

### 3. Hopfield Network Interpretation

Modern Hopfield networks show attention is an **associative memory**:

$$\xi^{\text{new}} = V^T \cdot \text{softmax}(\beta \cdot K \cdot \xi)$$

The patterns stored in $K$ are retrieved via the attention mechanism.

---

## Gradient Derivations

Using index notation, we derive all gradients explicitly:

**Gradient w.r.t. Queries:**
$$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{kj}} K^{jl}$$

**Gradient through Softmax:**
$$\frac{\partial L}{\partial S^{ij}} = A^{ij} \left( \frac{\partial L}{\partial A^{ij}} - \sum_{j'} A^{ij'} \frac{\partial L}{\partial A^{ij'}} \right)$$

**Gradient w.r.t. Values:**
$$\frac{\partial L}{\partial V^{kl}} = A^{ik} \frac{\partial L}{\partial O^{il}}$$

All gradients are verified against JAX autodiff.

---

## Quick Start

```python
import jax.numpy as jnp
from attn_tensors import scaled_dot_product_attention
from attn_tensors.bilinear import bilinear_form_batch, scaled_euclidean_metric

# Standard attention
Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
K = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
V = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

output = scaled_dot_product_attention(Q, K, V)

# With explicit metric tensor
g = scaled_euclidean_metric(d=2)
scores = bilinear_form_batch(Q, K, g)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/planckeon/attn-as-bilinear-form
cd attn-as-bilinear-form

# Install with uv
uv sync

# Run tests
uv run pytest tests/ -v
```

---

## Modules

| Module | Description |
|--------|-------------|
| `attention` | Core attention operations |
| `bilinear` | Metric tensors and bilinear forms |
| `gradients` | Manual gradient derivations |
| `softmax` | Softmax, entropy, Gibbs distribution |
| `multihead` | Multi-head attention |
| `masking` | Causal and padding masks |
| `hopfield` | Hopfield network interpretation |

---

## Theory Deep Dives

- [Bilinear Forms and Metrics](@/theory/bilinear.md)
- [Attention Mechanism](@/theory/attention.md)
- [Gradient Derivations](@/theory/gradients.md)
- [Statistical Mechanics View](@/theory/statistical.md)
- [Multi-Head Attention](@/theory/multihead.md)
- [Positional Encodings](@/theory/positional.md)
- [Efficient Attention](@/theory/efficient.md)

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Ramsauer et al. (2020). *Hopfield Networks is All You Need*
3. Amari (1998). *Natural Gradient Works Efficiently in Learning*
