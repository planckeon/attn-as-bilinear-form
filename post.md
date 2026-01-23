# Attention as Bilinear Form: A Physicist's Guide

*Tensor Calculus, Statistical Mechanics, and Differential Geometry*

**Baalateja Kataru** | January 2026

---

## Abstract

If you've ever wondered what's really going on inside a transformer, this tutorial is for you. The attention mechanism appears through a physicist's lens—using the language of tensor calculus, bilinear forms, and statistical mechanics. Don't worry if that sounds intimidating; the concepts build step by step.

The punchline—that innocent-looking formula `Attention(Q, K, V) = softmax(QK^T / √d_k) V` hides beautiful mathematical structure:
- The score computation is a **bilinear form** with a hidden metric tensor
- The softmax is actually the **Gibbs distribution** from thermodynamics
- The whole thing implements an **associative memory network** with exponential storage capacity

A companion Python library `attn-tensors` implements everything discussed, with 400+ tests verifying gradient derivations against the JAX autodiff.

**Code**: [github.com/planckeon/attn-as-bilinear-form](https://github.com/planckeon/attn-as-bilinear-form)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Foundations](#part-i-foundations)
3. [The Attention Mechanism](#part-ii-the-attention-mechanism)
4. [Statistical Mechanics](#part-iii-statistical-mechanics)
5. [Differential Geometry](#part-iv-differential-geometry)
6. [Gradient Derivations](#part-v-gradient-derivations)
7. [Multi-Head Attention](#part-vi-multi-head-attention)
8. [Attention Variants](#part-vii-attention-variants)
9. [Hopfield Networks](#part-viii-hopfield-networks-and-attention)
10. [Efficient Attention](#part-ix-efficient-attention)
11. [Worked Examples](#part-x-worked-examples)

---

## Introduction

The attention mechanism has become the foundation of modern deep learning. While typically presented in matrix notation optimized for implementation, the underlying mathematical structure reveals deep connections to classical physics and differential geometry.

### Goals

1. **Tensor Calculus**: Express attention using index notation with proper contravariant/covariant indices
2. **Bilinear Forms**: Show that attention scores arise from a bilinear form with an implicit metric tensor
3. **Statistical Mechanics**: Interpret softmax as the Gibbs/Boltzmann distribution
4. **Differential Geometry**: Understand the feature space as a Riemannian manifold
5. **Gradients**: Derive backpropagation formulas in index notation
6. **Efficiency**: Understand Flash Attention and linear attention

### Notation Conventions

| Index | Meaning |
|-------|---------|
| $i$ | Query sequence position ($n_q$ positions) |
| $j, k$ | Key/value sequence position ($n_k$ positions) |
| $a, b, c$ | Feature/embedding dimensions ($d_k$ or $d_v$) |
| $h$ | Attention head index ($H$ heads) |

**Einstein summation**: repeated indices (one up, one down) sum:
$$v^a u_a = \sum_a v^a u_a$$

---

## Part I: Foundations

### Vectors and Dual Vectors

In physics, distinction occurs between vectors and their duals (covectors). A vector $v^a$ lives in a vector space $V$, while a covector $u_a$ lives in the dual space $V^*$. The natural pairing between them is:

$$\langle u, v \rangle = u_a v^a$$

**Intuition**: in ML terms, a vector $v^a$ is a column vector, and a covector $u_a$ is a row vector. Their pairing is just matrix multiplication.

### Metric Tensor

A **metric tensor** $g_{ab}$ is a symmetric, positive-definite tensor that defines an inner product:

$$\langle u, v \rangle_g = u^a g_{ab} v^b$$

The metric allows:
- **Lower indices**: $v_a = g_{ab} v^b$ (vector → covector)
- **Raise indices**: $v^a = g^{ab} v_b$ (covector → vector)

### Bilinear Forms

A **bilinear form** is a map $B\colon V \times W \to \mathbb{R}$ that is linear in both arguments:

$$B(u, v) = u^a M_{ab} v^b$$

**The key insight**: the attention score between a query $q$ and key $k$ is precisely a bilinear form:

$$S = q^a g_{ab} k^b$$

where the metric $g_{ab}$ encodes how similarity is measured in feature space.

### Standard Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | $g_{ab} = \delta_{ab}$ | Standard dot product |
| Scaled Euclidean | $g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$ | Scaled dot-product attention |
| Learned | $g_{ab} = (W^T W)_{ab}$ | Learnable similarity |

**Remark**: the $1/\sqrt{d_k}$ scaling has a statistical interpretation—if $q^a$ and $k^a$ are i.i.d. with zero mean and unit variance, then $\text{Var}(q^a k_a) = d_k$. The scaling normalizes the variance to 1.

### Einstein Summation (Einsum)

Introducing index notation leads to implementation in code. The `einsum` function, available in NumPy, JAX, PyTorch, and other libraries, directly translates index notation to efficient tensor operations.

**Definition (Einstein Summation Convention)**: in the Einstein summation convention, repeated indices implicitly sum:

$$C^{ik} = A^{ij} B_j^{\ k} \quad \Leftrightarrow \quad C_{ik} = \sum_j A_{ij} B_{jk}$$

In code: `C = einsum('ij,jk->ik', A, B)`

The einsum string has a simple grammar:
- **Input specification** (left of →): comma-separated indices for each input tensor
- **Output specification** (right of →): indices in the output
- **Repeated indices** not in output sum over

**Intuition**: think of einsum indices as labels for tensor dimensions. When the same label appears in places, those dimensions pair up for multiplication. When a label is missing from the output, that dimension sums over.

**Common Einsum Patterns**:

| Operation | Einsum | Index Notation |
|-----------|--------|----------------|
| Dot product | `'a,a->'` | $s = u^a v_a$ |
| Outer product | `'a,b->ab'` | $M_{ab} = u_a v_b$ |
| Matrix multiply | `'ij,jk->ik'` | $C^{ik} = A^{ij} B_j^{\ k}$ |
| Transpose | `'ij->ji'` | $B_{ji} = A_{ij}$ |
| Trace | `'ii->'` | $s = A^i_i$ |
| Batch matmul | `'bij,bjk->bik'` | $C^{bik} = A^{bij} B^b_j^{\ k}$ |

#### Einsum for Attention

The attention mechanism maps beautifully to einsum:

**Attention scores** (query-key dot product):
$$S^{ij} = Q^{ia} K^{ja} / \sqrt{d_k}$$

```python
S = einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

**Attention output** (weighted sum of values):
$$O^{ib} = A^{ij} V^{jb}$$

```python
O = einsum('ij,jb->ib', A, V)
```

**With bilinear metric**:
$$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

```python
S = einsum('ia,ab,jb->ij', Q, g, K)
```

#### Multi-Head Einsum

Multi-head attention adds a head index $h$:

```python
# Project to per-head queries: X^{hia} = X^{id} W_Q^{hda}
Q_h = einsum('id,hda->hia', X, W_Q)

# Per-head attention scores: S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
S = einsum('hia,hja->hij', Q_h, K_h) / jnp.sqrt(d_k)

# Per-head outputs: O^{hic} = A^{hij} V^{hjc}
O = einsum('hij,hjc->hic', A, V_h)

# Combine heads: Y^{id} = O^{hic} W_O^{hcd}
Y = einsum('hic,hcd->id', O, W_O)
```

**Intuition**: einsum makes the summation indices explicit. When you see `'hia,hja->hij'`, you immediately know that `a` (the feature dimension) sums over, while `h`, `i`, `j` are preserved.

**Remark**: einsum is not just notation—it's often faster than explicit loops and reshapes because it avoids creating intermediate arrays. For more on einsum, see Sankalp's tutorial ["Shape Rotation 101."](https://sankalp.bearblog.dev/einsum-new/)

---

## Part II: The Attention Mechanism

### Inputs

- **Queries** $Q^{ia}$: what to look for (shape: $n_q \times d_k$)
- **Keys** $K^{ja}$: what to match against (shape: $n_k \times d_k$)
- **Values** $V^{jb}$: what to retrieve (shape: $n_k \times d_v$)

**Intuition**: think of it like a library search:
- The **query** is your question ("books about physics")
- The **keys** are the book titles (what you match against)
- The **values** are the actual book contents (what you get back)

### Step 1: Score Computation

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

Or with explicit metric: $S^{ij} = Q^{ia} g_{ab} K^{jb}$

```python
# S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
S = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

### Step 2: Softmax Normalization

$$A^{ij} = \frac{\exp(S^{ij})}{\sum_k \exp(S^{ik})} = \frac{\exp(S^{ij})}{Z^i}$$

where $Z^i = \sum_j \exp(S^{ij})$ is the **partition function** for query $i$.

### Step 3: Value Aggregation

$$O^{ib} = A^{ij} V^{jb}$$

This contracts over the key index $j$, producing an output for each query.

### Full Attention

**Theorem (Scaled Dot-Product Attention)**:
$$O^{ib} = \frac{\exp(Q^{ia} g_{ac} K^{jc})}{\sum_k \exp(Q^{ia} g_{ac} K^{kc})} V^{jb}$$

Or in matrix notation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

## Part III: Statistical Mechanics

The softmax function is not merely a normalization trick—this is the **Gibbs distribution** from statistical mechanics.

### The Gibbs/Boltzmann Distribution

In statistical mechanics, a system with energy levels $E_j$ at temperature $T$ has probability:

$$P(j) = \frac{\exp(-E_j / T)}{Z}, \quad Z = \sum_j \exp(-E_j / T)$$

For attention:
- **Scores as negative energies**: $S^{ij} = -E^{ij}$ (higher score = lower energy = preferred)
- **Temperature**: Standard attention uses $T = 1$

### Temperature Effects

| Temperature | Behavior |
|-------------|----------|
| $T \to 0$ | **Hard attention** (argmax) |
| $T = 1$ | Standard softmax |
| $T \to \infty$ | **Uniform attention** |

**Theorem (Temperature Limits)**:
$$\lim_{T \to 0} A^{ij} = \begin{cases} 1 & \text{if } j = \arg\max_k S^{ik} \\ 0 & \text{otherwise} \end{cases}$$

$$\lim_{T \to \infty} A^{ij} = \frac{1}{n_k}$$

### Entropy

The **Shannon entropy** of attention weights measures how focused the attention is:

$$H^i = -\sum_j A^{ij} \log A^{ij}$$

- $H = 0$: All attention on one key (focused)
- $H = \log(n_k)$: Uniform attention (diffuse)

**Intuition**: low entropy = confident model. High entropy = uncertain model.

### Free Energy

The **free energy** combines energy and entropy:

$$F^i = -T \log Z^i = \langle E \rangle - T \cdot H$$

**Theorem (Variational Principle)**: the attention weights reduce the free energy:
$$A^* = \arg\min_A [\langle E \rangle - T H(A)]$$

---

## Part IV: Differential Geometry

The feature space where queries and keys live has understanding as a Riemannian manifold.

### Riemannian Manifold

A **Riemannian manifold** $(M, g)$ is a smooth manifold equipped with a metric tensor $g_{ab}(x)$ at each point.

The metric defines:
- **Distances**: $ds^2 = g_{ab} dx^a dx^b$
- **Angles**: $\cos\theta = \frac{u^a g_{ab} v^b}{\|u\|_g \|v\|_g}$

In attention, the feature space $\mathbb{R}^{d_k}$ has metric $g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$.

### Christoffel Symbols

For a general metric, the **Christoffel symbols** are:

$$\Gamma^c_{ab} = \frac{1}{2} g^{cd} (\partial_a g_{bd} + \partial_b g_{ad} - \partial_d g_{ab})$$

For the standard (constant) attention metric, $\Gamma^c_{ab} = 0$.

### Natural Gradient

The **Fisher information matrix** provides a natural Riemannian metric on parameter space:

$$F_{ij} = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta^i} \frac{\partial \log p}{\partial \theta^j}\right]$$

**Natural gradient descent**:
$$\Delta\theta^i = -\eta (F^{-1})^{ij} \frac{\partial L}{\partial \theta^j}$$

This is invariant under reparameterization and often converges faster.

---

## Part V: Gradient Derivations

### Chain Rule in Index Notation

For a scalar loss $L$:
$$\frac{\partial L}{\partial Q^{kl}} = \frac{\partial L}{\partial O^{ib}} \frac{\partial O^{ib}}{\partial A^{mn}} \frac{\partial A^{mn}}{\partial S^{pq}} \frac{\partial S^{pq}}{\partial Q^{kl}}$$

### Value Gradient

From $O^{ib} = A^{ij} V^{jb}$:

$$\frac{\partial L}{\partial V^{mn}} = A^{im} \frac{\partial L}{\partial O^{in}}$$

**Matrix form**: $\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O}$

### Softmax Jacobian

**Lemma**: for $A^{ij} = \exp(S^{ij}) / \sum_k \exp(S^{ik})$:

$$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m A^{ij} (\delta^j_n - A^{in})$$

### Score Gradient

**Theorem (Softmax Gradient)**:
$$\frac{\partial L}{\partial S^{mn}} = A^{mn} \left( \frac{\partial L}{\partial A^{mn}} - \sum_j A^{mj} \frac{\partial L}{\partial A^{mj}} \right)$$

**Intuition**: the subtraction ensures the gradient respects the constraint that attention weights sum to 1.

### Query and Key Gradients

$$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{kj}} K^{jl}$$

$$\frac{\partial L}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{ik}} Q^{il}$$

### Complete Backward Pass

Given upstream gradient $\partial L / \partial O$:

1. $\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O}$
2. $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T$
3. $\frac{\partial L}{\partial S} = A \odot \left(\frac{\partial L}{\partial A} - \text{rowsum}(A \odot \frac{\partial L}{\partial A})\right)$
4. $\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K$
5. $\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial S}\right)^T Q$

---

## Part VI: Multi-Head Attention

Multi-head attention runs attention operations in parallel with different learned projections.

### Structure

For $H$ heads with projection matrices $W_Q^h, W_K^h, W_V^h, W_O^h$:

**Projections**:
$$Q^{hia} = X^{ib} W_Q^{hba}$$
$$K^{hja} = X^{jb} W_K^{hba}$$
$$V^{hjc} = X^{jb} W_V^{hbc}$$

**Per-head attention**:
$$S^{hij} = \frac{1}{\sqrt{d_k}} Q^{hia} K^{hja}$$
$$A^{hij} = \text{softmax}_j(S^{hij})$$
$$O^{hic} = A^{hij} V^{hjc}$$

**Output projection**:
$$Y^{id} = O^{hic} W_O^{hcd}$$

**Intuition**: each head can learn to attend to different aspects—syntax, semantics, positions. The output projection combines these perspectives.

---

## Part VII: Attention Variants

### Causal Masking

For autoregressive models, prevent position $i$ from attending to future positions $j > i\colon$

$$M^{ij} = \begin{cases} 1 & \text{if } j \le i \\ 0 & \text{otherwise} \end{cases}$$

Applied as: $S^{ij} \leftarrow S^{ij} + (1 - M^{ij}) \cdot (-\infty)$

### Learned Bilinear Attention

$$S^{ij} = Q^{ia} M^{ab} K^{jb}$$

where $M^{ab}$ is learnable (use $M = W^T W$ for positive definiteness).

### Relative Position Attention

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} (K^{ja} + R^{(i-j)a})$$

where $R^{ka}$ is a learned embedding for relative position $k$.

---

## Part VIII: Hopfield Networks and Attention

A remarkable connection exists between transformer attention and modern Hopfield networks.

### Classical Hopfield Networks

Store $M$ patterns $\xi_\mu$ in a weight matrix:
$$W_{ij} = \frac{1}{N} \sum_\mu \xi_\mu^i \xi_\mu^j$$

**Problem**: capacity scales only as $M \approx 0.14 N$ patterns.

### Modern Hopfield Networks

Energy function:
$$E(\xi) = -\text{lse}(\beta \cdot K \xi) + \frac{1}{2} \|\xi\|^2$$

**Theorem (Hopfield Update = Attention)**:
$$\xi^{\text{new}} = V^T \text{softmax}(\beta K \xi)$$

This is exactly attention with:
- Query: current state $\xi$
- Keys: stored patterns (rows of $K$)
- Values: pattern outputs (rows of $V$)
- Inverse temperature: $\beta$

### Exponential Capacity

**Theorem**: modern Hopfield networks can store exponentially many patterns:
$$M \approx \exp(d/2)$$

compared to $M \approx 0.14 N$ for classical networks.

**Intuition**: this is why transformers are so powerful. Each attention layer is an associative memory with exponential capacity.

---

## Part IX: Efficient Attention

Standard attention has $O(n^2)$ complexity. Efficient alternatives exist.

### Flash Attention

Achieves $O(n)$ memory by computing attention block-wise using **online softmax**:

Given running max $m$ and sum $\ell$, incorporate new elements:
$$m' = \max(m, \max(x_{\text{new}}))$$
$$\ell' = \ell \cdot \exp(m - m') + \sum \exp(x_{\text{new}} - m')$$

**Complexity**:
- Standard: $O(n^2)$ memory
- Flash: $O(n)$ memory (recomputes during backward)

### Linear Attention

Approximate the exponential kernel with feature maps $\phi$:
$$K(q, k) \approx \phi(q)^T \phi(k)$$

Then:
$$o_i = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

The sums can precompute, giving $O(nd^2)$ instead of $O(n^2 d)$.

**Common feature maps**: random Fourier features, ELU+1, Performers

---

## Part X: Worked Examples

### Example 1: 2-Query, 3-Key Attention

$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$$

**Step 1: scores** ($\sqrt{d_k} = \sqrt{2} \approx 1.414$)

$$S = \frac{1}{\sqrt{2}} Q K^T \approx \begin{pmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \end{pmatrix}$$

**Step 2: softmax**

Row 1: $\exp([0.707, 0, 0.707]) \approx [2.028, 1.000, 2.028]$, sum = 5.056

$$A \approx \begin{pmatrix} 0.401 & 0.198 & 0.401 \\ 0.198 & 0.401 & 0.401 \end{pmatrix}$$

**Step 3: output**

$$O = A V \approx \begin{pmatrix} 1.203 & 0.797 \\ 0.797 & 1.203 \end{pmatrix}$$

### Example 2: Temperature Effects

For scores $S = [2, 1, 0]$:

| T | Weights | Entropy |
|---|---------|---------|
| 0.25 | [0.997, 0.003, 0.000] | 0.02 |
| 0.5 | [0.876, 0.118, 0.006] | 0.42 |
| 1.0 | [0.665, 0.245, 0.090] | 0.80 |
| 2.0 | [0.474, 0.316, 0.211] | 1.02 |
| ∞ | [0.333, 0.333, 0.333] | 1.10 |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $Q^{ia}$ | query tensor, position $i$, feature $a$ |
| $K^{ja}$ | key tensor, position $j$, feature $a$ |
| $V^{jb}$ | value tensor, position $j$, feature $b$ |
| $S^{ij}$ | attention scores |
| $A^{ij}$ | attention weights |
| $O^{ib}$ | output tensor |
| $g_{ab}$ | metric tensor |
| $\delta^a_b$ | kronecker delta |
| $Z^i$ | partition function |
| $H^i$ | entropy |
| $T$ | temperature |
| $\beta$ | inverse temperature ($1/T$) |

---

## The attn-tensors Library

All derivations verify against the JAX autodiff. 400+ tests.

### Quick Start

```bash
git clone https://github.com/planckeon/attn-as-bilinear-form
cd attn-as-bilinear-form
uv sync

# Run tests
uv run pytest tests/ -v

# With MLX (Apple Silicon)
uv sync --extra mlx
```

### Gradient Verification

```python
from attn_tensors.gradients import verify_gradients
import jax.numpy as jnp

Q = jnp.array([[1., 0.], [0., 1.]])
K = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
V = jnp.array([[2., 0.], [0., 2.], [1., 1.]])

results = verify_gradients(Q, K, V)
print(results)  # {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True, 'all_correct': True}
```

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Ramsauer et al. (2020). *Hopfield Networks is All You Need*
3. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*
4. Amari (1998). *Natural Gradient Works Efficiently in Learning*
5. Sankalp (2024). [*Shape Rotation 101: An Intro to Einsum and Jax Transformers*](https://sankalp.bearblog.dev/einsum-new/)
6. Alex Riley. [*A basic introduction to NumPy's einsum*](https://ajcr.net/Basic-guide-to-einsum/)

---

**GitHub**: [github.com/planckeon/attn-as-bilinear-form](https://github.com/planckeon/attn-as-bilinear-form)

**Documentation**: [planckeon.github.io/attn-as-bilinear-form](https://planckeon.github.io/attn-as-bilinear-form/)

**License**: mit
