+++
title = "Multi-Head Attention"
weight = 5
+++

## Why Multiple Heads?

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

With a single head, averaging over positions inhibits this. Multiple heads provide independent "views" of the sequence.

## Index Notation for Multi-Head

Introduce head index $h \in \{1, \ldots, H\}$.

### Input Projections

Starting from input $X^{ib}$ (position $i$, feature $b$):

$$Q^{hia} = X^{ib} W_Q^{hba}$$
$$K^{hja} = X^{jb} W_K^{hba}$$
$$V^{hjc} = X^{jb} W_V^{hbc}$$

where:
- $W_Q^{hba}$: Query projection for head $h$ (shape: $d_{model} \times d_k$)
- $W_K^{hba}$: Key projection for head $h$ (shape: $d_{model} \times d_k$)
- $W_V^{hbc}$: Value projection for head $h$ (shape: $d_{model} \times d_v$)

### Per-Head Attention

Each head computes attention independently:

**Scores:**
$$S^{hij} = \frac{1}{\sqrt{d_k}} Q^{hia} K^{hja}$$

**Attention weights:**
$$A^{hij} = \frac{\exp(S^{hij})}{\sum_{j'} \exp(S^{hij'})}$$

**Per-head output:**
$$O^{hic} = A^{hij} V^{hjc}$$

### Concatenation and Output Projection

Concatenate all heads and project:

$$Y^{id} = O^{hic} W_O^{hcd}$$

where $W_O^{hcd}$ has shape $(H \times d_v) \times d_{model}$.

The sum over $h$ and $c$ combines all heads.

## Parameter Count

For a transformer with:
- $d_{model}$: Model dimension
- $H$: Number of heads
- $d_k = d_v = d_{model}/H$: Per-head dimension

Parameters per layer:
- $W_Q, W_K, W_V$: Each $d_{model} \times d_{model}$
- $W_O$: $d_{model} \times d_{model}$
- **Total**: $4 \cdot d_{model}^2$

## Gradient w.r.t. Projection Weights

### Gradient w.r.t. $W_Q$

Using chain rule:

$$\frac{\partial L}{\partial W_Q^{hba}} = \frac{\partial L}{\partial Q^{hia}} \frac{\partial Q^{hia}}{\partial W_Q^{hba}}$$

Since $Q^{hia} = X^{ib} W_Q^{hba}$:

$$\frac{\partial Q^{h'i'a'}}{\partial W_Q^{hba}} = \delta^{h'}_h X^{i'b} \delta^{a'}_a$$

Therefore:

$$\frac{\partial L}{\partial W_Q^{hba}} = X^{ib} \frac{\partial L}{\partial Q^{hia}}$$

**Matrix form** (for head $h$): $\frac{\partial L}{\partial W_Q^h} = X^T \frac{\partial L}{\partial Q^h}$

### Gradient w.r.t. $W_O$

From $Y^{id} = O^{hic} W_O^{hcd}$:

$$\frac{\partial L}{\partial W_O^{hcd}} = O^{hic} \frac{\partial L}{\partial Y^{id}}$$

## Geometric View: Subspace Projections

Each head projects queries and keys into a $d_k$-dimensional subspace:

$$Q_h = X W_Q^h \in \mathbb{R}^{n \times d_k}$$

Different heads learn to attend to different aspects:
- **Head 1**: Syntactic relationships
- **Head 2**: Semantic similarity
- **Head 3**: Positional patterns
- etc.

The output projection $W_O$ learns to combine these perspectives.

## Tensor Diagram Representation

Multi-head attention can be visualized as a tensor network:

```
      X ──┬── W_Q^h ── Q^h ──┐
          │                  ├── Attention ── O^h ──┬── W_O ── Y
          ├── W_K^h ── K^h ──┤                      │
          │                  │                      │
          └── W_V^h ── V^h ──┘                      │
              (for each head h)                     │
                                                    │
      [Concatenate over h] ─────────────────────────┘
```

## Attention Patterns Across Heads

Different heads often specialize:

| Head Type | Pattern | Example |
|-----------|---------|---------|
| Local | Attend to nearby tokens | "the cat sat" |
| Global | Attend to special tokens | [CLS], [SEP] |
| Syntactic | Attend to syntactic heads | Subject-verb |
| Positional | Fixed offset patterns | Previous token |

## Code Example

```python
from attn_tensors.multihead import (
    multihead_attention,
    split_heads,
    combine_heads,
)

# Input: (batch, seq_len, d_model)
X = jnp.randn(2, 10, 64)

# Multi-head attention
Y = multihead_attention(X, X, X, num_heads=8)
# Y.shape = (2, 10, 64)

# Manual head splitting
d_model, num_heads = 64, 8
d_k = d_model // num_heads  # 8

Q = split_heads(X, num_heads)  # (2, 8, 10, 8)
# Now: (batch, heads, seq, d_k)
```

## Efficient Implementation

### Fused Projections

Instead of separate $W_Q, W_K, W_V$, use a single fused projection:

$$[Q; K; V] = X W_{QKV}$$

where $W_{QKV}$ has shape $d_{model} \times 3d_{model}$.

### Memory Layout

For efficient GPU computation:
- Store as: `(batch, heads, seq, d_k)` not `(batch, seq, heads, d_k)`
- Enables parallel attention computation across heads

## Relation to Ensemble Methods

Multi-head attention resembles ensemble learning:
- Each head is an independent "expert"
- Output projection combines expert opinions
- Diversity encouraged by random initialization

Unlike ensembles, heads share the same input and are trained jointly.
