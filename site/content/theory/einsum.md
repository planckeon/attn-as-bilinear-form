+++
title = "Einstein Summation and Einsum"
weight = 2
+++

## What is Einsum?

**Einstein summation notation** (einsum) is a compact notation for expressing tensor operations. First introduced by Albert Einstein for tensor calculus in physics, it has become a powerful tool for implementing attention mechanisms.

The core idea: **repeated indices are summed over**.

$$C^{ik} = A^{ij} B^{jk} \quad \Leftrightarrow \quad C_{ik} = \sum_j A_{ij} B_{jk}$$

In code, this becomes:

```python
C = jnp.einsum('ij,jk->ik', A, B)  # Matrix multiplication
```

## Why Learn Einsum?

1. **Self-documenting**: The string `'ij,jk->ik'` tells you exactly what dimensions are involved
2. **Efficient**: Avoids intermediate arrays and unnecessary reshaping
3. **Universal**: Same notation in NumPy, JAX, PyTorch, TensorFlow
4. **Direct mapping**: Index notation in math → einsum in code

> "To become a true shape rotator, one must master einsum."
> — [Sankalp's blog](https://sankalp.bearblog.dev/einsum-new/)

## The Einsum Grammar

```
'input_indices -> output_indices'
```

**Input specification** (left of `→`):
- Comma-separated index labels for each input tensor
- Each index corresponds to one axis of the tensor

**Output specification** (right of `→`):
- Indices that appear in the output
- Order determines output shape

**Key rules:**

| Rule | Meaning |
|------|---------|
| Repeated indices | Multiply along that axis |
| Index not in output | Sum over that axis |
| Rearranged output | Transpose/reshape |

## Basic Examples

### Sum of all elements

$$S = \sum_{i,j} A_{ij}$$

```python
S = jnp.einsum('ij->', A)  # Omit both → sum both
```

### Transpose

$$B_{ji} = A_{ij}$$

```python
B = jnp.einsum('ij->ji', A)  # Rearrange indices
```

### Trace (sum of diagonal)

$$\text{tr}(A) = \sum_i A_{ii}$$

```python
trace = jnp.einsum('ii->', A)  # Same index → diagonal
```

### Matrix-vector multiplication

$$y_i = \sum_j A_{ij} x_j$$

```python
y = jnp.einsum('ij,j->i', A, x)
```

### Matrix multiplication

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

```python
C = jnp.einsum('ij,jk->ik', A, B)
```

### Batch matrix multiplication

$$C_{bij} = \sum_k A_{bik} B_{bkj}$$

```python
C = jnp.einsum('bik,bkj->bij', A, B)
```

## Understanding Summation Indices

Indices are partitioned into two types:

- **Free indices**: Appear in output → outer loops
- **Summation indices**: Not in output → summed (inner loops)

For `'ij,jk->ik'`:
- Free: `i`, `k` (appear in output)
- Summation: `j` (appears in inputs but not output)

This corresponds to nested loops:

```python
# Conceptual equivalent of 'ij,jk->ik'
for i in range(I):
    for k in range(K):
        C[i,k] = 0
        for j in range(J):  # Summation index → innermost
            C[i,k] += A[i,j] * B[j,k]
```

## Tensor Contraction

**Tensor contraction** generalizes matrix multiplication to higher dimensions. When we sum over shared indices, we're "contracting" tensors:

$$\text{result}_i = \sum_j A_i \cdot B_{i,j}$$

This is exactly what einsum does: multiply tensors and sum over specified indices.

## Einsum in Attention

The attention mechanism is a perfect use case for einsum. Let's see how each step maps:

### Standard Indices Convention

| Index | Meaning |
|-------|---------|
| `b` | Batch size |
| `l` or `i` | Query sequence length |
| `m` or `j` | Key/memory sequence length |
| `d` | Model dimension |
| `h` | Head index |
| `k` | Per-head dimension |

### Single-Head Attention

**Attention scores** (query-key dot product):

$$S_{ij} = \frac{1}{\sqrt{d_k}} \sum_a Q_{ia} K_{ja}$$

```python
# S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
S = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

**Attention output** (weighted sum of values):

$$O_{ib} = \sum_j A_{ij} V_{jb}$$

```python
# O^{ib} = A^{ij} V^{jb}
O = jnp.einsum('ij,jb->ib', A, V)
```

### Multi-Head Attention

Multi-head attention adds a head index `h`:

**Project to query space**:

$$Q^{hia} = \sum_d X_{id} W_Q^{hda}$$

```python
# Project input to per-head queries
Q_h = jnp.einsum('id,hda->hia', X, W_Q)
```

**Attention scores per head**:

$$S^{hij} = \sum_a Q^{hia} K^{hja} / \sqrt{d_k}$$

```python
S = jnp.einsum('hia,hja->hij', Q_h, K_h) / jnp.sqrt(d_k)
```

**Weighted values per head**:

$$O^{hic} = \sum_j A^{hij} V^{hjc}$$

```python
O = jnp.einsum('hij,hjc->hic', A, V_h)
```

**Combine heads**:

$$Y_{id} = \sum_{h,c} O^{hic} W_O^{hcd}$$

```python
Y = jnp.einsum('hic,hcd->id', O, W_O)
```

### Batched Multi-Head Attention

Add batch dimension `b`:

```python
# Project queries: X is (batch, seq, d_model)
Q_h = jnp.einsum('bid,hda->bhia', X, W_Q)

# Scores: (batch, heads, seq_q, seq_k)
S = jnp.einsum('bhia,bhja->bhij', Q_h, K_h) / jnp.sqrt(d_k)

# Weighted values
O = jnp.einsum('bhij,bhjc->bhic', A, V_h)

# Combine heads
Y = jnp.einsum('bhic,hcd->bid', O, W_O)
```

## Complete Attention Block Example

Here's the full multi-head attention in einsum (adapted from [xjdr's JAX transformer](https://github.com/xjdr-alt/simple_transformer)):

```python
def attention(input_bld, params):
    """
    B: batch size
    L: sequence length
    M: memory length 
    D: model dimension
    H: number of attention heads
    K: size of each attention key/value
    """
    # Layer norm
    normalized_bld = norm(input_bld, params.attn_norm)
    
    # Project to Q, K, V (summation over d)
    query_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_q_dhk)
    key_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_k_dhk)
    value_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_v_dhk)
    
    # Attention scores (summation over k)
    logits_bhlm = jnp.einsum('blhk,bmhk->bhlm', query_blhk, key_blhk)
    
    # Scale
    _, l, h, k = query_blhk.shape
    logits_bhlm = logits_bhlm / jnp.sqrt(k)
    
    # Causal mask
    mask = jnp.triu(jnp.ones((l, l)), k=1)
    logits_bhlm = logits_bhlm - jnp.inf * mask[None, None, :, :]
    
    # Softmax
    weights_bhlm = jax.nn.softmax(logits_bhlm, axis=-1)
    
    # Weighted sum of values
    wtd_values_blhk = jnp.einsum('blhk,bhlm->blhk', value_blhk, weights_bhlm)
    
    # Output projection
    out_bld = jnp.einsum('blhk,hkd->bld', wtd_values_blhk, params.w_o_hkd)
    
    return out_bld
```

## Connection to Tensor Calculus

Einsum is essentially index notation with automatic summation. Compare:

| Math (index notation) | Einsum |
|-----------------------|--------|
| $C^{ik} = A^{ij} B_j^{\ k}$ | `'ij,jk->ik'` |
| $S^{ij} = Q^{ia} g_{ab} K^{jb}$ | `'ia,ab,jb->ij'` |
| $O^{ib} = A^{ij} V_j^{\ b}$ | `'ij,jb->ib'` |

The summation convention (sum over repeated indices) maps directly to einsum's rule: indices not in output are summed.

## Common Patterns

| Operation | Einsum | Notes |
|-----------|--------|-------|
| Dot product | `'i,i->'` | Both indices same, sum |
| Outer product | `'i,j->ij'` | No shared indices |
| Hadamard product | `'ij,ij->ij'` | Element-wise, keep both |
| Matrix mult | `'ij,jk->ik'` | Contract middle index |
| Batch matmul | `'bij,bjk->bik'` | Preserve batch |
| Bilinear form | `'i,ij,j->'` | $u^T M v$ |
| Trace of product | `'ij,ji->'` | $\text{tr}(AB)$ |

## Why Einsum is Faster

Einsum can be faster than explicit loops and reshapes because:

1. **No intermediate arrays**: Operations are fused
2. **No reshaping overhead**: No need for `np.newaxis` or `reshape`
3. **Optimized paths**: Libraries find optimal contraction order

Example speedup:

```python
# Slow: requires reshape and intermediate array
A = A[:, np.newaxis] * B  # Creates (3,4) temporary
result = A.sum(axis=1)

# Fast: single operation
result = jnp.einsum('i,ij->i', A, B)
```

## Code Examples

Using the library:

```python
from attn_tensors.attention import attention_scores, attention_output

# Attention scores use einsum internally:
# S = jnp.einsum("ia,ja->ij", Q, K)
S = attention_scores(Q, K, scale=True)

# Attention output:
# O = jnp.einsum("ij,jb->ib", A, V)
O = attention_output(A, V)
```

Multi-head attention:

```python
from attn_tensors.multihead import multihead_attention

# All projections use einsum:
# Q_h = jnp.einsum("ib,hba->hia", Q, W_Q)
# S = jnp.einsum("hia,hja->hij", Q_h, K_h)
# O = jnp.einsum("hij,hjc->hic", A, V_h)
# Y = jnp.einsum("hic,hca->ia", O, W_O)
output = multihead_attention(Q, K, V, W_Q, W_K, W_V, W_O)
```

## Practice Problems

Try to write einsum strings for these operations:

1. **Column-wise sum**: $s_j = \sum_i A_{ij}$
2. **Row-wise mean**: $m_i = \frac{1}{n} \sum_j A_{ij}$ (hint: einsum + divide)
3. **Frobenius norm squared**: $\|A\|_F^2 = \sum_{i,j} A_{ij}^2$ (hint: square first)
4. **Bilinear form**: $B(u,v) = u^a g_{ab} v^b$
5. **Batch outer product**: $C_{bij} = a_{bi} b_{bj}$

<details>
<summary>Solutions</summary>

```python
# 1. Column-wise sum
s = jnp.einsum('ij->j', A)

# 2. Row-wise mean
m = jnp.einsum('ij->i', A) / A.shape[1]

# 3. Frobenius norm squared
norm_sq = jnp.einsum('ij,ij->', A, A)

# 4. Bilinear form
B = jnp.einsum('a,ab,b->', u, g, v)

# 5. Batch outer product
C = jnp.einsum('bi,bj->bij', a, b)
```

</details>

## References

This page draws from several excellent resources:

- [Shape Rotation 101: An Intro to Einsum and Jax Transformers](https://sankalp.bearblog.dev/einsum-new/) by Sankalp
- [Einstein summation in NumPy](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
- [Basic guide to einsum](https://ajcr.net/Basic-guide-to-einsum/)
- [Einstein summation in PyTorch](https://rockt.github.io/2018/04/30/einsum)
- [xjdr's simple JAX transformer](https://github.com/xjdr-alt/simple_transformer)
