+++
title = "Gradient Derivations"
weight = 3
+++

## Chain Rule in Index Notation

For backpropagation, we need gradients of the loss $L$ with respect to parameters.

Using the chain rule:

$$\frac{\partial L}{\partial x^j} = \frac{\partial L}{\partial y^i} \frac{\partial y^i}{\partial x^j}$$

Note the sum over the intermediate index $i$.

## Full Backward Pass

The attention forward pass is:

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$
$$A^{ij} = \text{softmax}_j(S^{ij})$$
$$O^{ib} = A^{ij} V^{jb}$$

For the backward pass, we propagate gradients in reverse order.

### Gradient w.r.t. Values

Given: $O^{ib} = A^{ij} V^{jb}$

$$\frac{\partial O^{ib}}{\partial V^{mn}} = A^{ij} \delta^j_m \delta^b_n = A^{im} \delta^b_n$$

Therefore:

$$\frac{\partial L}{\partial V^{mn}} = \frac{\partial L}{\partial O^{ib}} A^{im} \delta^b_n = A^{im} \frac{\partial L}{\partial O^{in}}$$

**Matrix form**: $\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O}$

### Gradient w.r.t. Attention Weights

$$\frac{\partial O^{ib}}{\partial A^{mn}} = \delta^i_m V^{nb}$$

Therefore:

$$\frac{\partial L}{\partial A^{mn}} = \frac{\partial L}{\partial O^{mb}} V^{nb}$$

**Matrix form**: $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T$

### Gradient Through Softmax

This is the trickiest part. For softmax $A^{ij} = \text{softmax}_j(S^{ij})$:

$$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m A^{ij} (\delta^j_n - A^{in})$$

Chain rule gives:

$$\frac{\partial L}{\partial S^{ij}} = A^{ij} \left( \frac{\partial L}{\partial A^{ij}} - \sum_{j'} A^{ij'} \frac{\partial L}{\partial A^{ij'}} \right)$$

**Intuition**: The gradient through softmax involves:
1. The local gradient at position $(i,j)$
2. Minus the weighted average of gradients (the normalization effect)

### Gradient w.r.t. Queries

Given: $S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$

$$\frac{\partial S^{ij}}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \delta^i_k \delta^a_l K^{ja} = \frac{1}{\sqrt{d_k}} \delta^i_k K^{jl}$$

Therefore:

$$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{kj}} K^{jl}$$

**Matrix form**: $\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K$

### Gradient w.r.t. Keys

Similarly:

$$\frac{\partial L}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{ik}} Q^{il}$$

**Matrix form**: $\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial S}\right)^T Q$

## Summary of Backward Pass

Given upstream gradient $\frac{\partial L}{\partial O}$:

1. **dL/dV** = $A^T \cdot \frac{\partial L}{\partial O}$
2. **dL/dA** = $\frac{\partial L}{\partial O} \cdot V^T$
3. **dL/dS** = $A \odot \left(\frac{\partial L}{\partial A} - \text{rowsum}(A \odot \frac{\partial L}{\partial A})\right)$
4. **dL/dQ** = $\frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} \cdot K$
5. **dL/dK** = $\frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial S}\right)^T \cdot Q$

## Verification

All manual gradients can be verified against JAX autodiff:

```python
from attn_tensors.gradients import verify_gradients

Q = jnp.randn(10, 64)
K = jnp.randn(20, 64)
V = jnp.randn(20, 64)

results = verify_gradients(Q, K, V)
print(results)
# {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True, 'all_correct': True}
```

## Why Manual Gradients?

1. **Education**: Understanding the math behind autodiff
2. **Debugging**: Verify your understanding is correct
3. **Optimization**: Sometimes manual gradients enable tricks (e.g., Flash Attention)
4. **Insight**: See gradient flow and potential issues (vanishing/exploding)

## Deriving the Softmax Jacobian

The softmax Jacobian is crucial for understanding gradient flow. Let's derive it carefully.

### Setup

Given scores $S = [s_1, \ldots, s_n]$, the softmax outputs are:

$$a_i = \frac{e^{s_i}}{\sum_k e^{s_k}} = \frac{e^{s_i}}{Z}$$

We want $\frac{\partial a_i}{\partial s_j}$.

### Case 1: $i = j$ (Diagonal elements)

Using the quotient rule:

$$\frac{\partial a_i}{\partial s_i} = \frac{e^{s_i} \cdot Z - e^{s_i} \cdot e^{s_i}}{Z^2} = \frac{e^{s_i}}{Z} - \frac{e^{2s_i}}{Z^2}$$

$$= a_i - a_i^2 = a_i(1 - a_i)$$

### Case 2: $i \neq j$ (Off-diagonal elements)

$$\frac{\partial a_i}{\partial s_j} = \frac{0 \cdot Z - e^{s_i} \cdot e^{s_j}}{Z^2} = -\frac{e^{s_i} e^{s_j}}{Z^2}$$

$$= -a_i a_j$$

### Combined Formula

$$\frac{\partial a_i}{\partial s_j} = a_i(\delta_{ij} - a_j)$$

Or in matrix form:

$$\frac{\partial A}{\partial S} = \text{diag}(a) - a a^T$$

### In Index Notation with Batch Dimension

For attention matrix $A^{ij}$ (query $i$, key $j$):

$$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m A^{ij}(\delta^j_n - A^{in})$$

The $\delta^i_m$ enforces that softmax is independent across queries.

## Gradient Flow Analysis

### Vanishing Gradients in Sharp Attention

When attention is very peaked ($A^{ij} \approx 1$ for one $j$, 0 elsewhere):

$$\frac{\partial L}{\partial S^{ij}} = A^{ij}(\bar{A}^{ij} - \sum_{j'} A^{ij'}\bar{A}^{ij'})$$

If $A^{ij} \approx 1$ and $A^{ij'} \approx 0$ for $j' \neq j$:

$$\frac{\partial L}{\partial S^{ij}} \approx 1 \cdot (\bar{A}^{ij} - \bar{A}^{ij}) = 0$$

The gradient vanishes! This is the "hard attention" problem.

### Temperature Scaling for Better Gradients

Using temperature $\tau$:

$$A^{ij} = \text{softmax}(S^{ij}/\tau)$$

Higher $\tau$ → softer attention → better gradient flow.

## Numerical Stability

### Log-Sum-Exp Trick

Computing softmax naively:

```python
exp_s = exp(s)  # Can overflow!
a = exp_s / sum(exp_s)
```

Stable version:

```python
s_max = max(s)
exp_s = exp(s - s_max)  # Subtract max for stability
a = exp_s / sum(exp_s)
```

### Gradient with Numerical Stability

The gradient $\frac{\partial L}{\partial S} = A \odot (\bar{A} - \text{rowsum}(A \odot \bar{A}))$ is already stable because:
- $A$ is normalized (no overflow)
- Operations are on bounded quantities

## Complete Backward Pass Algorithm

```python
def attention_backward(dL_dO, Q, K, V, A):
    """
    Args:
        dL_dO: Gradient w.r.t. output, shape (n_q, d_v)
        Q, K, V: Forward pass inputs
        A: Attention weights from forward pass
    
    Returns:
        dL_dQ, dL_dK, dL_dV
    """
    d_k = Q.shape[-1]
    scale = 1.0 / sqrt(d_k)
    
    # Step 1: Gradient w.r.t. Values
    # O = A @ V, so dL_dV = A.T @ dL_dO
    dL_dV = A.T @ dL_dO
    
    # Step 2: Gradient w.r.t. Attention weights
    # O = A @ V, so dL_dA = dL_dO @ V.T
    dL_dA = dL_dO @ V.T
    
    # Step 3: Gradient through softmax
    # dL_dS = A * (dL_dA - sum(A * dL_dA, axis=-1, keepdims=True))
    sum_term = (A * dL_dA).sum(axis=-1, keepdims=True)
    dL_dS = A * (dL_dA - sum_term)
    
    # Step 4: Gradient w.r.t. Queries and Keys
    # S = scale * Q @ K.T
    dL_dQ = scale * dL_dS @ K
    dL_dK = scale * dL_dS.T @ Q
    
    return dL_dQ, dL_dK, dL_dV
```
