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
