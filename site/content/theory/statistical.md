+++
title = "Statistical Mechanics View"
weight = 4
+++

## Softmax as Gibbs Distribution

The attention weights form a **Gibbs distribution** (Boltzmann distribution):

$$A^{ij} = \frac{e^{\beta S^{ij}}}{Z^i}$$

where:
- $\beta$ is the **inverse temperature**
- $Z^i = \sum_j e^{\beta S^{ij}}$ is the **partition function**

In standard attention, $\beta = 1$.

## Temperature Effects

The temperature $T = 1/\beta$ controls attention sharpness:

| Temperature | $\beta$ | Behavior |
|-------------|---------|----------|
| High ($T \to \infty$) | $\beta \to 0$ | Uniform attention |
| Normal ($T = 1$) | $\beta = 1$ | Standard attention |
| Low ($T \to 0$) | $\beta \to \infty$ | Hard attention (argmax) |

### High Temperature Limit

As $\beta \to 0$:

$$A^{ij} \to \frac{1}{n_k}$$

All keys receive equal attention.

### Low Temperature Limit

As $\beta \to \infty$:

$$A^{ij} \to \begin{cases} 1 & \text{if } j = \arg\max_{j'} S^{ij'} \\ 0 & \text{otherwise} \end{cases}$$

Only the highest-scoring key receives attention.

## Entropy of Attention

The **entropy** measures attention concentration:

$$H^i = -\sum_j A^{ij} \log A^{ij}$$

Properties:
- **Maximum entropy** ($H = \log n_k$): Uniform attention
- **Minimum entropy** ($H = 0$): Hard attention on single key
- Lower entropy = more focused attention

## Free Energy

The **free energy** connects entropy and energy:

$$F^i = -\frac{1}{\beta} \log Z^i = \langle S^{ij} \rangle - \frac{1}{\beta} H^i$$

where $\langle S^{ij} \rangle = \sum_j A^{ij} S^{ij}$ is the expected score.

## Connection to Hopfield Networks

Ramsauer et al. (2020) showed attention implements a **Hopfield network** update:

$$\xi^{\text{new}} = V^T \text{softmax}(\beta K \xi)$$

The stored patterns are the rows of $K$. Attention retrieves the pattern most similar to the query.

### Storage Capacity

Classical Hopfield networks store $\sim 0.14 n$ patterns. 

Modern (attention-based) Hopfield networks can store **exponentially many** patterns: $\sim e^{d/2}$ for $d$-dimensional patterns.

## Code Example

```python
from attn_tensors.softmax import (
    softmax_temperature,
    attention_entropy,
    log_partition_function,
)

scores = jnp.randn(10, 20)

# Standard softmax (beta = 1)
A_normal = softmax_temperature(scores, beta=1.0)

# Sharp attention (low temperature)
A_sharp = softmax_temperature(scores, beta=10.0)

# Soft attention (high temperature)
A_soft = softmax_temperature(scores, beta=0.1)

# Compute entropy
H = attention_entropy(A_normal)  # shape: (10,)

# Log partition function
log_Z = log_partition_function(scores, beta=1.0)
```

## Physical Interpretation

Think of attention as a **physical system**:

- **Keys** = possible states
- **Scores** = negative energies (higher score = lower energy = more probable)
- **Temperature** = randomness in state selection
- **Partition function** = normalization over states

At low temperature, the system "freezes" into the lowest energy state (highest score). At high temperature, all states are equally likely.
