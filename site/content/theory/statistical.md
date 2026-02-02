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

$$A^{ij} \to \begin{cases} 1 & \text{if } j = \arg\max_{j'} S^{ij'} \\\\ 0 & \text{otherwise} \end{cases}$$

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

## Energy-Based View

### Attention as Energy Minimization

Define an energy function:

$$E(i, j) = -S^{ij} = -\frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

The attention weight is the Boltzmann probability:

$$A^{ij} = \frac{e^{-\beta E(i,j)}}{Z^i} = \frac{e^{\beta S^{ij}}}{Z^i}$$

### Free Energy

The free energy for query $i$ is:

$$F^i = -\frac{1}{\beta} \log Z^i$$

This has the familiar form:

$$F^i = \langle E \rangle - \frac{1}{\beta} H^i$$

where $\langle E \rangle = -\sum_j A^{ij} S^{ij}$ is the expected energy and $H^i$ is the entropy.

### Variational Principle

The attention weights minimize:

$$A^* = \arg\min_A \left[ \langle E \rangle - \frac{1}{\beta} H(A) \right]$$

subject to $\sum_j A^{ij} = 1$ and $A^{ij} \geq 0$.

This is equivalent to softmax!

## Deep Dive: Hopfield Networks

### Classical Hopfield (1982)

Energy function:

$$E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j$$

where $s_i \in \{-1, +1\}$ and $W_{ij}$ are synaptic weights.

**Update rule:**

$$s_i \leftarrow \text{sign}\left(\sum_j W_{ij} s_j\right)$$

**Storage capacity:** $\sim 0.14 N$ patterns for $N$ neurons.

### Modern Hopfield (Ramsauer et al., 2020)

Energy function:

$$E = -\text{lse}(\beta K \xi) + \frac{1}{2}\xi^T \xi + \text{const}$$

where $\text{lse}(x) = \log \sum_i e^{x_i}$ is the log-sum-exp.

**Update rule:**

$$\xi^{new} = K^T \text{softmax}(\beta K \xi)$$

This is exactly attention! The query $\xi$ is updated to be a weighted combination of stored patterns (rows of $K$).

### Why Exponential Capacity?

Classical Hopfield fails when patterns have overlap (correlation). The error probability grows with pattern density.

Modern Hopfield uses exponential separation:

$$\text{softmax}(\beta x)_i \approx \begin{cases}
1 & x_i = \max(x) \\\\
e^{-\beta \Delta} & x_i = \max(x) - \Delta
\end{cases}$$

For large $\beta$, even small separation $\Delta$ gives clean retrieval.

**Capacity:** $\sim \exp(d/2)$ patterns in $d$ dimensions!

### Attention as Associative Memory

| Attention | Hopfield |
|-----------|----------|
| Query $Q$ | Pattern to retrieve |
| Keys $K$ | Stored patterns |
| Values $V$ | Pattern outputs |
| Softmax | Update rule |
| Output $O$ | Retrieved pattern |

## Worked Example: Pattern Retrieval

**Setup:** Store 3 patterns as keys, retrieve closest to query.

$$K = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 0.7 & 0.7 \end{pmatrix}, \quad q = \begin{pmatrix} 0.9 \\\\ 0.1 \end{pmatrix}$$

**Step 1: Compute scores**

$$s = K q = \begin{pmatrix} 0.9 \\\\ 0.1 \\\\ 0.7 \end{pmatrix}$$

**Step 2: Apply softmax** (with $\beta = 1$)

$$a = \text{softmax}(s) \approx \begin{pmatrix} 0.48 \\\\ 0.22 \\\\ 0.30 \end{pmatrix}$$

**Step 3: Retrieve pattern**

$$\xi^{new} = K^T a = \begin{pmatrix} 0.48 + 0.21 \\\\ 0.22 + 0.21 \end{pmatrix} = \begin{pmatrix} 0.69 \\\\ 0.43 \end{pmatrix}$$

The query moved toward pattern 1 (which it was closest to).

**With high temperature** ($\beta = 5$):

$$a = \text{softmax}(5s) \approx \begin{pmatrix} 0.88 \\\\ 0.01 \\\\ 0.11 \end{pmatrix}$$

Now retrieval is sharper—almost pure pattern 1.

## Thermodynamic Quantities

### Heat Capacity

The heat capacity measures sensitivity to temperature:

$$C = \frac{\partial \langle E \rangle}{\partial T} = \beta^2 \text{Var}(E)$$

High heat capacity near phase transitions—when attention is "deciding" between multiple keys.

### Susceptibility

Response to perturbation in scores:

$$\chi^{ij}&#95;{kl} = \frac{\partial A^{ij}}{\partial S^{kl}}$$

This is exactly the softmax Jacobian we derived for gradients!

## Connection to Information Theory

### Mutual Information

The attention weights encode mutual information:

$$I(Q; K) \approx H(A) - H(A|Q)$$

where $H(A)$ is the entropy of attention patterns.

### KL Divergence and Attention

The softmax minimizes KL divergence to a uniform prior:

$$A^* = \arg\min_A \left[ -\sum_j A^{ij} S^{ij} + \frac{1}{\beta} D_{\text{KL}}(A \| U) \right]$$

where $U$ is the uniform distribution.
