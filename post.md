From tensor structure to practical gradients. 

## Bilinear Form and Metric Tensors

definition: A bilinear form is a map
$$B: V \times W \rightarrow \mathbb{R}$$ which is linear in both its arguments.

In index notation,

$$B(u,v)=u^i M_{ij} v^j$$
The attention score computation,

$$ S^{ij}=Q^{ia}K^{ja} $$
is a bilinear form where:

- the metric is the scaled identity $M_{ab}$
		$$ M_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab} $$
- we're computing scores between all pairs of query positions $i$ and key positions $j$

### Generalized Metric

We could use any positive definite metric $M_{ab}$:

$$ S^{ij} = Q^{ia} M_{ab} K^{jb} $$
This gives us a family of attention mechanisms:
- Dot-product attention: $M_{ab}$ = $\delta_{ab} / \sqrt{d_k}$ 
- Additive attention: use a learned mapping instead.
- Learned behavior: $M_{ab}$ is a learned matrix.

## Full Attention as Tensor Contraction

Let's write out the full attention mechanism step by step in index notation, as tedious as that may be ;-;
### Inputs:
- Queries: $Q^{ia}$ - shape ($n_q$, $d_k$)
- Keys: $K^{ja}$ - shape ($n_k$, $d_k$)
- Values: $V^{jb}$ - shape ($n_k$, $d_v$)

### Step 1: Score Computation

$$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

The contraction over $a$ is a matrix multiply. Fully expanded:

$$S^{ij} = \frac{1}{\sqrt{d_k}} \sum_{a=1}^{d_k} Q^{ia} K^{ja}$$

### Step 2: Softmax Normalization

$$A^{ij} = \frac{\exp(S^{ij})}{\sum_{j'=1}^{n_k} \exp(S^{ij'})}$$

Note: The normalization is over the key dimension $j$ for each query $i$.

### Step 3: Value Aggregation

$$O^{ib} = A^{ij} V^{jb}$$

Fully expanded:

$$O^{ib} = \sum_{j=1}^{n_k} A^{ij} V^{jb}$$

## Geometric Interpretation

### Attention as a Kernel

The attention weights $A^{ij}$ form a kernel matrix. Think of it as measuring similarity in a feature space defined by the metric $M_{ab}$.

### Riemannian View

If we think of the feature space with metric $g_{ab} = M_{ab}$, then:

$$S^{ij} = g_{ab} Q^{ia} K^{jb}$$

is computing a "distance" or "similarity" between query and key vectors in this metric space.

### Information Geometry

The softmax can be viewed as a Gibbs distribution:

$$A^{ij} = \frac{e^{\beta S^{ij}}}{Z^i}, \quad Z^i = \sum_j e^{\beta S^{ij}}$$

where $\beta = 1$ is the inverse temperature. The partition function $Z^i$ normalizes probabilities.

## Gradient Derivations

### Chain Rule Structure

We have loss $L$ and upstream gradient $\frac{\partial L}{\partial O^{ib}} = \bar{O}^{ib}$.

**Chain rule:**

$$\frac{\partial L}{\partial Q^{kl}} = \frac{\partial L}{\partial O^{ib}} \frac{\partial O^{ib}}{\partial A^{ij}} \frac{\partial A^{ij}}{\partial S^{mn}} \frac{\partial S^{mn}}{\partial Q^{kl}}$$

### Gradient w.r.t. Values

From $O^{ib} = A^{ij} V^{jb}$:

$$\frac{\partial O^{ib}}{\partial V^{mn}} = A^{ij} \delta^j_m \delta^b_n = A^{im} \delta^b_n$$

Therefore:

$$\frac{\partial L}{\partial V^{mn}} = \frac{\partial L}{\partial O^{ib}} A^{im} \delta^b_n = A^{im} \frac{\partial L}{\partial O^{in}}$$

**Matrix form**: $\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O}$

### Gradient Through Softmax

For softmax applied along dimension $j$, the Jacobian is:

$$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m A^{ij}(\delta^j_n - A^{in})$$

Therefore:

$$\frac{\partial L}{\partial S^{ij}} = A^{ij} \left( \frac{\partial L}{\partial A^{ij}} - \sum_{j'} A^{ij'} \frac{\partial L}{\partial A^{ij'}} \right)$$

### Gradient w.r.t. Queries

From $S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$:

$$\frac{\partial S^{ij}}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \delta^i_k K^{jl}$$

**Final gradient:**

$$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{kj}} K^{jl}$$

**Matrix form**: $\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K$

### Gradient w.r.t. Keys

$$\frac{\partial L}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{ik}} Q^{il}$$

**Matrix form**: $\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right)^T Q$

## Multi-Head Attention

Introduce head index $h \in \{1, \ldots, H\}$.

### Projections

$$Q^{hia} = X^{ib} W_Q^{hba}$$
$$K^{hja} = X^{jb} W_K^{hba}$$
$$V^{hjb} = X^{jc} W_V^{hcb}$$

where $X^{ib}$ is the input (sequence position $i$, feature $b$).

### Per-Head Attention

$$S^{hij} = \frac{1}{\sqrt{d_k}} Q^{hia} K^{hja}$$
$$A^{hij} = \text{softmax}_j(S^{hij})$$
$$O^{hib} = A^{hij} V^{hjb}$$

### Concatenation and Projection

Final projection:

$$Y^{ic} = O^{hib} W_O^{hbc}$$

where we sum over both $h$ and $b$.

## Worked Example

Let's compute a tiny example by hand.

**Setup:**
- $n_q = 2, n_k = 3, d_k = 2, d_v = 2$
- $\sqrt{d_k} = \sqrt{2}$

$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$$

**Step 1: Scores**

$$S = \frac{1}{\sqrt{2}} Q K^T = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$

**Step 2: Softmax** (row-wise)

For row 1: $[1/\sqrt{2}, 0, 1/\sqrt{2}]$
- $e^{1/\sqrt{2}} \approx 2.03$, $e^0 = 1$
- $Z_1 = 2.03 + 1 + 2.03 = 5.06$
- $A_1 = [0.40, 0.20, 0.40]$

For row 2: $[0, 1/\sqrt{2}, 1/\sqrt{2}]$
- $A_2 = [0.20, 0.40, 0.40]$

So:

$$A \approx \begin{pmatrix} 0.40 & 0.20 & 0.40 \\ 0.20 & 0.40 & 0.40 \end{pmatrix}$$

**Step 3: Output**

$$O = A V = \begin{pmatrix} 1.20 & 0.80 \\ 0.80 & 1.20 \end{pmatrix}$$

## Statistical Mechanics View

### Softmax as Gibbs Distribution

The attention weights form a Gibbs distribution:

$$A^{ij} = \frac{e^{\beta S^{ij}}}{Z^i}, \quad Z^i = \sum_j e^{\beta S^{ij}}$$

### Temperature Effects

| Temperature | $\beta$ | Behavior |
|-------------|---------|----------|
| High ($T \to \infty$) | $\beta \to 0$ | Uniform attention |
| Normal ($T = 1$) | $\beta = 1$ | Standard attention |
| Low ($T \to 0$) | $\beta \to \infty$ | Hard attention (argmax) |

### Entropy

The entropy measures attention concentration:

$$H^i = -\sum_j A^{ij} \log A^{ij}$$

- Maximum entropy ($H = \log n_k$): Uniform attention
- Minimum entropy ($H = 0$): Hard attention on single key

## Connection to Hopfield Networks

Ramsauer et al. (2020) showed attention implements a Hopfield network update:

$$\xi^{\text{new}} = V^T \text{softmax}(\beta K \xi)$$

The stored patterns are the rows of $K$. Attention retrieves the pattern most similar to the query.

### Storage Capacity

- Classical Hopfield networks: $\sim 0.14n$ patterns
- Modern (attention-based) Hopfield networks: $\sim e^{d/2}$ patterns

## References

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Ramsauer et al. (2020). *Hopfield Networks is All You Need*
3. Amari (1998). *Natural Gradient Works Efficiently in Learning*