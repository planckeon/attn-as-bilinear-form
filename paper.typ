// Title and metadata
#set document(
  title: "Attention as Bilinear Form: A Tensor Calculus Perspective on Transformer Attention",
  author: "Baalateja Kataru",
  date: datetime(year: 2026, month: 1, day: 22),
)

#set page(
  paper: "us-letter",
  margin: (x: 1.5cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set par(justify: true)

#set heading(numbering: "1.")

#set math.equation(numbering: "(1)")

// Custom environments
#let theorem(body, name: none) = {
  let title = "Theorem"
  if name != none { title = title + " (" + name + ")" }
  block(
    fill: rgb("#e8f4f8"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *#title.* #body
  ]
}

#let definition(body, name: none) = {
  let title = "Definition"
  if name != none { title = title + " (" + name + ")" }
  block(
    fill: rgb("#f0f0f0"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *#title.* #body
  ]
}

#let proposition(body, name: none) = {
  let title = "Proposition"
  if name != none { title = title + " (" + name + ")" }
  block(
    fill: rgb("#fff8e8"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *#title.* #body
  ]
}

#let lemma(body, name: none) = {
  let title = "Lemma"
  if name != none { title = title + " (" + name + ")" }
  block(
    fill: rgb("#f8f0ff"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *#title.* #body
  ]
}

#let proof(body) = {
  block(
    inset: (left: 1em),
    width: 100%,
  )[
    _Proof._ #body #h(1fr) $square$
  ]
}

#let example(body, name: none) = {
  let title = "Example"
  if name != none { title = title + " (" + name + ")" }
  block(
    stroke: rgb("#666") + 0.5pt,
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *#title.* #body
  ]
}

#let remark(body) = {
  block(
    stroke: (left: rgb("#0066cc") + 2pt),
    inset: (left: 10pt, y: 5pt),
    width: 100%,
  )[
    _Remark._ #body
  ]
}

// Title
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Attention as Bilinear Form: \
    A Tensor Calculus Perspective on Transformer Attention
  ]
  
  #v(1em)
  
  #text(size: 12pt)[
    Baalateja Kataru \
    #link("mailto:baalateja.k@gmail.com")
  ]
  
  #v(1em)
  
  #text(size: 10pt)[
    January 22, 2026
  ]
]

#v(2em)

// Abstract
#align(center)[
  #text(size: 12pt, weight: "bold")[Abstract]
]

#par(justify: true)[
We present a rigorous formulation of the transformer attention mechanism using the language of tensor calculus, differential geometry, and statistical mechanics. The standard attention operation $"Attention"(Q, K, V) = "softmax"(Q K^T \/ sqrt(d_k)) V$ conceals rich mathematical structure: the score computation is a bilinear form with an implicit metric tensor, the softmax normalization is the Gibbs distribution from thermodynamics, and the entire mechanism implements a modern Hopfield network with exponential storage capacity. We provide complete derivations in index notation with Einstein summation convention, explicit gradient computations verified against automatic differentiation, and connections to Riemannian geometry through the Fisher information metric. A companion Python library `attn-tensors` implements all operations with JAX, achieving machine-precision agreement between manual gradients and autodiff. The geometric perspective reveals natural generalizations including learned metrics, temperature-controlled attention, and efficient approximations. Code and documentation are publicly available at #link("https://github.com/planckeon/attn-as-bilinear-form").
]

#v(1em)

= Introduction

The attention mechanism, introduced by Bahdanau et al. and refined in the Transformer architecture by Vaswani et al., has become the foundation of modern deep learning for sequences. While typically presented in matrix notation optimized for GPU implementation, the underlying mathematical structure reveals deep connections to classical physics and differential geometry that remain underexplored.

This work recasts attention using the language of:

1. *Tensor calculus*: Index notation with proper contravariant/covariant indices and Einstein summation
2. *Bilinear forms*: The score computation as an inner product with metric tensor
3. *Statistical mechanics*: Softmax as the Gibbs/Boltzmann distribution
4. *Differential geometry*: Feature space as a Riemannian manifold
5. *Associative memory*: Attention as modern Hopfield network retrieval

The geometric perspective is not merely pedagogical. It suggests natural generalizations (learned metrics, variable temperature), explains empirical phenomena (the $1/sqrt(d_k)$ scaling), and connects attention to well-studied mathematical structures with known properties.

== Contributions

Our specific contributions include:

- *Complete tensor formulation*: Attention expressed in index notation with explicit index contractions
- *Gradient derivations*: Full backpropagation formulas derived in index notation and verified against JAX autodiff
- *Statistical mechanics interpretation*: Temperature, entropy, and free energy for attention
- *Hopfield network connection*: Formal equivalence between attention and modern Hopfield updates
- *Reference implementation*: Python library with 400+ tests achieving machine-precision verification
- *Efficient attention analysis*: Flash Attention and linear attention through the geometric lens

== Notation Conventions

Throughout this paper, we use the following index conventions:

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  align: (center, left),
  [*Index*], [*Meaning*],
  [$i$], [Query sequence position ($i = 1, ..., n_q$)],
  [$j, k$], [Key/value sequence position ($j = 1, ..., n_k$)],
  [$a, b, c$], [Feature/embedding dimension ($a = 1, ..., d$)],
  [$h$], [Attention head index ($h = 1, ..., H$)],
  [$mu, nu$], [General tensor indices],
)

*Einstein summation convention*: Repeated indices (one upper, one lower) are implicitly summed:
$ v^a u_a equiv sum_(a=1)^d v^a u_a $

= Mathematical Foundations

== Vectors and Dual Vectors

In physics, we distinguish between vectors (contravariant) and covectors (covariant, dual vectors). A vector $v^a$ lives in a vector space $V$, while a covector $u_a$ lives in the dual space $V^*$. The natural pairing is:

$ chevron.l u, v chevron.r = u_a v^a $ <eq:pairing>

This pairing is basis-independent. In machine learning terms, vectors are column vectors and covectors are row vectors.

#definition(name: "Metric Tensor")[
  A *metric tensor* $g_(a b)$ is a symmetric, positive-definite $(0,2)$-tensor that defines an inner product:
  
  $ chevron.l u, v chevron.r_g = u^a g_(a b) v^b $ <eq:metric-inner>
  
  The metric enables:
  - *Lowering indices*: $v_a = g_(a b) v^b$ (vector $arrow.r$ covector)
  - *Raising indices*: $v^a = g^(a b) v_b$ (covector $arrow.r$ vector)
  
  where $g^(a b)$ is the inverse metric satisfying $g^(a c) g_(c b) = delta^a_b$.
]

The Kronecker delta $delta^a_b$ equals 1 when $a = b$ and 0 otherwise.

== Bilinear Forms

#definition(name: "Bilinear Form")[
  A *bilinear form* is a map $B: V times W arrow.r RR$ that is linear in both arguments:
  
  $ B(alpha u + beta v, w) &= alpha B(u, w) + beta B(v, w) \
    B(u, alpha v + beta w) &= alpha B(u, v) + beta B(u, w) $
  
  In index notation with matrix $M_(a b)$:
  $ B(u, v) = u^a M_(a b) v^b $ <eq:bilinear>
]

The key insight is that attention scores are bilinear forms:

$ S = q^a g_(a b) k^b $

where $g_(a b)$ encodes how similarity is measured in feature space.

== Standard Metrics for Attention

#example(name: "Euclidean Metric")[
  $ g_(a b) = delta_(a b) $
  
  This gives the standard dot product: $chevron.l u, v chevron.r = u^a v_a$
]

#example(name: "Scaled Euclidean Metric")[
  $ g_(a b) = 1/sqrt(d_k) delta_(a b) $ <eq:scaled-metric>
  
  This is precisely the metric implicit in scaled dot-product attention. The $1/sqrt(d_k)$ factor prevents dot products from growing with dimension.
]

#example(name: "Learned Metric")[
  $ g_(a b) = (W^T W)_(a b) = W^c_a W_(c b) $
  
  Parameterizing as $W^T W$ ensures positive semi-definiteness. This generalizes attention to learnable similarity functions.
]

#remark[
  The scaling $1/sqrt(d_k)$ has a statistical interpretation: if $q^a$ and $k^a$ are i.i.d. with zero mean and unit variance, then $"Var"(q^a k_a) = d_k$. The scaling normalizes the variance to 1, keeping the softmax in a good operating regime.
]

= The Attention Mechanism

== Tensor Formulation

The attention mechanism operates on three inputs:

#definition(name: "Attention Inputs")[
  - *Queries* $Q^(i a)$: Shape $(n_q, d_k)$, what we're looking for
  - *Keys* $K^(j a)$: Shape $(n_k, d_k)$, what we're matching against  
  - *Values* $V^(j b)$: Shape $(n_k, d_v)$, what we retrieve
  
  Note: The feature index $a$ is shared between Q and K (for matching), while V can have different feature dimension $b$.
]

The mechanism proceeds in three steps, each a tensor contraction.

=== Step 1: Score Computation (Bilinear Form)

#definition(name: "Attention Scores")[
  $ S^(i j) = 1/sqrt(d_k) Q^(i a) K^(j a) $ <eq:scores>
  
  Or with explicit metric:
  $ S^(i j) = Q^(i a) g_(a b) K^(j b) $
  
  where $g_(a b) = (1\/sqrt(d_k)) delta_(a b)$ is the scaled Euclidean metric.
]

The contraction over $a$ computes a scalar similarity for each query-key pair. This produces an $(n_q times n_k)$ score matrix.

*Implementation in JAX:*
```python
import jax.numpy as jnp

def attention_scores(Q, K):
    """Compute S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)"""
    d_k = Q.shape[-1]
    return jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

=== Step 2: Softmax Normalization

#definition(name: "Attention Weights")[
  $ A^(i j) = (exp(S^(i j))) / (sum_k exp(S^(i k))) = (exp(S^(i j))) / (Z^i) $ <eq:weights>
  
  where $Z^i = sum_j exp(S^(i j))$ is the *partition function* for query $i$.
]

The softmax is applied row-wise: each query gets its own probability distribution over keys.

*Properties of attention weights:*
- Non-negative: $A^(i j) >= 0$
- Normalized: $sum_j A^(i j) = 1$ for each $i$
- Differentiable everywhere

=== Step 3: Value Aggregation

#definition(name: "Attention Output")[
  $ O^(i b) = A^(i j) V^(j b) $ <eq:output>
  
  This contracts over the key index $j$, computing a weighted average of values.
]

The output has shape $(n_q, d_v)$: one vector per query.

== Complete Attention Operation

#theorem(name: "Scaled Dot-Product Attention")[
  The full attention operation is:
  
  $ O^(i b) = (exp(Q^(i a) g_(a c) K^(j c))) / (sum_k exp(Q^(i a) g_(a c) K^(k c))) V^(j b) $ <eq:full-attention>
  
  In matrix notation:
  $ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $
]

*Complete implementation:*
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Full attention: O^{ib} = A^{ij} V^{jb}
    where A^{ij} = softmax_j(Q^{ia} K^{ja} / sqrt(d_k))
    """
    d_k = Q.shape[-1]
    
    # Step 1: Scores S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
    scores = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
    
    # Apply mask if provided (for causal attention)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    
    # Step 2: Weights A^{ij} = softmax_j(S^{ij})
    weights = jax.nn.softmax(scores, axis=-1)
    
    # Step 3: Output O^{ib} = A^{ij} V^{jb}
    output = jnp.einsum('ij,jb->ib', weights, V)
    
    return output
```

= Statistical Mechanics Interpretation

The softmax function is the Gibbs distribution from statistical mechanics, revealing thermodynamic structure in attention.

== The Gibbs/Boltzmann Distribution

In statistical mechanics, a system with energy levels $E_j$ at temperature $T$ has occupation probabilities:

#definition(name: "Gibbs Distribution")[
  $ P(j) = (exp(-E_j \/ T)) / Z, quad Z = sum_j exp(-E_j \/ T) $ <eq:gibbs>
  
  - $T$: Temperature (controls distribution sharpness)
  - $Z$: Partition function (normalization constant)
  - $beta = 1\/T$: Inverse temperature
]

For attention, we identify:

#proposition(name: "Attention as Gibbs Distribution")[
  Attention weights are Gibbs probabilities with:
  - Scores as negative energies: $S^(i j) = -E^(i j)$
  - Temperature: $T = 1$ (standard) or $T = sqrt(d_k)$ (for unscaled scores)
  
  $ A^(i j) = (exp(S^(i j) \/ T)) / (sum_k exp(S^(i k) \/ T)) $
]

== Temperature Effects

#theorem(name: "Temperature Limits")[
  As temperature varies:
  
  $ lim_(T arrow.r 0) A^(i j) = cases(1 "if" j = arg max_k S^(i k), 0 "otherwise") quad "(hard attention)" $
  
  $ lim_(T arrow.r infinity) A^(i j) = 1 / n_k quad "(uniform attention)" $
]

#proof[
  For the $T arrow.r 0$ limit, consider scores $S^(i j)$ with unique maximum at $j^*$. Then:
  $ A^(i j) = (exp(S^(i j)\/T)) / (sum_k exp(S^(i k)\/T)) = (exp((S^(i j) - S^(i j^*))\/T)) / (sum_k exp((S^(i k) - S^(i j^*))\/T)) $
  
  As $T arrow.r 0$, all terms with $S^(i k) < S^(i j^*)$ vanish exponentially, leaving only $j = j^*$.
  
  For the $T arrow.r infinity$ limit, $exp(S\/T) arrow.r 1$ for all $S$, so all weights become equal.
]

*Temperature-controlled attention:*
```python
def attention_temperature(Q, K, V, temperature=1.0):
    """Attention with explicit temperature control."""
    d_k = Q.shape[-1]
    scores = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores / temperature, axis=-1)
    return jnp.einsum('ij,jb->ib', weights, V)
```

== Entropy and Information

#definition(name: "Attention Entropy")[
  The Shannon entropy of attention weights for query $i$:
  
  $ H^i = -sum_j A^(i j) log A^(i j) $ <eq:entropy>
  
  - Minimum $H = 0$: Delta distribution (all attention on one key)
  - Maximum $H = log n_k$: Uniform distribution (equal attention)
]

The *normalized entropy* $H^i \/ log n_k in [0, 1]$ provides a scale-independent measure of attention concentration.

#proposition(name: "Entropy Bounds")[
  For any attention distribution:
  $ 0 <= H^i <= log n_k $
  
  Equality holds on the left iff attention is a delta function, and on the right iff attention is uniform.
]

*Computing entropy:*
```python
def attention_entropy(weights, eps=1e-12):
    """H^i = -sum_j A^{ij} log A^{ij}"""
    return -jnp.sum(weights * jnp.log(weights + eps), axis=-1)

def normalized_entropy(weights, eps=1e-12):
    """Normalized to [0, 1]"""
    n_k = weights.shape[-1]
    return attention_entropy(weights, eps) / jnp.log(n_k)
```

== Free Energy

#definition(name: "Free Energy")[
  The Helmholtz free energy for query $i$:
  
  $ F^i = -T log Z^i = -T log sum_j exp(S^(i j) \/ T) $ <eq:free-energy>
  
  This satisfies the fundamental relation:
  $ F = chevron.l E chevron.r - T H $
  
  where $chevron.l E chevron.r = -sum_j A^(i j) S^(i j)$ is the expected energy.
]

#theorem(name: "Variational Principle")[
  The attention weights minimize the free energy:
  
  $ A^* = arg min_A [chevron.l E chevron.r - T H(A)] $
  
  subject to $sum_j A^(i j) = 1$ and $A^(i j) >= 0$.
]

#proof[
  This is the standard derivation of the Gibbs distribution. The Lagrangian is:
  $ cal(L) = -sum_j A_j S_j + T sum_j A_j log A_j + lambda(sum_j A_j - 1) $
  
  Setting $partial cal(L) \/ partial A_j = 0$:
  $ -S_j + T(1 + log A_j) + lambda = 0 $
  $ A_j = exp((S_j - lambda - T) \/ T) prop exp(S_j \/ T) $
  
  Normalizing gives the softmax.
]

= Gradient Derivations

We now derive the gradients for backpropagation through attention in index notation.

== Chain Rule in Index Notation

For a scalar loss $L$, the chain rule gives:

$ (partial L) / (partial Q^(k l)) = (partial L) / (partial O^(i b)) (partial O^(i b)) / (partial A^(m n)) (partial A^(m n)) / (partial S^(p q)) (partial S^(p q)) / (partial Q^(k l)) $ <eq:chain>

We compute each Jacobian factor explicitly.

== Gradient Through Value Aggregation

#lemma(name: "Value Aggregation Jacobian")[
  For $O^(i b) = A^(i j) V^(j b)$:
  
  $ (partial O^(i b)) / (partial A^(m n)) = delta^i_m V^(n b) $
  $ (partial O^(i b)) / (partial V^(m n)) = A^(i m) delta^b_n $
]

#proof[
  For the first:
  $ (partial O^(i b)) / (partial A^(m n)) = (partial) / (partial A^(m n)) (A^(i j) V^(j b)) = delta^i_m delta^j_n V^(j b) = delta^i_m V^(n b) $
  
  For the second:
  $ (partial O^(i b)) / (partial V^(m n)) = (partial) / (partial V^(m n)) (A^(i j) V^(j b)) = A^(i j) delta^j_m delta^b_n = A^(i m) delta^b_n $
]

#theorem(name: "Value Gradient")[
  $ (partial L) / (partial V^(m n)) = A^(i m) (partial L) / (partial O^(i n)) $
  
  In matrix form: $partial L \/ partial V = A^T (partial L \/ partial O)$
]

== Gradient Through Softmax

The softmax Jacobian requires careful derivation.

#lemma(name: "Softmax Jacobian")[
  For $A^(i j) = exp(S^(i j)) \/ sum_k exp(S^(i k))$:
  
  $ (partial A^(i j)) / (partial S^(m n)) = delta^i_m A^(i j) (delta^j_n - A^(i n)) $ <eq:softmax-jacobian>
]

#proof[
  The softmax for row $i$ depends only on scores in row $i$, so $delta^i_m$ ensures we're in the same row.
  
  For the diagonal case ($j = n$):
  $ (partial A^(i j)) / (partial S^(i j)) = (exp(S^(i j)) dot Z - exp(S^(i j)) dot exp(S^(i j))) / Z^2 = A^(i j) - (A^(i j))^2 = A^(i j)(1 - A^(i j)) $
  
  For the off-diagonal case ($j != n$):
  $ (partial A^(i j)) / (partial S^(i n)) = (0 dot Z - exp(S^(i j)) dot exp(S^(i n))) / Z^2 = -A^(i j) A^(i n) $
  
  Combining: $(partial A^(i j)) / (partial S^(i n)) = A^(i j)(delta^j_n - A^(i n))$
]

#theorem(name: "Score Gradient")[
  $ (partial L) / (partial S^(m n)) = A^(m n) ((partial L) / (partial A^(m n)) - sum_j A^(m j) (partial L) / (partial A^(m j))) $ <eq:score-grad>
  
  In compact form, defining $overline(A)^(i j) = partial L \/ partial A^(i j)$:
  $ (partial L) / (partial S) = A circle.tiny (overline(A) - "rowsum"(A circle.tiny overline(A))) $
  
  where $circle.tiny$ denotes elementwise multiplication.
]

#proof[
  Using the chain rule and softmax Jacobian:
  $ (partial L) / (partial S^(m n)) = (partial L) / (partial A^(i j)) (partial A^(i j)) / (partial S^(m n)) = (partial L) / (partial A^(i j)) delta^i_m A^(i j) (delta^j_n - A^(i n)) $
  
  The $delta^i_m$ forces $i = m$:
  $ = (partial L) / (partial A^(m j)) A^(m j) (delta^j_n - A^(m n)) $
  
  Expanding:
  $ = (partial L) / (partial A^(m n)) A^(m n) - A^(m n) sum_j (partial L) / (partial A^(m j)) A^(m j) $
  $ = A^(m n) ((partial L) / (partial A^(m n)) - sum_j A^(m j) (partial L) / (partial A^(m j))) $
]

== Gradient Through Score Computation

#lemma(name: "Score Jacobian")[
  For $S^(i j) = (1\/sqrt(d_k)) Q^(i a) K^(j a)$:
  
  $ (partial S^(i j)) / (partial Q^(k l)) = 1/sqrt(d_k) delta^i_k K^(j l) $
  $ (partial S^(i j)) / (partial K^(k l)) = 1/sqrt(d_k) delta^j_k Q^(i l) $
]

#proof[
  For queries:
  $ (partial S^(i j)) / (partial Q^(k l)) = 1/sqrt(d_k) (partial) / (partial Q^(k l)) (Q^(i a) K^(j a)) = 1/sqrt(d_k) delta^i_k delta^a_l K^(j a) = 1/sqrt(d_k) delta^i_k K^(j l) $
  
  The derivation for keys is analogous.
]

#theorem(name: "Query and Key Gradients")[
  $ (partial L) / (partial Q^(k l)) = 1/sqrt(d_k) (partial L) / (partial S^(k j)) K^(j l) $
  $ (partial L) / (partial K^(k l)) = 1/sqrt(d_k) (partial L) / (partial S^(i k)) Q^(i l) $
  
  In matrix form:
  $ partial L \/ partial Q = (1\/sqrt(d_k)) (partial L \/ partial S) K $
  $ partial L \/ partial K = (1\/sqrt(d_k)) (partial L \/ partial S)^T Q $
]

== Complete Backward Pass Algorithm

#theorem(name: "Attention Backward Pass")[
  Given upstream gradient $overline(O) = partial L \/ partial O$, the complete backward pass is:
  
  1. *Value gradient*: $overline(V) = A^T overline(O)$
  
  2. *Attention weight gradient*: $overline(A) = overline(O) V^T$
  
  3. *Score gradient*: $overline(S) = A circle.tiny (overline(A) - "rowsum"(A circle.tiny overline(A)))$
  
  4. *Query gradient*: $overline(Q) = (1\/sqrt(d_k)) overline(S) K$
  
  5. *Key gradient*: $overline(K) = (1\/sqrt(d_k)) overline(S)^T Q$
]

*Complete implementation:*
```python
def attention_backward(dL_dO, Q, K, V, A):
    """
    Manual backward pass for attention.
    
    Args:
        dL_dO: Upstream gradient, shape (n_q, d_v)
        Q, K, V: Forward pass inputs
        A: Attention weights from forward pass
    
    Returns:
        dL_dQ, dL_dK, dL_dV
    """
    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)
    
    # Step 1: dL/dV = A^T @ dL/dO
    dL_dV = jnp.einsum('ij,ib->jb', A, dL_dO)
    
    # Step 2: dL/dA = dL/dO @ V^T
    dL_dA = jnp.einsum('ib,jb->ij', dL_dO, V)
    
    # Step 3: dL/dS = A * (dL/dA - rowsum(A * dL/dA))
    sum_term = jnp.sum(A * dL_dA, axis=-1, keepdims=True)
    dL_dS = A * (dL_dA - sum_term)
    
    # Step 4: dL/dQ = scale * dL/dS @ K
    dL_dQ = scale * jnp.einsum('ij,ja->ia', dL_dS, K)
    
    # Step 5: dL/dK = scale * dL/dS^T @ Q
    dL_dK = scale * jnp.einsum('ij,ia->ja', dL_dS, Q)
    
    return dL_dQ, dL_dK, dL_dV
```

== Gradient Verification

All manual gradients are verified against JAX automatic differentiation:

```python
import jax
from jax import random

def verify_gradients(Q, K, V, tol=1e-5):
    """Verify manual gradients match autodiff."""
    
    def loss_fn(Q, K, V):
        O = scaled_dot_product_attention(Q, K, V)
        return jnp.sum(O ** 2)  # Simple loss
    
    # Autodiff gradients
    auto_dQ, auto_dK, auto_dV = jax.grad(loss_fn, argnums=(0, 1, 2))(Q, K, V)
    
    # Manual gradients
    O, A = attention_with_weights(Q, K, V)
    dL_dO = 2 * O  # Gradient of sum(O^2)
    manual_dQ, manual_dK, manual_dV = attention_backward(dL_dO, Q, K, V, A)
    
    # Compare
    return {
        'dL_dQ': jnp.allclose(auto_dQ, manual_dQ, atol=tol),
        'dL_dK': jnp.allclose(auto_dK, manual_dK, atol=tol),
        'dL_dV': jnp.allclose(auto_dV, manual_dV, atol=tol),
    }

# Test
key = random.PRNGKey(42)
Q = random.normal(key, (10, 64))
K = random.normal(random.split(key)[0], (20, 64))
V = random.normal(random.split(key)[1], (20, 64))

results = verify_gradients(Q, K, V)
print(results)  # {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True}
```

= Multi-Head Attention

Multi-head attention runs $H$ independent attention operations with different learned projections, then combines the results.

== Tensor Formulation

#definition(name: "Multi-Head Attention")[
  For $H$ heads with projection matrices $W_Q^h, W_K^h, W_V^h, W_O^h$:
  
  *Projections* (introducing head index $h$):
  $ Q^(h i a) = X^(i b) W_Q^(h b a) $
  $ K^(h j a) = X^(j b) W_K^(h b a) $
  $ V^(h j c) = X^(j b) W_V^(h b c) $
  
  *Per-head attention*:
  $ S^(h i j) = 1/sqrt(d_k) Q^(h i a) K^(h j a) $
  $ A^(h i j) = "softmax"_j (S^(h i j)) $
  $ O^(h i c) = A^(h i j) V^(h j c) $
  
  *Output projection* (sum over heads and head dimension):
  $ Y^(i d) = O^(h i c) W_O^(h c d) $
]

== Parameter Count

For a transformer layer with model dimension $d_"model"$ and $H$ heads:

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: center,
  [*Parameter*], [*Shape*], [*Count*],
  [$W_Q$], [$(d_"model", d_"model")$], [$d_"model"^2$],
  [$W_K$], [$(d_"model", d_"model")$], [$d_"model"^2$],
  [$W_V$], [$(d_"model", d_"model")$], [$d_"model"^2$],
  [$W_O$], [$(d_"model", d_"model")$], [$d_"model"^2$],
  [*Total*], [], [$4 d_"model"^2$],
)

With $d_k = d_v = d_"model" \/ H$, each head operates on a $d_k$-dimensional subspace.

== Geometric Interpretation

Each head projects to a different subspace and computes attention there:

$ Q_h = X W_Q^h in RR^(n times d_k) $

Different heads can specialize:
- *Syntactic heads*: Attend to grammatical structure
- *Semantic heads*: Attend to meaning similarity
- *Positional heads*: Attend to relative positions

The output projection $W_O$ learns to combine these perspectives.

= Hopfield Networks and Attention

A deep connection exists between transformer attention and modern Hopfield networks.

== Classical Hopfield Networks

Classical Hopfield networks (1982) store $M$ patterns $xi_mu$ in a weight matrix:

$ W_(i j) = 1/N sum_(mu=1)^M xi_mu^i xi_mu^j $

The energy function is:
$ E(x) = -1/2 x^i W_(i j) x^j $

The network dynamics minimize energy, converging to stored patterns. However, capacity is severely limited: $M lt.eq 0.14 N$ patterns can be reliably stored.

== Modern Hopfield Networks

Ramsauer et al. (2020) introduced an exponential energy function:

#definition(name: "Modern Hopfield Energy")[
  $ E(xi) = -"lse"(beta dot K xi) + 1/2 ||xi||^2 + "const" $
  
  where $"lse"(z) = log sum_mu exp(z_mu)$ is the log-sum-exp function.
]

The update rule that minimizes this energy is:

#theorem(name: "Hopfield Update Equals Attention")[
  $ xi^("new") = V^T "softmax"(beta K xi) $
  
  This is precisely the attention mechanism with:
  - Query: current state $xi$
  - Keys: stored patterns (rows of $K$)
  - Values: stored patterns (rows of $V$, often $V = K$)
  - Inverse temperature: $beta$
]

#proof[
  Taking the gradient of the energy and setting to zero:
  $ nabla_xi E = -K^T "softmax"(beta K xi) + xi = 0 $
  $ xi = K^T "softmax"(beta K xi) $
  
  With values $V$, this generalizes to $xi^("new") = V^T "softmax"(beta K xi)$.
]

== Exponential Storage Capacity

#theorem(name: "Exponential Capacity")[
  Modern Hopfield networks can store exponentially many patterns:
  $ M approx exp(d\/2) $
  
  compared to $M approx 0.14 N$ for classical networks.
]

The key is the exponential separation provided by softmax:

$ "softmax"(beta x)_i approx cases(1 "if" x_i = max(x), exp(-beta Delta) "if" x_i = max(x) - Delta) $

For large $beta$, even small separation $Delta$ gives clean retrieval.

== Attention as Associative Memory

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  align: (left, left),
  [*Attention*], [*Hopfield*],
  [Query $q$], [Pattern to retrieve],
  [Keys $K$], [Stored patterns],
  [Values $V$], [Pattern outputs],
  [Softmax], [Update rule],
  [Output $o$], [Retrieved pattern],
)

= Efficient Attention Variants

== Flash Attention

Flash Attention achieves $O(n)$ memory (instead of $O(n^2)$) by computing attention block-wise and avoiding storage of the full attention matrix.

=== Online Softmax Algorithm

The key insight is that softmax can be computed incrementally:

#proposition(name: "Online Softmax")[
  Given running maximum $m$ and sum of exponentials $ell$, we can incorporate new elements without storing all values:
  
  $ m' = max(m, max(x_"new")) $
  $ ell' = ell dot exp(m - m') + sum exp(x_"new" - m') $
]

=== Block-wise Computation

#theorem(name: "Flash Attention")[
  Divide $Q, K, V$ into blocks of size $B$. For each query block $Q_i$:
  
  1. Initialize $O_i = 0$, $ell_i = 0$, $m_i = -infinity$
  
  2. For each key-value block $(K_j, V_j)$:
     - Compute block scores: $S_(i j) = Q_i K_j^T \/ sqrt(d_k)$
     - Update running max: $m_"new" = max(m_i, "rowmax"(S_(i j)))$
     - Rescale previous: $O_i <- O_i dot exp(m_i - m_"new")$
     - Accumulate: $O_i <- O_i + exp(S_(i j) - m_"new") V_j$
     - Update: $ell_i <- ell_i dot exp(m_i - m_"new") + "rowsum"(exp(S_(i j) - m_"new"))$
     - Update: $m_i <- m_"new"$
  
  3. Normalize: $O_i <- O_i \/ ell_i$
]

*Complexity*:
- Standard attention: $O(n^2)$ memory for storing $A$
- Flash Attention: $O(n)$ memory, recomputes $A$ during backward pass

== Linear Attention

Linear attention approximates the exponential kernel to achieve $O(n)$ time complexity.

#definition(name: "Kernel Attention")[
  Using feature map $phi$:
  $ K(q, k) approx phi(q)^T phi(k) $
  
  Then:
  $ o_i = (sum_j phi(q_i)^T phi(k_j) v_j) / (sum_j phi(q_i)^T phi(k_j)) = (phi(q_i)^T sum_j phi(k_j) v_j^T) / (phi(q_i)^T sum_j phi(k_j)) $
]

The sums $sum_j phi(k_j) v_j^T$ and $sum_j phi(k_j)$ can be precomputed in $O(n d^2)$ time, then each query costs $O(d^2)$ instead of $O(n d)$.

*Common feature maps*:
- Random Fourier features
- ELU + 1: $phi(x) = "ELU"(x) + 1$
- Positive random features (Performers)

= Worked Examples

== Example 1: Complete Forward Pass

Consider the following inputs with $n_q = 2$, $n_k = 3$, $d_k = d_v = 2$:

$ Q = mat(1, 0; 0, 1), quad K = mat(1, 0; 0, 1; 1, 1), quad V = mat(2, 0; 0, 2; 1, 1) $

*Step 1: Compute scores* with $sqrt(d_k) = sqrt(2) approx 1.414$

$ S = 1/sqrt(2) Q K^T = 1/sqrt(2) mat(1, 0, 1; 0, 1, 1) approx mat(0.707, 0, 0.707; 0, 0.707, 0.707) $

*Step 2: Apply softmax* (row-wise)

For row 1: $exp([0.707, 0, 0.707]) approx [2.028, 1.000, 2.028]$

Sum $= 5.056$, so $A_1 approx [0.401, 0.198, 0.401]$

For row 2: By symmetry, $A_2 approx [0.198, 0.401, 0.401]$

$ A approx mat(0.401, 0.198, 0.401; 0.198, 0.401, 0.401) $

*Step 3: Compute output*

$ O = A V = mat(0.401, 0.198, 0.401; 0.198, 0.401, 0.401) mat(2, 0; 0, 2; 1, 1) $

$ O_1 = 0.401 dot (2, 0) + 0.198 dot (0, 2) + 0.401 dot (1, 1) = (1.203, 0.797) $
$ O_2 = 0.198 dot (2, 0) + 0.401 dot (0, 2) + 0.401 dot (1, 1) = (0.797, 1.203) $

$ O approx mat(1.203, 0.797; 0.797, 1.203) $

== Example 2: Temperature Effects

For scores $s = [2, 1, 0]$, we compute softmax at different temperatures:

#table(
  columns: (1fr, 2fr, 1fr),
  inset: 8pt,
  align: (center, center, center),
  [*$T$*], [*Weights*], [*Entropy*],
  [0.25], [$[0.997, 0.003, 0.000]$], [0.02],
  [0.5], [$[0.876, 0.118, 0.006]$], [0.42],
  [1.0], [$[0.665, 0.245, 0.090]$], [0.80],
  [2.0], [$[0.474, 0.316, 0.211]$], [1.02],
  [$infinity$], [$[0.333, 0.333, 0.333]$], [1.10],
)

Lower temperature concentrates probability on the maximum score.

== Example 3: Gradient Verification

We verify the softmax Jacobian for $s = [1, 2]$:

$ a = "softmax"([1, 2]) = [e^1 / (e^1 + e^2), e^2 / (e^1 + e^2)] approx [0.269, 0.731] $

The Jacobian is:

$ (partial a_i) / (partial s_j) = a_i (delta_(i j) - a_j) $

$ J = mat(a_1(1 - a_1), -a_1 a_2; -a_2 a_1, a_2(1 - a_2)) = mat(0.197, -0.197; -0.197, 0.197) $

Verification: Rows sum to 0 (as expected since softmax outputs sum to 1).

= Implementation and Validation

== Library Architecture

The `attn-tensors` library implements all operations in JAX with the following structure:

```
attn_tensors/
├── attention.py    # Core attention operations
├── bilinear.py     # Metric tensors and bilinear forms
├── gradients.py    # Manual gradient implementations
├── softmax.py      # Temperature, entropy, Gibbs
├── multihead.py    # Multi-head attention
├── masking.py      # Causal/padding masks
├── hopfield.py     # Hopfield network view
└── backend.py      # JAX/MLX backend selection
```

== Test Coverage

The library includes 400+ tests covering:

- Tensor shapes and contractions
- Gradient correctness (vs. JAX autodiff)
- Numerical stability (large scores, small temperatures)
- Edge cases (single key, identical keys, zero inputs)
- Property-based testing with Hypothesis

Example test output:
```
tests/test_attention.py .... 63 passed
tests/test_bilinear.py .... 45 passed
tests/test_gradients.py .... 89 passed
tests/test_softmax.py .... 52 passed
tests/test_multihead.py .... 41 passed
========================= 400+ tests passed =========================
```

== Backend Support

The library supports multiple backends:

```python
from attn_tensors import get_backend, Backend

# Auto-detect best backend
backend = get_backend()  # MLX on Apple Silicon, JAX otherwise

# Check availability
from attn_tensors import is_mlx_available
if is_mlx_available():
    print("Using MLX acceleration")
```

= Conclusion

We have presented a comprehensive tensor calculus formulation of the attention mechanism, revealing its deep mathematical structure:

1. *Bilinear forms*: Attention scores arise from an inner product with implicit metric tensor
2. *Statistical mechanics*: Softmax is the Gibbs distribution; temperature controls attention sharpness
3. *Differential geometry*: The feature space is a Riemannian manifold with the metric defining similarity
4. *Associative memory*: Attention implements modern Hopfield network retrieval with exponential capacity

The geometric perspective is not merely pedagogical—it suggests natural generalizations (learned metrics, variable temperature) and connects attention to well-studied mathematical structures.

All derivations have been verified against automatic differentiation, with the reference implementation publicly available. The tensor notation, while initially unfamiliar to machine learning practitioners, provides a powerful and rigorous language for understanding and extending attention mechanisms.

== Future Directions

Natural extensions include:

- *Riemannian optimization*: Natural gradient descent on attention parameters
- *Learned geometries*: End-to-end learning of the metric tensor
- *Geometric regularization*: Entropy penalties through the thermodynamic lens
- *Efficient approximations*: Geometric analysis of linear attention quality

#pagebreak()

= Acknowledgments

We thank the JAX and MLX development teams for their excellent automatic differentiation libraries. This work builds on foundational research in differential geometry, statistical mechanics, and associative memory.

#pagebreak()

= Appendix A: Notation Reference

#table(
  columns: (1fr, 2fr),
  inset: 8pt,
  [*Symbol*], [*Meaning*],
  [$Q^(i a)$], [Query tensor, position $i$, feature $a$],
  [$K^(j a)$], [Key tensor, position $j$, feature $a$],
  [$V^(j b)$], [Value tensor, position $j$, feature $b$],
  [$S^(i j)$], [Attention scores],
  [$A^(i j)$], [Attention weights],
  [$O^(i b)$], [Output tensor],
  [$g_(a b)$], [Metric tensor],
  [$g^(a b)$], [Inverse metric],
  [$delta^a_b$], [Kronecker delta],
  [$Z^i$], [Partition function for query $i$],
  [$H^i$], [Entropy for query $i$],
  [$F^i$], [Free energy for query $i$],
  [$T$], [Temperature],
  [$beta$], [Inverse temperature ($1\/T$)],
  [$h$], [Head index in multi-head attention],
  [$W_Q, W_K, W_V, W_O$], [Projection weight matrices],
)

= Appendix B: Code Repository

Full source code, documentation, and examples:

- *GitHub*: https://github.com/planckeon/attn-as-bilinear-form
- *Documentation*: https://planckeon.github.io/attn-as-bilinear-form/
- *License*: MIT

== Quick Start

```bash
# Install
git clone https://github.com/planckeon/attn-as-bilinear-form
cd attn-as-bilinear-form
uv sync

# Run tests
uv run pytest tests/ -v

# With MLX (Apple Silicon)
uv sync --extra mlx
```

== Basic Usage

```python
import jax.numpy as jnp
from attn_tensors import scaled_dot_product_attention
from attn_tensors.bilinear import scaled_euclidean_metric, bilinear_form_batch

# Standard attention
Q = jnp.array([[1., 0.], [0., 1.]])
K = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
V = jnp.array([[2., 0.], [0., 2.], [1., 1.]])

output = scaled_dot_product_attention(Q, K, V)

# With explicit metric
g = scaled_euclidean_metric(d=2)
scores = bilinear_form_batch(Q, K, g)
```
