// Attention as Bilinear Form: A Physicist's Guide
// Document configuration

#set document(
  title: "Attention as Bilinear Form: A Physicist's Guide",
  author: "Tutorial Notes",
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Custom theorem environments
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

#let proposition(body) = {
  block(
    fill: rgb("#fff8e8"),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Proposition.* #body
  ]
}

#let example(body) = {
  block(
    stroke: rgb("#666") + 0.5pt,
    inset: 10pt,
    radius: 4pt,
    width: 100%,
  )[
    *Example.* #body
  ]
}

// Title page
#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold")[
    Attention as Bilinear Form
  ]
  #v(0.5cm)
  #text(size: 18pt)[
    A Physicist's Guide to Transformer Attention
  ]
  #v(2cm)
  #text(size: 12pt)[
    _Tensor Calculus, Statistical Mechanics, and Differential Geometry_
  ]
  #v(3cm)
  #text(size: 11pt)[
    Tutorial Notes \
    January 2025
  ]
]

#pagebreak()

// Table of contents
#outline(
  title: "Contents",
  indent: 2em,
  depth: 3,
)

#pagebreak()

= Introduction

The attention mechanism, introduced by Bahdanau et al. @bahdanau2014 and refined in the Transformer architecture by Vaswani et al. @vaswani2017, has become the foundation of modern deep learning. While typically presented in matrix notation optimized for implementation, the underlying mathematical structure reveals deep connections to classical physics and differential geometry.

This tutorial recasts the attention mechanism in the language of tensor calculus, making explicit the geometric and statistical mechanical structure hidden in the standard formulation. Our goals are:

1. *Tensor Calculus*: Express attention using index notation with proper contravariant/covariant indices and Einstein summation convention
2. *Bilinear Forms*: Show that attention scores arise from a bilinear form with an implicit metric tensor
3. *Statistical Mechanics*: Interpret softmax as the Gibbs/Boltzmann distribution from thermodynamics
4. *Differential Geometry*: Understand the feature space as a Riemannian manifold
5. *Gradients*: Derive backpropagation formulas in index notation and verify against autodiff

== Notation Conventions

Throughout this document, we use the following conventions:

#definition(name: "Index Notation")[
  - *Superscripts* denote contravariant (column vector) indices: $v^a$, $Q^(i a)$
  - *Subscripts* denote covariant (row vector/dual) indices: $g_(a b)$, $u_a$
  - *Einstein summation*: Repeated indices (one up, one down) are summed: $v^a u_a = sum_a v^a u_a$
  - *Kronecker delta*: $delta^a_b = cases(1 "if" a = b, 0 "otherwise")$
]

The key index labels we use:

#table(
  columns: (1fr, 2fr),
  inset: 8pt,
  [*Index*], [*Meaning*],
  [$i$], [Query sequence position ($n_q$ positions)],
  [$j$], [Key/value sequence position ($n_k$ positions)],
  [$a, b$], [Feature/embedding dimensions ($d_k$ or $d_v$)],
  [$h$], [Attention head index ($H$ heads)],
  [$mu, nu$], [Pattern index in Hopfield networks],
)

#pagebreak()

= Part I: Foundations

== Vectors and Dual Vectors

In physics, we distinguish between vectors and their duals (covectors). A vector $v^a$ lives in a vector space $V$, while a covector $u_a$ lives in the dual space $V^*$. The natural pairing between them is:

$ angle.l u, v angle.r = u_a v^a $

This is basis-independent. In a coordinate basis, this becomes the familiar dot product.

#definition(name: "Metric Tensor")[
  A *metric tensor* $g_(a b)$ is a symmetric, positive-definite $(0,2)$-tensor that defines an inner product on the vector space:
  
  $ angle.l u, v angle.r_g = u^a g_(a b) v^b $
  
  The metric allows us to:
  1. *Lower indices*: $v_a = g_(a b) v^b$ (convert vector to covector)
  2. *Raise indices*: $v^a = g^(a b) v_b$ (convert covector to vector)
  
  where $g^(a b)$ is the inverse metric satisfying $g^(a c) g_(c b) = delta^a_b$.
]

== Bilinear Forms

#definition(name: "Bilinear Form")[
  A *bilinear form* is a map $B: V times W -> RR$ that is linear in both arguments:
  
  $ B(alpha u + beta v, w) = alpha B(u, w) + beta B(v, w) $
  $ B(u, alpha v + beta w) = alpha B(u, v) + beta B(u, w) $
  
  In index notation with a matrix $M_(a b)$:
  $ B(u, v) = u^a M_(a b) v^b $
]

The connection to attention: the attention score between a query $q$ and key $k$ is precisely a bilinear form:

$ S = q^a g_(a b) k^b $

where the metric $g_(a b)$ encodes how we measure similarity in feature space.

== Standard Metrics

#example[
  *Euclidean metric*: $g_(a b) = delta_(a b)$ (identity matrix)
  
  This gives the standard dot product: $angle.l u, v angle.r = u^a delta_(a b) v^b = u^a v_a$
]

#example[
  *Scaled Euclidean metric*: $g_(a b) = 1/sqrt(d_k) delta_(a b)$
  
  This is precisely the metric implicit in scaled dot-product attention! The $1/sqrt(d_k)$ factor prevents dot products from growing too large in high dimensions.
]

#example[
  *Learned bilinear metric*: $g_(a b) = W^c_a W_(c b)$
  
  Parameterizing the metric as $W^T W$ ensures positive semi-definiteness. This generalizes standard attention to learnable similarity functions.
]

#pagebreak()

= Part II: The Attention Mechanism

== Attention as Tensor Contraction

The attention mechanism operates on three inputs:
- *Queries* $Q^(i a)$: What we're looking for (shape: $n_q times d_k$)
- *Keys* $K^(j a)$: What we're matching against (shape: $n_k times d_k$)
- *Values* $V^(j b)$: What we retrieve (shape: $n_k times d_v$)

The mechanism proceeds in three steps:

=== Step 1: Score Computation

Compute pairwise similarity between all queries and keys:

#definition(name: "Attention Scores")[
  $ S^(i j) = 1/sqrt(d_k) Q^(i a) K^(j a) $
  
  Or with explicit metric: $S^(i j) = Q^(i a) g_(a b) K^(j b)$
  
  where $g_(a b) = 1/sqrt(d_k) delta_(a b)$.
]

The contraction over the feature index $a$ computes a scalar similarity for each query-key pair. This is a *bilinear form* evaluated for all pairs.

In code (JAX with einsum):
```python
# S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
S = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

=== Step 2: Softmax Normalization

Convert scores to a probability distribution over keys:

#definition(name: "Attention Weights")[
  $ A^(i j) = (exp(S^(i j))) / (sum_k exp(S^(i k))) = (exp(S^(i j))) / Z^i $
  
  where $Z^i = sum_j exp(S^(i j))$ is the *partition function* for query $i$.
]

The softmax is applied row-wise: each query $i$ gets its own probability distribution over keys.

=== Step 3: Value Aggregation

Compute weighted sum of values:

#definition(name: "Attention Output")[
  $ O^(i b) = A^(i j) V^(j b) $
  
  This contracts over the key index $j$, producing an output for each query.
]

== Full Attention in One Equation

Combining all steps:

#theorem(name: "Scaled Dot-Product Attention")[
  $ O^(i b) = (exp(Q^(i a) g_(a c) K^(j c))) / (sum_k exp(Q^(i a) g_(a c) K^(k c))) V^(j b) $
  
  Or in matrix notation:
  $ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $
]

#pagebreak()

= Part III: Statistical Mechanics

The softmax function is not merely a normalization trick—it is the *Gibbs distribution* from statistical mechanics, revealing deep connections to thermodynamics.

== The Gibbs/Boltzmann Distribution

In statistical mechanics, a system with energy levels $E_j$ at temperature $T$ has probability:

#definition(name: "Gibbs Distribution")[
  $ P(j) = (exp(-E_j \/ T)) / Z, quad "where" Z = sum_j exp(-E_j \/ T) $
  
  - $T$: Temperature (controls distribution sharpness)
  - $Z$: Partition function (normalization)
  - $beta = 1\/T$: Inverse temperature
]

For attention, we identify:
- *Scores as negative energies*: $S^(i j) = -E^(i j)$ (higher score = lower energy = preferred)
- *Temperature*: Standard attention uses $T = 1$

Thus:
$ A^(i j) = (exp(S^(i j) \/ T)) / (sum_k exp(S^(i k) \/ T)) $

== Temperature Effects

The temperature parameter controls how "peaked" the attention distribution is:

#proposition[
  As temperature varies:
  - $T -> 0$: *Hard attention* (argmax). Attention concentrates on highest-scoring key.
  - $T = 1$: *Standard softmax*. Balanced distribution.
  - $T -> infinity$: *Uniform attention*. All keys weighted equally.
]

The $1/sqrt(d_k)$ scaling in standard attention can be interpreted as setting an effective temperature $T = sqrt(d_k)$ when using unscaled scores.

== Entropy and Information

The *Shannon entropy* of attention weights measures how diffuse or focused the attention is:

#definition(name: "Attention Entropy")[
  $ H^i = -sum_j A^(i j) log A^(i j) $
  
  - $H = 0$: Delta distribution (all attention on one key)
  - $H = log(n_k)$: Uniform distribution (equal attention on all keys)
]

The *normalized entropy* $H^i \/ log(n_k) in [0, 1]$ provides a scale-independent measure.

== Free Energy

The *free energy* combines energy and entropy:

#definition(name: "Free Energy")[
  $ F^i = -T log Z^i = -T log sum_j exp(S^(i j) \/ T) $
  
  At temperature $T$, the free energy satisfies:
  $ F = angle.l E angle.r - T dot H $
  
  where $angle.l E angle.r$ is the expected energy and $H$ is the entropy.
]

Minimizing free energy balances:
1. *Low energy*: Attend to high-scoring keys
2. *High entropy*: Spread attention broadly (regularization)

This provides a principled way to understand attention with temperature.

#pagebreak()

= Part IV: Differential Geometry

The feature space where queries and keys live can be understood as a Riemannian manifold, with the metric tensor defining geometry.

== Riemannian Metrics

#definition(name: "Riemannian Manifold")[
  A *Riemannian manifold* $(M, g)$ is a smooth manifold $M$ equipped with a metric tensor $g_(a b)(x)$ at each point $x in M$.
  
  The metric defines:
  - *Distances*: $d s^2 = g_(a b) d x^a d x^b$
  - *Angles*: $cos theta = (u^a g_(a b) v^b) / (||u||_g ||v||_g)$
  - *Volumes*: $d V = sqrt(det g) d x^1 ... d x^n$
]

In attention, the feature space $RR^(d_k)$ is (implicitly) a Riemannian manifold with metric $g_(a b) = 1/sqrt(d_k) delta_(a b)$.

== Christoffel Symbols and Covariant Derivatives

For a general metric $g_(a b)(x)$ that varies with position, we need *covariant derivatives* to properly differentiate tensors:

#definition(name: "Christoffel Symbols")[
  The *Christoffel symbols of the second kind* are:
  $ Gamma^c_(a b) = 1/2 g^(c d) (partial_a g_(b d) + partial_b g_(a d) - partial_d g_(a b)) $
]

#definition(name: "Covariant Derivative")[
  For a contravariant vector $v^b$:
  $ nabla_a v^b = partial_a v^b + Gamma^b_(a c) v^c $
  
  For a covariant vector $u_b$:
  $ nabla_a u_b = partial_a u_b - Gamma^c_(a b) u_c $
]

For the standard (constant) attention metric, $Gamma^c_(a b) = 0$ and covariant derivatives reduce to ordinary derivatives.

== Natural Gradient Descent

When optimizing over parameter space, the *Fisher information matrix* provides a natural Riemannian metric:

#definition(name: "Fisher Information")[
  $ F_(i j) = EE[(partial log p(x|theta)) / (partial theta^i) (partial log p(x|theta)) / (partial theta^j)] $
]

*Natural gradient descent* uses this metric:
$ Delta theta^i = -eta (F^(-1))^(i j) (partial L) / (partial theta^j) $

This is invariant under reparameterization and often converges faster than standard gradient descent.

#pagebreak()

= Part V: Gradient Derivations

We now derive the gradients for backpropagation through attention, using index notation throughout.

== Chain Rule in Index Notation

For a scalar loss $L$, the chain rule gives:
$ (partial L) / (partial Q^(k l)) = (partial L) / (partial O^(i b)) (partial O^(i b)) / (partial A^(m n)) (partial A^(m n)) / (partial S^(p q)) (partial S^(p q)) / (partial Q^(k l)) $

We compute each factor.

== Gradient Through Value Aggregation

Given $O^(i b) = A^(i j) V^(j b)$:

#proposition[
  $ (partial O^(i b)) / (partial A^(m n)) = delta^i_m delta^j_n V^(j b) = delta^i_m V^(n b) $
  
  $ (partial O^(i b)) / (partial V^(m n)) = A^(i j) delta^j_m delta^b_n = A^(i m) delta^b_n $
]

Therefore:
$ (partial L) / (partial A^(m n)) = (partial L) / (partial O^(m b)) V^(n b) $

In matrix form: $partial L \/ partial A = (partial L \/ partial O) V^T$

And:
$ (partial L) / (partial V^(m n)) = A^(i m) (partial L) / (partial O^(i n)) $

In matrix form: $partial L \/ partial V = A^T (partial L \/ partial O)$

== Gradient Through Softmax

The softmax Jacobian is:

#proposition[
  $ (partial A^(i j)) / (partial S^(m n)) = delta^i_m A^(i j) (delta^j_n - A^(i n)) $
]

The gradient through softmax becomes:

#theorem(name: "Softmax Gradient")[
  $ (partial L) / (partial S^(m n)) = A^(m n) ((partial L) / (partial A^(m n)) - sum_j A^(m j) (partial L) / (partial A^(m j))) $
  
  In compact form:
  $ (partial L) / (partial S) = A circle.tiny ((partial L) / (partial A) - "rowsum"(A circle.tiny (partial L) / (partial A))) $
  
  where $circle.tiny$ is elementwise multiplication.
]

== Gradient Through Score Computation

Given $S^(i j) = 1/sqrt(d_k) Q^(i a) K^(j a)$:

#proposition[
  $ (partial S^(i j)) / (partial Q^(k l)) = 1/sqrt(d_k) delta^i_k delta^a_l K^(j a) = 1/sqrt(d_k) delta^i_k K^(j l) $
  
  $ (partial S^(i j)) / (partial K^(k l)) = 1/sqrt(d_k) delta^j_k Q^(i l) $
]

Therefore:

#theorem(name: "Query Gradient")[
  $ (partial L) / (partial Q^(k l)) = 1/sqrt(d_k) (partial L) / (partial S^(k j)) K^(j l) $
  
  In matrix form: $partial L \/ partial Q = 1/sqrt(d_k) dot (partial L \/ partial S) K$
]

#theorem(name: "Key Gradient")[
  $ (partial L) / (partial K^(k l)) = 1/sqrt(d_k) (partial L) / (partial S^(i k)) Q^(i l) $
  
  In matrix form: $partial L \/ partial K = 1/sqrt(d_k) dot (partial L \/ partial S)^T Q$
]

== Complete Backward Pass

Combining all the pieces:

#theorem(name: "Attention Backward Pass")[
  Given upstream gradient $partial L \/ partial O$:
  
  1. $partial L \/ partial V = A^T (partial L \/ partial O)$
  
  2. $partial L \/ partial A = (partial L \/ partial O) V^T$
  
  3. $partial L \/ partial S = A circle.tiny (partial L \/ partial A - "rowsum"(A circle.tiny partial L \/ partial A))$
  
  4. $partial L \/ partial Q = 1/sqrt(d_k) (partial L \/ partial S) K$
  
  5. $partial L \/ partial K = 1/sqrt(d_k) (partial L \/ partial S)^T Q$
]

#pagebreak()

= Part VI: Multi-Head Attention

Multi-head attention runs multiple attention operations in parallel with different learned projections.

== Structure

#definition(name: "Multi-Head Attention")[
  For $H$ heads with projection matrices $W_Q^h, W_K^h, W_V^h, W_O^h$:
  
  $ Q^(h i a) = Q^(i b) W_Q^(h b a) $
  $ K^(h j a) = K^(j b) W_K^(h b a) $
  $ V^(h j c) = V^(j b) W_V^(h b c) $
  
  Per-head attention:
  $ S^(h i j) = 1/sqrt(d_k) Q^(h i a) K^(h j a) $
  $ A^(h i j) = "softmax"_j (S^(h i j)) $
  $ O^(h i c) = A^(h i j) V^(h j c) $
  
  Final output (sum over heads):
  $ Y^(i a) = O^(h i c) W_O^(h c a) $
]

The head index $h$ creates independent attention patterns, allowing the model to attend to different aspects of the input simultaneously.

== Head Diversity

Well-trained multi-head attention should have *diverse* heads that capture different relationships. We can measure diversity using:

$ "diversity" = 1 - "avg pairwise cosine similarity between heads" $

#pagebreak()

= Part VII: Attention Variants

== Causal Masking

For autoregressive models, we prevent position $i$ from attending to future positions $j > i$:

#definition(name: "Causal Mask")[
  $ M^(i j) = cases(1 "if" j <= i, 0 "otherwise") $
  
  Applied as: $S^(i j) <- S^(i j) + (1 - M^(i j)) dot (-infinity)$
]

== Learned Bilinear Attention

Instead of scaled dot-product, use a learned metric:

$ S^(i j) = Q^(i a) M^(a b) K^(j b) $

where $M^(a b)$ is a learnable parameter (or $M = W^T W$ for positive definiteness).

== Relative Position Attention

Add relative position information to keys:

$ S^(i j) = 1/sqrt(d_k) Q^(i a) (K^(j a) + R^((i-j) a)) $

where $R^(k a)$ is a learned embedding for relative position $k$.

#pagebreak()

= Part VIII: Attention as Kernel Regression

The attention mechanism can be viewed through the lens of kernel methods.

== Kernel Formulation

Define a kernel:
$ K(q, k) = exp((q dot k) / sqrt(d_k)) $

Then attention weights are:
$ A^(i j) = (K(q_i, k_j)) / (sum_k K(q_i, k_k)) $

And the output is:
$ o_i = (sum_j K(q_i, k_j) v_j) / (sum_j K(q_i, k_j)) $

This is exactly *Nadaraya-Watson kernel regression*!

== Linear Attention

For efficiency, we can approximate the kernel using feature maps $phi$:

$ K(q, k) approx phi(q)^T phi(k) $

This allows computing attention in $O(n)$ instead of $O(n^2)$:

$ o_i = (sum_j phi(q_i)^T phi(k_j) v_j) / (sum_j phi(q_i)^T phi(k_j)) = (phi(q_i)^T sum_j phi(k_j) v_j^T) / (phi(q_i)^T sum_j phi(k_j)) $

The sums can be precomputed once and reused for all queries.

#pagebreak()

= Part IX: Hopfield Networks and Attention

A remarkable connection exists between transformer attention and modern Hopfield networks @ramsauer2020.

== Classical Hopfield Networks

Classical Hopfield networks store patterns $xi_mu$ in a weight matrix:
$ W_(i j) = 1/N sum_mu xi_mu^i xi_mu^j $

The energy function is:
$ E(x) = -1/2 x^i W_(i j) x^j $

Fixed points (local minima) correspond to stored patterns.

*Problem*: Capacity scales only as $M approx 0.14 N$ patterns.

== Modern Hopfield Networks

Modern Hopfield networks @ramsauer2020 use an exponential energy:

#definition(name: "Modern Hopfield Energy")[
  $ E(x) = -"lse"(beta dot "patterns"^T x) + 1/2 ||x||^2 $
  
  where $"lse"(z) = log sum_mu exp(z_mu)$ is the log-sum-exp (smooth maximum).
]

The update rule is:

#theorem(name: "Hopfield Update = Attention")[
  $ x_"new" = "patterns"^T dot "softmax"(beta dot "patterns" dot x) $
  
  This is exactly the attention mechanism with:
  - Query: current state $x$
  - Keys: stored patterns
  - Values: stored patterns
  - Temperature: $1\/beta$
]

*Key insight*: Capacity scales *exponentially* as $M approx exp(d\/2)$!

#pagebreak()

= Part X: Worked Examples

== Example 1: 2-Query, 3-Key Attention

Consider:
$ Q = mat(1, 0; 0, 1), quad K = mat(1, 0; 0, 1; 1, 1), quad V = mat(2, 0; 0, 2; 1, 1) $

*Step 1: Scores* ($d_k = 2$, so $1/sqrt(d_k) = 1/sqrt(2) approx 0.707$)

$ S = 1/sqrt(2) Q K^T = 1/sqrt(2) mat(1, 0, 1; 0, 1, 1) approx mat(0.707, 0, 0.707; 0, 0.707, 0.707) $

*Step 2: Softmax* (row-wise)

For row 1: $[e^(0.707), e^0, e^(0.707)] approx [2.03, 1, 2.03]$, sum $= 5.06$

$ A_1 approx [0.401, 0.198, 0.401] $

Row 2 is symmetric: $A_2 approx [0.198, 0.401, 0.401]$

*Step 3: Output*

$ O = A V = mat(0.401, 0.198, 0.401; 0.198, 0.401, 0.401) mat(2, 0; 0, 2; 1, 1) = mat(1.203, 0.797; 0.797, 1.203) $

== Example 2: Temperature Effects

For scores $S = [2, 1, 0]$, attention weights at different temperatures:

#table(
  columns: (1fr, 2fr, 1fr),
  inset: 8pt,
  [*T*], [*Weights*], [*Entropy*],
  [0.5], [[0.88, 0.11, 0.01]], [0.42],
  [1.0], [[0.67, 0.24, 0.09]], [0.80],
  [2.0], [[0.47, 0.32, 0.21]], [1.02],
  [$infinity$], [[0.33, 0.33, 0.33]], [1.10],
)

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
  [$A^(i j)$], [Attention weights (probabilities)],
  [$O^(i b)$], [Output tensor],
  [$g_(a b)$], [Metric tensor],
  [$delta^a_b$], [Kronecker delta],
  [$Gamma^c_(a b)$], [Christoffel symbols],
  [$Z^i$], [Partition function for query $i$],
  [$H^i$], [Entropy for query $i$],
  [$T$], [Temperature],
  [$beta$], [Inverse temperature ($1\/T$)],
)

= Appendix B: Code Verification

All derivations in this document have been verified against JAX autodiff. The code is available in the accompanying `attn_tensors` Python package:

```python
from attn_tensors.gradients import verify_gradients
import jax.numpy as jnp

Q = jnp.array([[1., 0.], [0., 1.]])
K = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
V = jnp.array([[2., 0.], [0., 2.], [1., 1.]])

results = verify_gradients(Q, K, V)
print(results)  # {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True, 'all_correct': True}
```

#pagebreak()

#bibliography("refs.bib", style: "ieee")

