// Attention as Bilinear Form: A Physicist's Guide
// An accessible tutorial on transformer attention through tensor calculus

#set document(
  title: "Attention as Bilinear Form: A Physicist's Guide",
  author: "Baalateja Kataru",
  date: datetime(year: 2026, month: 1, day: 22),
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

#set par(justify: true)

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

#let intuition(body) = {
  block(
    stroke: (left: rgb("#22aa44") + 2pt),
    inset: (left: 10pt, y: 5pt),
    width: 100%,
  )[
    _Intuition._ #body
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
  #v(2cm)
  #text(size: 11pt)[
    Baalateja Kataru \
    #link("mailto:baalateja.k@gmail.com") \
    #v(0.5cm)
    January 2026
  ]
]

#pagebreak()

// Abstract
#align(center)[
  #text(size: 12pt, weight: "bold")[Abstract]
]

#block[
If you've ever wondered what's really going on inside a transformer, this tutorial is for you. We're going to look at the attention mechanism through a physicist's lens---using the language of tensor calculus, bilinear forms, and statistical mechanics. Don't worry if that sounds intimidating; we'll build up the concepts step by step.
]

#block[
The punchline? That innocent-looking formula $"Attention"(Q, K, V) = "softmax"(Q K^T \/ sqrt(d_k)) V$ hides beautiful mathematical structure: the score computation is a bilinear form with a hidden metric tensor, the softmax is actually the Gibbs distribution from thermodynamics, and the whole thing implements an associative memory network with exponential storage capacity!
]

#block[
A companion Python library `attn-tensors` implements everything we discuss, with 400+ tests verifying our gradient derivations against JAX autodiff. Code at: #link("https://github.com/planckeon/attn-as-bilinear-form").
]

#v(1em)

// Table of contents
#outline(
  title: "Contents",
  indent: 2em,
  depth: 2,
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
6. *Efficiency*: Understand Flash Attention and linear attention through the geometric lens

== Who Is This For?

This tutorial is aimed at:

- *ML practitioners* who want a deeper understanding of what attention is really doing
- *Physics students* curious about how their mathematical toolkit applies to deep learning
- *Researchers* looking for new perspectives on attention mechanisms

We assume familiarity with basic linear algebra and calculus. Prior exposure to index notation or differential geometry is helpful but not required---we'll introduce everything we need.

== Notation Conventions

Throughout this document, we use the following conventions:

#definition(name: "Index Notation")[
  - *Superscripts* denote contravariant (column vector) indices: $v^a$, $Q^(i a)$
  - *Subscripts* denote covariant (row vector/dual) indices: $g_(a b)$, $u_a$
  - *Einstein summation*: Repeated indices (one up, one down) are summed: $v^a u_a = sum_a v^a u_a$
  - *Kronecker delta*: $delta^a_b = cases(1 "if" a = b, 0 "otherwise")$
]

#intuition[
  Think of superscripts as "column vectors" and subscripts as "row vectors." When you contract (sum over) a matching pair, you're doing a dot product. The Einstein convention just saves us from writing summation signs everywhere.
]

The key index labels we use:

#table(
  columns: (1fr, 2fr),
  inset: 8pt,
  [*Index*], [*Meaning*],
  [$i$], [Query sequence position ($n_q$ positions)],
  [$j, k$], [Key/value sequence position ($n_k$ positions)],
  [$a, b, c$], [Feature/embedding dimensions ($d_k$ or $d_v$)],
  [$h$], [Attention head index ($H$ heads)],
  [$mu, nu$], [Pattern index in Hopfield networks],
)

#pagebreak()

= Part I: Foundations

== Vectors and Dual Vectors

In physics, we distinguish between vectors and their duals (covectors). A vector $v^a$ lives in a vector space $V$, while a covector $u_a$ lives in the dual space $V^*$. The natural pairing between them is:

$ chevron.l u, v chevron.r = u_a v^a $

This is basis-independent. In a coordinate basis, this becomes the familiar dot product.

#intuition[
  In machine learning terms: a vector $v^a$ is a column vector, and a covector $u_a$ is a row vector. Their pairing $u_a v^a$ is just matrix multiplication of a row times a column, giving a scalar.
]

#definition(name: "Metric Tensor")[
  A *metric tensor* $g_(a b)$ is a symmetric, positive-definite $(0,2)$-tensor that defines an inner product on the vector space:
  
  $ chevron.l u, v chevron.r_g = u^a g_(a b) v^b $
  
  The metric allows us to:
  1. *Lower indices*: $v_a = g_(a b) v^b$ (convert vector to covector)
  2. *Raise indices*: $v^a = g^(a b) v_b$ (convert covector to vector)
  
  where $g^(a b)$ is the inverse metric satisfying $g^(a c) g_(c b) = delta^a_b$.
]

#remark[
  The metric tells us how to measure distances and angles in our space. Different metrics lead to different notions of "similarity"---this will be crucial for understanding attention.
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

#intuition[
  You can think of the metric $g_(a b)$ as answering the question: "How should I weight different features when computing similarity?" The standard dot product weights all features equally, but we could do something more sophisticated.
]

== Standard Metrics

#example(name: "Euclidean Metric")[
  $ g_(a b) = delta_(a b) $ (identity matrix)
  
  This gives the standard dot product: $chevron.l u, v chevron.r = u^a delta_(a b) v^b = u^a v_a$
]

#example(name: "Scaled Euclidean Metric")[
  $ g_(a b) = 1/sqrt(d_k) delta_(a b) $
  
  This is precisely the metric implicit in scaled dot-product attention! The $1/sqrt(d_k)$ factor prevents dot products from growing too large in high dimensions.
]

#example(name: "Learned Bilinear Metric")[
  $ g_(a b) = W^c_a W_(c b) = (W^T W)_(a b) $
  
  Parameterizing the metric as $W^T W$ ensures positive semi-definiteness. This generalizes standard attention to learnable similarity functions.
]

#remark[
  The scaling $1/sqrt(d_k)$ has a statistical interpretation: if $q^a$ and $k^a$ are i.i.d. with zero mean and unit variance, then $"Var"(q^a k_a) = d_k$. The scaling normalizes the variance to 1, keeping the softmax in a good operating regime. This is the "variance explosion" problem that the $sqrt(d_k)$ scaling solves.
]

== Einstein Summation (Einsum)

Now that we've introduced index notation, let's talk about how to implement it in code. The `einsum` function, available in NumPy, JAX, PyTorch, and other libraries, directly translates index notation to efficient tensor operations.

#definition(name: "Einstein Summation Convention")[
  In the *Einstein summation convention*, repeated indices are implicitly summed:
  
  $ C^(i k) = A^(i j) B_j^(space k) quad arrow.l.r.double quad C_(i k) = sum_j A_(i j) B_(j k) $
  
  In code: `C = einsum('ij,jk->ik', A, B)`
]

The einsum string has a simple grammar:

- *Input specification* (left of `→`): Comma-separated indices for each input tensor
- *Output specification* (right of `→`): Indices in the output
- *Repeated indices* not in output are summed over

#intuition[
  Think of einsum indices as a way to label the dimensions of tensors. When the same label appears in multiple places, those dimensions are "paired up" for multiplication. When a label is missing from the output, that dimension is summed over.
]

#example(name: "Basic Einsum Patterns")[
  Common operations in einsum:
  
  #table(
    columns: (1fr, 1fr, 1fr),
    align: (left, left, left),
    stroke: 0.5pt,
    inset: 5pt,
    [*Operation*], [*Einsum*], [*Index Notation*],
    [Dot product], [`'a,a->'`], [$s = u^a v_a$],
    [Outer product], [`'a,b->ab'`], [$M_(a b) = u_a v_b$],
    [Matrix multiply], [`'ij,jk->ik'`], [$C^(i k) = A^(i j) B_j^(space k)$],
    [Transpose], [`'ij->ji'`], [$B_(j i) = A_(i j)$],
    [Trace], [`'ii->'`], [$s = A^i_i$],
    [Batch matmul], [`'bij,bjk->bik'`], [$C^(b i k) = A^(b i j) B^(b)_j^(space k)$],
  )
]

=== Einsum for Attention

The attention mechanism maps beautifully to einsum. Here are the key operations:

*Attention scores* (query-key dot product):
$ S^(i j) = Q^(i a) K^(j a) \/ sqrt(d_k) $

```python
S = einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
```

*Attention output* (weighted sum of values):
$ O^(i b) = A^(i j) V^(j b) $

```python
O = einsum('ij,jb->ib', A, V)
```

*With bilinear metric*:
$ S^(i j) = Q^(i a) g_(a b) K^(j b) $

```python
S = einsum('ia,ab,jb->ij', Q, g, K)
```

=== Multi-Head Einsum

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

#intuition[
  Einsum makes the summation indices explicit. When you see `'hia,hja->hij'`, you immediately know that `a` (the feature dimension) is being summed over, while `h`, `i`, `j` are preserved. This is exactly what the index notation tells us: $S^(h i j) = sum_a Q^(h i a) K^(h j a)$.
]

#remark[
  Einsum is not just notation---it's often faster than explicit loops and reshapes because it avoids creating intermediate arrays. The library can optimize the contraction order and fuse operations. For more on einsum, see Sankalp's excellent tutorial "Shape Rotation 101" @sankalp2024.
]

#pagebreak()

= Part II: The Attention Mechanism

Now let's see how all this machinery applies to attention.

== Attention as Tensor Contraction

The attention mechanism operates on three inputs:
- *Queries* $Q^(i a)$: What we're looking for (shape: $n_q times d_k$)
- *Keys* $K^(j a)$: What we're matching against (shape: $n_k times d_k$)
- *Values* $V^(j b)$: What we retrieve (shape: $n_k times d_v$)

#intuition[
  Think of it like a library search:
  - The *query* is your question ("I want books about physics")
  - The *keys* are the book titles/descriptions (what you match against)
  - The *values* are the actual book contents (what you get back)
]

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

#intuition[
  The output for each query is a weighted average of the values, where the weights are determined by how well each key matches the query. High-scoring keys contribute more to the output.
]

== Full Attention in One Equation

Combining all steps:

#theorem(name: "Scaled Dot-Product Attention")[
  $ O^(i b) = (exp(Q^(i a) g_(a c) K^(j c))) / (sum_k exp(Q^(i a) g_(a c) K^(k c))) V^(j b) $
  
  Or in matrix notation:
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

#pagebreak()

= Part III: Statistical Mechanics

The softmax function is not merely a normalization trick---it is the *Gibbs distribution* from statistical mechanics, revealing deep connections to thermodynamics.

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

#intuition[
  In physics, systems prefer low-energy states. In attention, we prefer high-scoring keys. The connection: if we define energy as negative score, then high scores become low energy, and softmax gives us the equilibrium distribution!
]

== Temperature Effects

The temperature parameter controls how "peaked" the attention distribution is:

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

The $1/sqrt(d_k)$ scaling in standard attention can be interpreted as setting an effective temperature $T = sqrt(d_k)$ when using unscaled scores.

```python
def attention_temperature(Q, K, V, temperature=1.0):
    """Attention with explicit temperature control."""
    d_k = Q.shape[-1]
    scores = jnp.einsum('ia,ja->ij', Q, K) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores / temperature, axis=-1)
    return jnp.einsum('ij,jb->ib', weights, V)
```

== Entropy and Information

The *Shannon entropy* of attention weights measures how diffuse or focused the attention is:

#definition(name: "Attention Entropy")[
  $ H^i = -sum_j A^(i j) log A^(i j) $
  
  - $H = 0$: Delta distribution (all attention on one key)
  - $H = log(n_k)$: Uniform distribution (equal attention on all keys)
]

The *normalized entropy* $H^i \/ log(n_k) in [0, 1]$ provides a scale-independent measure.

#intuition[
  Low entropy = focused attention (the model is "confident" about which keys to attend to). High entropy = diffuse attention (the model is spreading its attention broadly). Both can be useful depending on the task!
]

== Free Energy

The *free energy* combines energy and entropy:

#definition(name: "Free Energy")[
  $ F^i = -T log Z^i = -T log sum_j exp(S^(i j) \/ T) $
  
  At temperature $T$, the free energy satisfies:
  $ F = chevron.l E chevron.r - T dot H $
  
  where $chevron.l E chevron.r$ is the expected energy and $H$ is the entropy.
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

For the standard (constant) attention metric, $Gamma^c_(a b) = 0$ and covariant derivatives reduce to ordinary derivatives. This is why we can get away without differential geometry in standard attention---but learned metrics would need these tools!

== Natural Gradient Descent

When optimizing over parameter space, the *Fisher information matrix* provides a natural Riemannian metric:

#definition(name: "Fisher Information")[
  $ F_(i j) = EE[(partial log p(x|theta)) / (partial theta^i) (partial log p(x|theta)) / (partial theta^j)] $
]

*Natural gradient descent* uses this metric:
$ Delta theta^i = -eta (F^(-1))^(i j) (partial L) / (partial theta^j) $

#proposition(name: "Reparameterization Invariance")[
  Natural gradient descent gives the same update in parameter space regardless of how we parameterize the model. Standard gradient descent does not have this property.
]

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

#lemma(name: "Value Aggregation Jacobian")[
  $ (partial O^(i b)) / (partial A^(m n)) = delta^i_m V^(n b) $
  $ (partial O^(i b)) / (partial V^(m n)) = A^(i m) delta^b_n $
]

#proof[
  For the first:
  $ (partial O^(i b)) / (partial A^(m n)) = (partial) / (partial A^(m n)) (A^(i j) V^(j b)) = delta^i_m delta^j_n V^(j b) = delta^i_m V^(n b) $
  
  For the second:
  $ (partial O^(i b)) / (partial V^(m n)) = (partial) / (partial V^(m n)) (A^(i j) V^(j b)) = A^(i j) delta^j_m delta^b_n = A^(i m) delta^b_n $
]

Therefore:
$ (partial L) / (partial A^(m n)) = (partial L) / (partial O^(m b)) V^(n b) $

In matrix form: $partial L \/ partial A = (partial L \/ partial O) V^T$

And:
$ (partial L) / (partial V^(m n)) = A^(i m) (partial L) / (partial O^(i n)) $

In matrix form: $partial L \/ partial V = A^T (partial L \/ partial O)$

== Gradient Through Softmax

The softmax Jacobian is the trickiest part:

#lemma(name: "Softmax Jacobian")[
  For $A^(i j) = exp(S^(i j)) \/ sum_k exp(S^(i k))$:
  
  $ (partial A^(i j)) / (partial S^(m n)) = delta^i_m A^(i j) (delta^j_n - A^(i n)) $
]

#proof[
  The softmax for row $i$ depends only on scores in row $i$, so $delta^i_m$ ensures we're in the same row.
  
  For the diagonal case ($j = n$):
  $ (partial A^(i j)) / (partial S^(i j)) = (exp(S^(i j)) dot Z - exp(S^(i j)) dot exp(S^(i j))) / Z^2 = A^(i j) - (A^(i j))^2 = A^(i j)(1 - A^(i j)) $
  
  For the off-diagonal case ($j != n$):
  $ (partial A^(i j)) / (partial S^(i n)) = (0 dot Z - exp(S^(i j)) dot exp(S^(i n))) / Z^2 = -A^(i j) A^(i n) $
  
  Combining: $(partial A^(i j)) / (partial S^(i n)) = A^(i j)(delta^j_n - A^(i n))$
]

The gradient through softmax becomes:

#theorem(name: "Softmax Gradient")[
  $ (partial L) / (partial S^(m n)) = A^(m n) ((partial L) / (partial A^(m n)) - sum_j A^(m j) (partial L) / (partial A^(m j))) $
  
  In compact form:
  $ (partial L) / (partial S) = A circle.tiny ((partial L) / (partial A) - "rowsum"(A circle.tiny (partial L) / (partial A))) $
  
  where $circle.tiny$ is elementwise multiplication.
]

#intuition[
  The subtraction of the weighted sum is crucial---it ensures the gradient respects the constraint that attention weights sum to 1. If we increase one weight, others must decrease.
]

== Gradient Through Score Computation

Given $S^(i j) = 1/sqrt(d_k) Q^(i a) K^(j a)$:

#lemma(name: "Score Jacobian")[
  $ (partial S^(i j)) / (partial Q^(k l)) = 1/sqrt(d_k) delta^i_k K^(j l) $
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

#pagebreak()

= Part VI: Multi-Head Attention

Multi-head attention runs multiple attention operations in parallel with different learned projections.

== Structure

#definition(name: "Multi-Head Attention")[
  For $H$ heads with projection matrices $W_Q^h, W_K^h, W_V^h, W_O^h$:
  
  *Projections* (introducing head index $h$):
  $ Q^(h i a) = X^(i b) W_Q^(h b a) $
  $ K^(h j a) = X^(j b) W_K^(h b a) $
  $ V^(h j c) = X^(j b) W_V^(h b c) $
  
  Per-head attention:
  $ S^(h i j) = 1/sqrt(d_k) Q^(h i a) K^(h j a) $
  $ A^(h i j) = "softmax"_j (S^(h i j)) $
  $ O^(h i c) = A^(h i j) V^(h j c) $
  
  Final output (sum over heads and head dimension):
  $ Y^(i d) = O^(h i c) W_O^(h c d) $
]

#intuition[
  Each head can learn to attend to different aspects of the input. One head might focus on syntax, another on semantics, another on nearby tokens. The output projection $W_O$ learns to combine these perspectives.
]

== Geometric Interpretation

Each head projects to a different subspace:
$ Q_h = X W_Q^h in RR^(n times d_k) $

With $d_k = d_"model" \/ H$, each head operates on a $d_k$-dimensional subspace of the full $d_"model"$-dimensional space.

Different heads can specialize:
- *Syntactic heads*: Attend to grammatical structure
- *Semantic heads*: Attend to meaning similarity  
- *Positional heads*: Attend to relative positions

== Head Diversity

Well-trained multi-head attention should have *diverse* heads that capture different relationships. We can measure diversity using:

$ "diversity" = 1 - "avg pairwise cosine similarity between head attention patterns" $

#pagebreak()

= Part VII: Attention Variants

== Causal Masking

For autoregressive models, we prevent position $i$ from attending to future positions $j > i$:

#definition(name: "Causal Mask")[
  $ M^(i j) = cases(1 "if" j <= i, 0 "otherwise") $
  
  Applied as: $S^(i j) <- S^(i j) + (1 - M^(i j)) dot (-infinity)$
]

After softmax, $exp(-infinity) = 0$, so future positions get zero weight.

== Learned Bilinear Attention

Instead of scaled dot-product, use a learned metric:

$ S^(i j) = Q^(i a) M^(a b) K^(j b) $

where $M^(a b)$ is a learnable parameter (or $M = W^T W$ for positive definiteness).

== Relative Position Attention

Add relative position information to keys:

$ S^(i j) = 1/sqrt(d_k) Q^(i a) (K^(j a) + R^((i-j) a)) $

where $R^(k a)$ is a learned embedding for relative position $k$.

#pagebreak()

= Part VIII: Hopfield Networks and Attention

A remarkable connection exists between transformer attention and modern Hopfield networks.

== Classical Hopfield Networks

Classical Hopfield networks store patterns $xi_mu$ in a weight matrix:
$ W_(i j) = 1/N sum_mu xi_mu^i xi_mu^j $

The energy function is:
$ E(x) = -1/2 x^i W_(i j) x^j $

Fixed points (local minima) correspond to stored patterns.

*Problem*: Capacity scales only as $M approx 0.14 N$ patterns---not great!

== Modern Hopfield Networks

Modern Hopfield networks use an exponential energy:

#definition(name: "Modern Hopfield Energy")[
  $ E(xi) = -"lse"(beta dot K xi) + 1/2 ||xi||^2 + "const" $
  
  where $"lse"(z) = log sum_mu exp(z_mu)$ is the log-sum-exp (smooth maximum).
]

The update rule is:

#theorem(name: "Hopfield Update = Attention")[
  $ xi^("new") = V^T "softmax"(beta K xi) $
  
  This is exactly the attention mechanism with:
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

The key is the exponential separation provided by softmax: even small differences in scores lead to large differences in weights.

#intuition[
  This is why transformers are so powerful! Each attention layer is essentially a Hopfield network that can store and retrieve from an exponentially large pattern library. The keys are the "stored patterns" and the query retrieves a weighted combination of values.
]

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  [*Attention*], [*Hopfield*],
  [Query $q$], [Pattern to retrieve],
  [Keys $K$], [Stored patterns],
  [Values $V$], [Pattern outputs],
  [Softmax], [Update rule],
  [Output $o$], [Retrieved pattern],
)

#pagebreak()

= Part IX: Efficient Attention

Standard attention has $O(n^2)$ complexity, which is problematic for long sequences. Let's look at two major approaches to efficiency.

== Flash Attention

Flash Attention achieves $O(n)$ memory (instead of $O(n^2)$) by computing attention block-wise and avoiding storage of the full attention matrix.

=== The Key Insight: Online Softmax

Softmax can be computed incrementally without storing all values:

#proposition(name: "Online Softmax")[
  Given running maximum $m$ and sum of exponentials $ell$, we can incorporate new elements:
  
  $ m' = max(m, max(x_"new")) $
  $ ell' = ell dot exp(m - m') + sum exp(x_"new" - m') $
]

This numerical trick is the foundation of Flash Attention.

=== Block-wise Computation

#theorem(name: "Flash Attention Algorithm")[
  Divide $Q, K, V$ into blocks of size $B$. For each query block $Q_i$:
  
  1. Initialize $O_i = 0$, $ell_i = 0$, $m_i = -infinity$
  
  2. For each key-value block $(K_j, V_j)$:
     - Compute block scores: $S_(i j) = Q_i K_j^T \/ sqrt(d_k)$
     - Update running max: $m_"new" = max(m_i, "rowmax"(S_(i j)))$
     - Rescale previous: $O_i <- O_i dot exp(m_i - m_"new")$
     - Accumulate: $O_i <- O_i + exp(S_(i j) - m_"new") V_j$
     - Update normalization: $ell_i <- ell_i dot exp(m_i - m_"new") + "rowsum"(exp(S_(i j) - m_"new"))$
     - Update: $m_i <- m_"new"$
  
  3. Normalize: $O_i <- O_i \/ ell_i$
]

*Complexity comparison:*
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

*Common feature maps:*
- Random Fourier features
- ELU + 1: $phi(x) = "ELU"(x) + 1$
- Positive random features (Performers)

#intuition[
  The catch? Linear attention is an approximation. It loses some expressiveness compared to full softmax attention. The trade-off between efficiency and quality depends on your application.
]

#pagebreak()

= Part X: Worked Examples

== Example 1: 2-Query, 3-Key Attention

Consider:
$ Q = mat(1, 0; 0, 1), quad K = mat(1, 0; 0, 1; 1, 1), quad V = mat(2, 0; 0, 2; 1, 1) $

*Step 1: Scores* ($d_k = 2$, so $1/sqrt(d_k) = 1/sqrt(2) approx 0.707$)

$ S = 1/sqrt(2) Q K^T = 1/sqrt(2) mat(1, 0, 1; 0, 1, 1) approx mat(0.707, 0, 0.707; 0, 0.707, 0.707) $

*Step 2: Softmax* (row-wise)

For row 1: $exp([0.707, 0, 0.707]) approx [2.028, 1.000, 2.028]$

Sum $= 5.056$, so $A_1 approx [0.401, 0.198, 0.401]$

Row 2 is symmetric: $A_2 approx [0.198, 0.401, 0.401]$

$ A approx mat(0.401, 0.198, 0.401; 0.198, 0.401, 0.401) $

*Step 3: Output*

$ O = A V = mat(0.401, 0.198, 0.401; 0.198, 0.401, 0.401) mat(2, 0; 0, 2; 1, 1) $

$ O_1 = 0.401 dot (2, 0) + 0.198 dot (0, 2) + 0.401 dot (1, 1) = (1.203, 0.797) $
$ O_2 = 0.198 dot (2, 0) + 0.401 dot (0, 2) + 0.401 dot (1, 1) = (0.797, 1.203) $

$ O approx mat(1.203, 0.797; 0.797, 1.203) $

== Example 2: Temperature Effects

For scores $S = [2, 1, 0]$, attention weights at different temperatures:

#table(
  columns: (1fr, 2fr, 1fr),
  inset: 8pt,
  align: (center, center, center),
  [*T*], [*Weights*], [*Entropy*],
  [0.25], [$[0.997, 0.003, 0.000]$], [0.02],
  [0.5], [$[0.876, 0.118, 0.006]$], [0.42],
  [1.0], [$[0.665, 0.245, 0.090]$], [0.80],
  [2.0], [$[0.474, 0.316, 0.211]$], [1.02],
  [$infinity$], [$[0.333, 0.333, 0.333]$], [1.10],
)

Lower temperature concentrates probability on the maximum score.

== Example 3: Softmax Jacobian Verification

For $s = [1, 2]$:

$ a = "softmax"([1, 2]) = [e^1 / (e^1 + e^2), e^2 / (e^1 + e^2)] approx [0.269, 0.731] $

The Jacobian is:

$ (partial a_i) / (partial s_j) = a_i (delta_(i j) - a_j) $

$ J = mat(a_1(1 - a_1), -a_1 a_2; -a_2 a_1, a_2(1 - a_2)) = mat(0.197, -0.197; -0.197, 0.197) $

Note: Rows sum to 0 (as expected since softmax outputs sum to 1).

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
  [$g^(a b)$], [Inverse metric tensor],
  [$delta^a_b$], [Kronecker delta],
  [$Gamma^c_(a b)$], [Christoffel symbols],
  [$Z^i$], [Partition function for query $i$],
  [$H^i$], [Entropy for query $i$],
  [$F^i$], [Free energy for query $i$],
  [$T$], [Temperature],
  [$beta$], [Inverse temperature ($1\/T$)],
  [$h$], [Head index in multi-head attention],
  [$W_Q, W_K, W_V, W_O$], [Projection weight matrices],
)

= Appendix B: The attn-tensors Library

All derivations in this document have been verified against JAX autodiff. The code is available in the accompanying `attn_tensors` Python package.

== Library Architecture

```
attn_tensors/
├── attention.py    # Core attention operations
├── bilinear.py     # Metric tensors and bilinear forms
├── einsum.py       # Einstein summation utilities
├── gradients.py    # Manual gradient implementations
├── softmax.py      # Temperature, entropy, Gibbs
├── multihead.py    # Multi-head attention
├── masking.py      # Causal/padding masks
├── hopfield.py     # Hopfield network view
└── backend.py      # JAX/MLX backend selection
```

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

== Gradient Verification

```python
from attn_tensors.gradients import verify_gradients
import jax.numpy as jnp

Q = jnp.array([[1., 0.], [0., 1.]])
K = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
V = jnp.array([[2., 0.], [0., 2.], [1., 1.]])

results = verify_gradients(Q, K, V)
print(results)  # {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True, 'all_correct': True}
```

== Backend Support

```python
from attn_tensors import get_backend, Backend

# Auto-detect best backend
backend = get_backend()  # MLX on Apple Silicon, JAX otherwise

# Check availability
from attn_tensors import is_mlx_available
if is_mlx_available():
    print("Using MLX acceleration")
```

#pagebreak()

= Appendix C: Further Reading

For those who want to dive deeper:

*Original Papers:*
- Vaswani et al., "Attention Is All You Need" (2017) - The Transformer paper
- Ramsauer et al., "Hopfield Networks is All You Need" (2020) - The Hopfield connection
- Dao et al., "FlashAttention" (2022) - Efficient attention

*Mathematical Background:*
- Wald, "General Relativity" - Tensor calculus and differential geometry
- Amari, "Information Geometry" - Fisher information and natural gradients
- Kardar, "Statistical Physics of Fields" - Statistical mechanics

*Code and Documentation:*
- GitHub: https://github.com/planckeon/attn-as-bilinear-form
- Documentation: https://planckeon.github.io/attn-as-bilinear-form/

*Einsum Resources:*
- Sankalp, "Shape Rotation 101: An Intro to Einsum and Jax Transformers" - Excellent practical einsum tutorial
- Alex Riley, "A basic introduction to NumPy's einsum" - Clear einsum fundamentals
- Tim Rocktäschel, "Einstein Summation in Numpy" - Einsum internals

#pagebreak()

#bibliography("refs.bib", style: "ieee")
