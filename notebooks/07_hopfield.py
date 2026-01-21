# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax",
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 7. Hopfield Networks and Attention

        This notebook explores the deep connection between **modern Hopfield networks**
        and **attention mechanisms**.

        ## Key Insight (Ramsauer et al., 2020)

        The attention update rule is *exactly* the update rule of a modern Hopfield network:

        $$x_{\text{new}} = V^T \cdot \text{softmax}(\beta \cdot K \cdot x)$$

        where:
        - $x$ is the query (current state)
        - $K$ are the keys (stored patterns)
        - $V$ are the values (retrieved patterns)
        - $\beta = 1/\sqrt{d_k}$ is the inverse temperature
        """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import jax
    import jax.random as random
    import matplotlib.pyplot as plt

    return jax, jnp, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Classical vs Modern Hopfield Networks

        | Property | Classical | Modern |
        |----------|-----------|--------|
        | State space | Binary $\{-1, +1\}^N$ | Continuous $\mathbb{R}^d$ |
        | Energy | Quadratic | Log-sum-exp |
        | Capacity | $\sim 0.14N$ | $\sim \exp(d)$ |
        | Update | Sign function | Softmax |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Modern Hopfield Energy

        The energy function for modern Hopfield networks is:

        $$E(x) = -\text{lse}(\beta \cdot K^T x) + \frac{1}{2}\|x\|^2 + \text{const}$$

        where $\text{lse}(z) = \log \sum_\mu \exp(z_\mu)$ is the log-sum-exp (smooth maximum).

        Energy is minimized when $x$ aligns with one of the stored patterns (rows of $K$).
        """
    )
    return


@app.cell
def _(jnp):
    def modern_hopfield_energy(state, patterns, beta=1.0):
        """
        Modern Hopfield network energy.

        E(x) = -lse(beta * patterns @ x) + 0.5 * ||x||^2

        Args:
            state: Current state x of shape (d,)
            patterns: Stored patterns of shape (M, d)
            beta: Inverse temperature
        """
        # Similarities: s_mu = patterns_mu @ x
        similarities = jnp.einsum("ma,a->m", patterns, state)

        # Log-sum-exp (numerically stable)
        max_sim = jnp.max(beta * similarities)
        lse = max_sim + jnp.log(jnp.sum(jnp.exp(beta * similarities - max_sim)))

        # Energy
        return -lse / beta + 0.5 * jnp.sum(state**2)

    return (modern_hopfield_energy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Hopfield Update = Attention

        The update rule that minimizes energy is:

        $$x_{\text{new}} = K^T \cdot \text{softmax}(\beta \cdot K \cdot x)$$

        Compare to attention:
        $$\text{output} = V^T \cdot \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)$$

        They are the same operation!
        - Query $Q$ = current state $x$
        - Keys $K$ = stored patterns for similarity
        - Values $V$ = patterns to retrieve (can equal $K$)
        """
    )
    return


@app.cell
def _(jnp):
    def softmax(x):
        """Numerically stable softmax."""
        x_stable = x - jnp.max(x)
        exp_x = jnp.exp(x_stable)
        return exp_x / jnp.sum(exp_x)

    def hopfield_update(state, patterns, beta=1.0):
        """
        Modern Hopfield update rule.

        x_new = patterns^T @ softmax(beta * patterns @ x)

        This is exactly attention with Q=x, K=V=patterns!
        """
        # Similarities (attention scores)
        similarities = jnp.einsum("ma,a->m", patterns, state)

        # Softmax (attention weights)
        weights = softmax(beta * similarities)

        # Weighted combination (attention output)
        new_state = jnp.einsum("m,ma->a", weights, patterns)

        return new_state, weights

    return hopfield_update, softmax


@app.cell
def _(jnp, random):
    # Create some stored patterns
    key = random.PRNGKey(42)
    d = 8  # Pattern dimension
    M = 4  # Number of patterns

    # Random orthogonal-ish patterns
    patterns = random.normal(key, (M, d))
    # Normalize for easier visualization
    patterns = patterns / jnp.linalg.norm(patterns, axis=-1, keepdims=True)

    print("Stored Patterns (M x d):")
    print(patterns)
    print()
    print("Pattern norms:", jnp.linalg.norm(patterns, axis=-1))
    return M, d, key, patterns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pattern Retrieval

        Given a noisy query, the Hopfield network retrieves the closest stored pattern:
        """
    )
    return


@app.cell
def _(hopfield_update, jnp, patterns, random):
    def hopfield_retrieve(query, patterns, beta=1.0, max_iters=20, tol=1e-6):
        """Iterate Hopfield update until convergence."""
        state = query.copy()
        history = [state]

        for i in range(max_iters):
            new_state, weights = hopfield_update(state, patterns, beta)

            if jnp.max(jnp.abs(new_state - state)) < tol:
                break

            state = new_state
            history.append(state)

        return state, history, weights

    # Create a noisy version of pattern 0
    key2 = random.PRNGKey(123)
    noise = random.normal(key2, (patterns.shape[1],)) * 0.3
    noisy_query = patterns[0] + noise
    noisy_query = noisy_query / jnp.linalg.norm(noisy_query)

    print("Testing pattern retrieval:")
    print()
    print("Original pattern 0:", patterns[0][:4], "...")
    print("Noisy query:       ", noisy_query[:4], "...")
    print()

    # Retrieve
    retrieved, history, final_weights = hopfield_retrieve(noisy_query, patterns, beta=5.0)

    print(f"Converged in {len(history)} iterations")
    print()
    print("Retrieved:         ", retrieved[:4], "...")
    print()
    print("Final attention weights:", final_weights)
    print("(Should be peaked on pattern 0)")

    # Check similarity to each pattern
    sims = jnp.einsum("ma,a->m", patterns, retrieved)
    print()
    print("Similarities to stored patterns:", sims)
    return (
        final_weights,
        history,
        hopfield_retrieve,
        key2,
        noise,
        noisy_query,
        retrieved,
        sims,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualization: Convergence Trajectory
        """
    )
    return


@app.cell
def _(history, jnp, patterns, plt):
    # Track similarity to each pattern over iterations
    similarities_over_time = []
    for state in history:
        sims = jnp.einsum("ma,a->m", patterns, state)
        similarities_over_time.append(sims)
    similarities_over_time = jnp.array(similarities_over_time)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in range(patterns.shape[0]):
        ax.plot(similarities_over_time[:, m], label=f"Pattern {m}", marker="o")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Similarity")
    ax.set_title("Hopfield Retrieval: Convergence to Stored Pattern")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig
    return ax, fig, m, similarities_over_time, sims, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Temperature Effects

        The inverse temperature $\beta$ controls retrieval sharpness:
        - **High $\beta$** (low temperature): Sharp retrieval, nearest pattern wins
        - **Low $\beta$** (high temperature): Soft retrieval, mixture of patterns
        """
    )
    return


@app.cell
def _(hopfield_retrieve, jnp, noisy_query, patterns, plt):
    betas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Final weights for different temperatures
    weights_by_beta = []
    iters_by_beta = []

    for beta in betas:
        _, hist, weights = hopfield_retrieve(noisy_query, patterns, beta=beta)
        weights_by_beta.append(weights)
        iters_by_beta.append(len(hist))

    weights_by_beta = jnp.array(weights_by_beta)

    # Plot 1: Attention weights
    for m in range(patterns.shape[0]):
        axes[0].plot(betas, weights_by_beta[:, m], label=f"Pattern {m}", marker="o")

    axes[0].set_xlabel("Inverse Temperature β")
    axes[0].set_ylabel("Attention Weight")
    axes[0].set_title("Retrieval Sharpness vs Temperature")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Convergence speed
    axes[1].plot(betas, iters_by_beta, marker="o", color="green")
    axes[1].set_xlabel("Inverse Temperature β")
    axes[1].set_ylabel("Iterations to Converge")
    axes[1].set_title("Convergence Speed")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return (
        axes,
        beta,
        betas,
        fig,
        hist,
        iters_by_beta,
        m,
        weights,
        weights_by_beta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Energy Landscape

        The energy landscape shows minima at stored patterns:
        """
    )
    return


@app.cell
def _(jnp, modern_hopfield_energy, patterns, plt):
    # Visualize energy along interpolation between two patterns
    pattern_a = patterns[0]
    pattern_b = patterns[1]

    alphas = jnp.linspace(-0.5, 1.5, 100)
    energies = []

    for alpha in alphas:
        state = (1 - alpha) * pattern_a + alpha * pattern_b
        E = modern_hopfield_energy(state, patterns, beta=5.0)
        energies.append(E)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, energies, linewidth=2)
    ax.axvline(0, color="red", linestyle="--", label="Pattern 0", alpha=0.7)
    ax.axvline(1, color="blue", linestyle="--", label="Pattern 1", alpha=0.7)
    ax.set_xlabel("Interpolation α (0=Pattern 0, 1=Pattern 1)")
    ax.set_ylabel("Energy E(x)")
    ax.set_title("Energy Landscape Between Two Patterns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig
    return E, alpha, alphas, ax, energies, fig, pattern_a, pattern_b, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Memory Capacity

        The key advantage of modern Hopfield networks is **exponential capacity**:

        | Network | Capacity |
        |---------|----------|
        | Classical | $\approx 0.14N$ (linear in dimension) |
        | Modern | $\approx \exp(\alpha d)$ (exponential in dimension!) |

        This is why transformers can "remember" so much information.
        """
    )
    return


@app.cell
def _(jnp, plt):
    # Compare capacity scaling
    dims = jnp.arange(10, 110, 10)

    classical_capacity = 0.14 * dims
    modern_capacity_approx = jnp.exp(0.3 * dims)  # Conservative estimate

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, classical_capacity, label="Classical Hopfield", marker="o")
    ax.plot(dims, modern_capacity_approx, label="Modern Hopfield", marker="s")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Memory Capacity (log scale)")
    ax.set_title("Memory Capacity: Classical vs Modern Hopfield")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig
    return ax, classical_capacity, dims, fig, modern_capacity_approx


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Attention as Parallel Retrieval

        In a transformer:
        - Each query retrieves from the same memory (keys/values)
        - All queries retrieve in parallel
        - This is **batched Hopfield retrieval**

        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

        Each row of the output is an independent Hopfield retrieval.
        """
    )
    return


@app.cell
def _(jnp, softmax):
    def attention_as_hopfield(queries, keys, values, beta=None):
        """
        Attention interpreted as parallel Hopfield retrieval.

        Args:
            queries: Probe states of shape (n_q, d)
            keys: Stored patterns for similarity (n_k, d)
            values: Patterns to retrieve (n_k, d_v)
            beta: Inverse temperature (default: 1/sqrt(d))
        """
        d = queries.shape[-1]
        if beta is None:
            beta = 1.0 / jnp.sqrt(d)

        # Similarities (scores)
        scores = jnp.einsum("qa,ka->qk", queries, keys) * beta

        # Retrieval weights (attention)
        weights = jnp.array([softmax(s) for s in scores])

        # Retrieved values
        output = jnp.einsum("qk,kv->qv", weights, values)

        return output, weights

    return (attention_as_hopfield,)


@app.cell
def _(attention_as_hopfield, jnp, patterns, random):
    # Multiple queries retrieving from the same memory
    key3 = random.PRNGKey(456)
    n_queries = 3

    # Create queries near different patterns
    queries = patterns[:n_queries] + random.normal(key3, (n_queries, patterns.shape[1])) * 0.2

    # Use patterns as both keys and values (auto-associative)
    output, weights = attention_as_hopfield(queries, patterns, patterns, beta=5.0)

    print("Parallel Hopfield Retrieval (Attention)")
    print("=" * 50)
    print()
    print("Queries (noisy versions of patterns):")
    for q in range(n_queries):
        print(f"  Query {q}: closest to pattern {jnp.argmax(weights[q])}")
    print()
    print("Attention weights (retrieval distribution):")
    print(weights)
    print()
    print("Note: Each query independently retrieves from the memory")
    return key3, n_queries, output, q, queries, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        The Hopfield-Attention connection reveals:

        1. **Attention is memory retrieval**: Queries probe stored patterns
        2. **Softmax is Gibbs distribution**: Energy-based retrieval
        3. **Exponential capacity**: Modern formulation stores vastly more patterns
        4. **Temperature controls sharpness**: Higher $\beta$ = more focused retrieval

        This perspective explains why transformers are so powerful:
        they perform massively parallel associative memory retrieval.

        Next notebook: worked numerical examples through the full attention pipeline.
        """
    )
    return


if __name__ == "__main__":
    app.run()
