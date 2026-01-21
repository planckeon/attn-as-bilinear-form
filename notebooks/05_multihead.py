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
        # 5. Multi-Head Attention

        This notebook explores **multi-head attention**, which extends single-head attention
        by computing multiple attention patterns in parallel.

        ## Key Ideas

        1. **Multiple perspectives**: Each head learns different attention patterns
        2. **Subspace projections**: Project Q, K, V into lower-dimensional subspaces
        3. **Concatenation**: Combine head outputs into final representation

        ## Index Notation

        We add a head index $h$:
        - $Q^{hia}$: Query for head $h$, position $i$, feature $a$
        - $K^{hja}$: Key for head $h$, position $j$, feature $a$
        - $V^{hjc}$: Value for head $h$, position $j$, feature $c$
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
        ## Multi-Head Architecture

        For each head $h$:

        1. **Project**: $Q^{hia} = X^{ib} W_Q^{hba}$
        2. **Attend**: $A^{hij} = \text{softmax}_j\left(\frac{Q^{hia} K^{hja}}{\sqrt{d_k}}\right)$
        3. **Aggregate**: $O^{hic} = A^{hij} V^{hjc}$

        Then combine heads:

        4. **Output**: $Y^{ia} = O^{hic} W_O^{hca}$ (sum over $h$ and $c$)
        """
    )
    return


@app.cell
def _(jnp):
    def softmax_rows(x):
        """Numerically stable softmax over last axis."""
        x_stable = x - jnp.max(x, axis=-1, keepdims=True)
        exp_x = jnp.exp(x_stable)
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def multihead_attention(X, W_Q, W_K, W_V, W_O, return_weights=False):
        """
        Multi-head attention (Vaswani et al., 2017).

        Index notation:
            Q^{hia} = X^{ib} W_Q^{hba}
            K^{hja} = X^{jb} W_K^{hba}
            V^{hjc} = X^{jb} W_V^{hbc}
            S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
            A^{hij} = softmax_j(S^{hij})
            O^{hic} = A^{hij} V^{hjc}
            Y^{ia} = O^{hic} W_O^{hca}

        Args:
            X: Input of shape (n, d_model)
            W_Q: Query projections (H, d_model, d_k)
            W_K: Key projections (H, d_model, d_k)
            W_V: Value projections (H, d_model, d_v)
            W_O: Output projection (H, d_v, d_model)
        """
        d_k = W_Q.shape[2]

        # Project Q, K, V for all heads
        Q_h = jnp.einsum("ib,hba->hia", X, W_Q)  # (H, n, d_k)
        K_h = jnp.einsum("jb,hba->hja", X, W_K)  # (H, n, d_k)
        V_h = jnp.einsum("jb,hbc->hjc", X, W_V)  # (H, n, d_v)

        # Attention scores: S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
        S = jnp.einsum("hia,hja->hij", Q_h, K_h) / jnp.sqrt(d_k)

        # Softmax: A^{hij} = softmax_j(S^{hij})
        A = softmax_rows(S)

        # Weighted sum: O^{hic} = A^{hij} V^{hjc}
        O = jnp.einsum("hij,hjc->hic", A, V_h)

        # Combine heads: Y^{ia} = O^{hic} W_O^{hca}
        Y = jnp.einsum("hic,hca->ia", O, W_O)

        if return_weights:
            return Y, A
        return Y

    return multihead_attention, softmax_rows


@app.cell
def _(jnp, random):
    def init_weights(key, d_model, num_heads, d_k=None, d_v=None):
        """Initialize multi-head attention weights."""
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        keys = random.split(key, 4)

        # Xavier initialization
        scale_qk = jnp.sqrt(2.0 / (d_model + d_k))
        scale_v = jnp.sqrt(2.0 / (d_model + d_v))
        scale_o = jnp.sqrt(2.0 / (num_heads * d_v + d_model))

        return {
            "W_Q": random.normal(keys[0], (num_heads, d_model, d_k)) * scale_qk,
            "W_K": random.normal(keys[1], (num_heads, d_model, d_k)) * scale_qk,
            "W_V": random.normal(keys[2], (num_heads, d_model, d_v)) * scale_v,
            "W_O": random.normal(keys[3], (num_heads, d_v, d_model)) * scale_o,
        }

    return (init_weights,)


@app.cell
def _(init_weights, jnp, multihead_attention, random):
    # Setup: 4 tokens, model dim 8, 2 heads
    n_tokens = 4
    d_model = 8
    num_heads = 2
    d_k = d_model // num_heads  # 4
    d_v = d_model // num_heads  # 4

    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    # Random input sequence
    X = random.normal(subkey, (n_tokens, d_model))

    # Initialize weights
    weights = init_weights(key, d_model, num_heads)

    # Forward pass
    Y, A = multihead_attention(
        X,
        weights["W_Q"],
        weights["W_K"],
        weights["W_V"],
        weights["W_O"],
        return_weights=True,
    )

    print("Multi-Head Attention")
    print("=" * 50)
    print()
    print(f"Configuration:")
    print(f"  Tokens: {n_tokens}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Per-head key/query dim: {d_k}")
    print(f"  Per-head value dim: {d_v}")
    print()
    print(f"Weight shapes:")
    print(f"  W_Q: {weights['W_Q'].shape} (H, d_model, d_k)")
    print(f"  W_K: {weights['W_K'].shape} (H, d_model, d_k)")
    print(f"  W_V: {weights['W_V'].shape} (H, d_model, d_v)")
    print(f"  W_O: {weights['W_O'].shape} (H, d_v, d_model)")
    print()
    print(f"Input X: {X.shape}")
    print(f"Output Y: {Y.shape}")
    print(f"Attention weights A: {A.shape} (H, n, n)")
    return (
        A,
        X,
        Y,
        d_k,
        d_model,
        d_v,
        key,
        n_tokens,
        num_heads,
        subkey,
        weights,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualizing Attention Heads

        Each head learns different attention patterns. Let's visualize them:
        """
    )
    return


@app.cell
def _(A, num_heads, plt):
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 3.5))

    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        im = axes[h].imshow(A[h], cmap="Blues", vmin=0, vmax=1)
        axes[h].set_title(f"Head {h}")
        axes[h].set_xlabel("Key position")
        axes[h].set_ylabel("Query position")

        # Add value annotations
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                val = A[h, i, j]
                color = "white" if val > 0.5 else "black"
                axes[h].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=axes, shrink=0.8, label="Attention weight")
    plt.suptitle("Attention Patterns per Head")
    plt.tight_layout()
    fig
    return axes, color, fig, h, i, im, j, val


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Head Diversity Analysis

        Good multi-head attention should have **diverse heads** that capture different patterns.

        We measure diversity using:
        1. **Cosine similarity** between attention patterns
        2. **Entropy** of each head's attention distribution
        """
    )
    return


@app.cell
def _(jnp):
    def head_diversity(A):
        """
        Measure diversity of attention patterns across heads.

        Returns average pairwise cosine distance (higher = more diverse).
        """
        H = A.shape[0]

        # Flatten each head's attention pattern
        A_flat = A.reshape(H, -1)  # (H, n*n)

        # Normalize to unit vectors
        A_norm = A_flat / (jnp.linalg.norm(A_flat, axis=-1, keepdims=True) + 1e-8)

        # Pairwise cosine similarities
        similarities = jnp.einsum("hi,ji->hj", A_norm, A_norm)

        # Average off-diagonal
        mask = 1 - jnp.eye(H)
        avg_similarity = jnp.sum(similarities * mask) / (H * (H - 1) + 1e-8)

        return 1 - avg_similarity  # Convert to diversity

    def head_entropy(A):
        """Compute mean entropy for each head."""
        # Entropy per (head, query): -sum_j A log A
        log_A = jnp.log(A + 1e-10)
        H_per_query = -jnp.sum(A * log_A, axis=-1)  # (H, n_q)
        return jnp.mean(H_per_query, axis=-1)  # (H,)

    return head_diversity, head_entropy


@app.cell
def _(A, head_diversity, head_entropy, jnp, num_heads):
    diversity = head_diversity(A)
    entropies = head_entropy(A)

    print("Head Analysis")
    print("=" * 50)
    print()
    print(f"Diversity score: {diversity:.4f}")
    print("  (0 = identical heads, 1 = orthogonal heads)")
    print()
    print("Entropy per head:")
    max_entropy = jnp.log(A.shape[-1])
    for h in range(num_heads):
        print(
            f"  Head {h}: {entropies[h]:.4f} / {max_entropy:.4f} max ({100 * entropies[h] / max_entropy:.1f}%)"
        )
    print()
    print("Interpretation:")
    print("  Low entropy = focused attention (few keys attended)")
    print("  High entropy = diffuse attention (many keys attended)")
    return diversity, entropies, h, max_entropy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Decomposing Multi-Head Attention

        Let's trace through each step of the computation:
        """
    )
    return


@app.cell
def _(X, jnp, softmax_rows, weights):
    # Step-by-step decomposition
    W_Q = weights["W_Q"]
    W_K = weights["W_K"]
    W_V = weights["W_V"]
    W_O = weights["W_O"]

    d_k_step = W_Q.shape[2]

    # Step 1: Project Q, K, V
    Q_h = jnp.einsum("ib,hba->hia", X, W_Q)
    K_h = jnp.einsum("jb,hba->hja", X, W_K)
    V_h = jnp.einsum("jb,hbc->hjc", X, W_V)

    print("Step 1: Projections")
    print(f"  Q_h shape: {Q_h.shape} (H, n, d_k)")
    print(f"  K_h shape: {K_h.shape} (H, n, d_k)")
    print(f"  V_h shape: {V_h.shape} (H, n, d_v)")

    # Step 2: Attention scores
    S_h = jnp.einsum("hia,hja->hij", Q_h, K_h) / jnp.sqrt(d_k_step)
    print(f"\nStep 2: Scores")
    print(f"  S_h shape: {S_h.shape} (H, n, n)")

    # Step 3: Softmax
    A_h = softmax_rows(S_h)
    print(f"\nStep 3: Attention weights")
    print(f"  A_h shape: {A_h.shape} (H, n, n)")

    # Step 4: Weighted sum of values
    O_h = jnp.einsum("hij,hjc->hic", A_h, V_h)
    print(f"\nStep 4: Per-head outputs")
    print(f"  O_h shape: {O_h.shape} (H, n, d_v)")

    # Step 5: Combine heads
    Y_final = jnp.einsum("hic,hca->ia", O_h, W_O)
    print(f"\nStep 5: Combined output")
    print(f"  Y shape: {Y_final.shape} (n, d_model)")
    return (
        A_h,
        K_h,
        O_h,
        Q_h,
        S_h,
        V_h,
        W_K,
        W_O,
        W_Q,
        W_V,
        Y_final,
        d_k_step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parameter Counting

        Multi-head attention has the same parameter count as a single large head,
        but provides more expressiveness through diverse attention patterns.
        """
    )
    return


@app.cell
def _(d_k, d_model, d_v, num_heads):
    # Count parameters
    params_Q = num_heads * d_model * d_k
    params_K = num_heads * d_model * d_k
    params_V = num_heads * d_model * d_v
    params_O = num_heads * d_v * d_model

    total_params = params_Q + params_K + params_V + params_O

    # Compare to single-head with same dimensions
    single_head_params = d_model * d_model * 4  # Q, K, V, O projections

    print("Parameter Count")
    print("=" * 50)
    print()
    print(f"Multi-head ({num_heads} heads, d_k={d_k}, d_v={d_v}):")
    print(f"  W_Q: {params_Q} = {num_heads} * {d_model} * {d_k}")
    print(f"  W_K: {params_K} = {num_heads} * {d_model} * {d_k}")
    print(f"  W_V: {params_V} = {num_heads} * {d_model} * {d_v}")
    print(f"  W_O: {params_O} = {num_heads} * {d_v} * {d_model}")
    print(f"  Total: {total_params}")
    print()
    print(f"Equivalent single-head (d_model projections):")
    print(f"  Total: {single_head_params}")
    print()
    print(f"Multi-head uses {100 * total_params / single_head_params:.1f}% of single-head params")
    return (
        params_K,
        params_O,
        params_Q,
        params_V,
        single_head_params,
        total_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why Multiple Heads?

        Each head can learn to attend to different aspects:
        - **Syntactic heads**: Subject-verb agreement
        - **Semantic heads**: Entity relationships
        - **Positional heads**: Local context
        - **Long-range heads**: Distant dependencies

        The diversity analysis above shows how different the heads are in practice.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        Multi-head attention:

        1. **Projects** input into $H$ separate subspaces
        2. **Computes** attention independently in each head
        3. **Combines** outputs via learned projection

        Index notation makes the tensor structure explicit:
        - Head index $h$ runs over heads
        - Each head has its own Q, K, V projections
        - Final output sums contributions from all heads

        The next notebook explores attention masking for causal and sparse patterns.
        """
    )
    return


if __name__ == "__main__":
    app.run()
