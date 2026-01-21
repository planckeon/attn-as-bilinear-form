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
        # 2. Attention Forward Pass

        This notebook walks through the **complete attention mechanism** step by step,
        showing how tensor contractions flow from inputs to outputs.

        ## The Attention Formula

        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

        In index notation:

        1. **Scores**: $S^{ij} = Q^{ia} g_{ab} K^{jb}$ where $g_{ab} = \frac{1}{\sqrt{d_k}}\delta_{ab}$
        2. **Weights**: $A^{ij} = \frac{\exp(S^{ij})}{\sum_{j'} \exp(S^{ij'})}$ (Gibbs distribution)
        3. **Output**: $O^{ib} = A^{ij} V^{jb}$ (weighted sum)
        """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    return jax, jnp, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 1: Score Computation

        The attention score between query $i$ and key $j$ is their similarity:

        $$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

        This is a **bilinear form** with the scaled Euclidean metric.
        The contraction over index $a$ is a dot product.
        """
    )
    return


@app.cell
def _(jnp):
    def attention_scores(Q, K, scale=True):
        """
        Compute attention scores: S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)

        The einsum 'ia,ja->ij' contracts over the feature index a.
        """
        d_k = Q.shape[-1]

        # S^{ij} = Q^{ia} K^{ja}
        S = jnp.einsum("ia,ja->ij", Q, K)

        if scale:
            S = S / jnp.sqrt(d_k)

        return S

    return (attention_scores,)


@app.cell
def _(attention_scores, jnp):
    # Example: 2 queries, 3 keys, dimension 4
    Q = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    K = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Aligned with Q[0]
            [0.0, 1.0, 0.0, 0.0],  # Aligned with Q[1]
            [0.7, 0.7, 0.0, 0.0],  # Between Q[0] and Q[1]
        ]
    )

    V = jnp.array(
        [
            [1.0, 0.0],  # Value for K[0]
            [0.0, 1.0],  # Value for K[1]
            [0.5, 0.5],  # Value for K[2]
        ]
    )

    S = attention_scores(Q, K)

    print("Queries Q^{ia}:")
    print(Q)
    print()
    print("Keys K^{ja}:")
    print(K)
    print()
    print("Scores S^{ij} = Q^{ia} K^{ja} / sqrt(d_k):")
    print(S)
    print()
    print(f"Scale factor: 1/sqrt({Q.shape[-1]}) = {1 / jnp.sqrt(Q.shape[-1]):.4f}")
    return K, Q, S, V


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 2: Softmax (Gibbs Distribution)

        Convert scores to probability weights using softmax:

        $$A^{ij} = \frac{\exp(S^{ij})}{\sum_{j'} \exp(S^{ij'})}$$

        This is a **Gibbs distribution** from statistical mechanics!
        - The scores $S^{ij}$ act as negative energies
        - The partition function $Z^i = \sum_{j'} \exp(S^{ij'})$ normalizes

        Each row of $A$ sums to 1 (probability distribution over keys).
        """
    )
    return


@app.cell
def _(jnp):
    def softmax_rows(S):
        """
        Apply softmax row-wise: A^{ij} = exp(S^{ij}) / sum_j' exp(S^{ij'})

        Uses numerically stable computation by subtracting max.
        """
        # Subtract max for numerical stability
        S_stable = S - jnp.max(S, axis=-1, keepdims=True)

        # Exponential
        exp_S = jnp.exp(S_stable)

        # Partition function Z^i = sum_j exp(S^{ij})
        Z = jnp.sum(exp_S, axis=-1, keepdims=True)

        # Attention weights
        return exp_S / Z

    def compute_entropy(A):
        """
        Compute entropy of attention distribution: H^i = -sum_j A^{ij} log(A^{ij})

        Higher entropy = more uniform attention (less focused)
        Lower entropy = more peaked attention (more focused)
        """
        # Add small epsilon to avoid log(0)
        log_A = jnp.log(A + 1e-10)
        return -jnp.sum(A * log_A, axis=-1)

    return compute_entropy, softmax_rows


@app.cell
def _(S, compute_entropy, jnp, softmax_rows):
    A = softmax_rows(S)

    print("Scores S^{ij}:")
    print(S)
    print()
    print("Attention Weights A^{ij} = softmax(S^{ij}):")
    print(A)
    print()
    print("Row sums (should be 1.0):", jnp.sum(A, axis=-1))
    print()
    print("Entropy per query:", compute_entropy(A))
    print("(Max entropy for 3 keys:", jnp.log(3.0), "= uniform distribution)")
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 3: Value Aggregation

        The output is a weighted sum of values:

        $$O^{ib} = A^{ij} V^{jb}$$

        This contracts over the key index $j$, mixing values according to attention weights.

        **Interpretation**: Each output position $i$ receives information from all keys,
        weighted by how much each query-key pair matches.
        """
    )
    return


@app.cell
def _(jnp):
    def attention_output(A, V):
        """
        Compute attention output: O^{ib} = A^{ij} V^{jb}

        Weighted sum of values by attention weights.
        """
        return jnp.einsum("ij,jb->ib", A, V)

    return (attention_output,)


@app.cell
def _(A, V, attention_output, jnp):
    O = attention_output(A, V)

    print("Attention Weights A^{ij}:")
    print(A)
    print()
    print("Values V^{jb}:")
    print(V)
    print()
    print("Output O^{ib} = A^{ij} V^{jb}:")
    print(O)
    print()
    print("Interpretation:")
    print(f"  Output[0] = {A[0, 0]:.3f}*V[0] + {A[0, 1]:.3f}*V[1] + {A[0, 2]:.3f}*V[2]")
    print(f"           = {A[0, 0]:.3f}*[1,0] + {A[0, 1]:.3f}*[0,1] + {A[0, 2]:.3f}*[0.5,0.5]")
    manual_0 = A[0, 0] * V[0] + A[0, 1] * V[1] + A[0, 2] * V[2]
    print(f"           = {manual_0}")
    print()
    print(f"  Output[1] = {A[1, 0]:.3f}*V[0] + {A[1, 1]:.3f}*V[1] + {A[1, 2]:.3f}*V[2]")
    manual_1 = A[1, 0] * V[0] + A[1, 1] * V[1] + A[1, 2] * V[2]
    print(f"           = {manual_1}")
    return O, manual_0, manual_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Complete Attention Function

        Putting it all together:
        """
    )
    return


@app.cell
def _(jnp):
    def scaled_dot_product_attention(Q, K, V, return_intermediates=False):
        """
        Full scaled dot-product attention (Vaswani et al., 2017).

        Steps in index notation:
            S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)    # Scores
            A^{ij} = softmax_j(S^{ij})             # Weights
            O^{ib} = A^{ij} V^{jb}                 # Output
        """
        d_k = Q.shape[-1]

        # Step 1: Scores
        S = jnp.einsum("ia,ja->ij", Q, K) / jnp.sqrt(d_k)

        # Step 2: Softmax (numerically stable)
        S_stable = S - jnp.max(S, axis=-1, keepdims=True)
        exp_S = jnp.exp(S_stable)
        Z = jnp.sum(exp_S, axis=-1, keepdims=True)
        A = exp_S / Z

        # Step 3: Weighted sum
        O = jnp.einsum("ij,jb->ib", A, V)

        if return_intermediates:
            return {
                "scores": S,
                "weights": A,
                "partition": Z.squeeze(-1),
                "output": O,
            }
        return O

    return (scaled_dot_product_attention,)


@app.cell
def _(K, Q, V, scaled_dot_product_attention):
    result = scaled_dot_product_attention(Q, K, V, return_intermediates=True)

    print("Complete Attention Flow:")
    print("=" * 50)
    print()
    print("Input Shapes:")
    print(f"  Q: {Q.shape} (n_q, d_k)")
    print(f"  K: {K.shape} (n_k, d_k)")
    print(f"  V: {V.shape} (n_k, d_v)")
    print()
    print("Scores S^{ij}:")
    print(result["scores"])
    print()
    print("Partition Function Z^i:")
    print(result["partition"])
    print()
    print("Attention Weights A^{ij}:")
    print(result["weights"])
    print()
    print("Output O^{ib}:")
    print(result["output"])
    print(f"  Shape: {result['output'].shape} (n_q, d_v)")
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualization

        Let's visualize the attention flow:
        """
    )
    return


@app.cell
def _(result, plt):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Scores
    im0 = axes[0].imshow(result["scores"], cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Scores $S^{ij}$")
    axes[0].set_xlabel("Key j")
    axes[0].set_ylabel("Query i")
    plt.colorbar(im0, ax=axes[0])

    # Add annotations
    for i in range(result["scores"].shape[0]):
        for j in range(result["scores"].shape[1]):
            axes[0].text(
                j,
                i,
                f"{result['scores'][i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    # Attention weights
    im1 = axes[1].imshow(result["weights"], cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Weights $A^{ij}$")
    axes[1].set_xlabel("Key j")
    axes[1].set_ylabel("Query i")
    plt.colorbar(im1, ax=axes[1])

    for i in range(result["weights"].shape[0]):
        for j in range(result["weights"].shape[1]):
            axes[1].text(
                j,
                i,
                f"{result['weights'][i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    # Output
    im2 = axes[2].imshow(result["output"], cmap="viridis", aspect="auto")
    axes[2].set_title("Output $O^{ib}$")
    axes[2].set_xlabel("Value dim b")
    axes[2].set_ylabel("Query i")
    plt.colorbar(im2, ax=axes[2])

    for i in range(result["output"].shape[0]):
        for j in range(result["output"].shape[1]):
            axes[2].text(
                j,
                i,
                f"{result['output'][i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )

    plt.tight_layout()
    fig
    return axes, fig, i, im0, im1, im2, j


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Attention with Explicit Metric

        We can generalize to arbitrary bilinear forms by using an explicit metric tensor:

        $$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

        where $g_{ab}$ can be:
        - Identity: $g_{ab} = \delta_{ab}$ (unscaled dot product)
        - Scaled identity: $g_{ab} = \frac{1}{\sqrt{d_k}}\delta_{ab}$ (standard attention)
        - Learned: $g_{ab} = W^T W$ (learned similarity)
        """
    )
    return


@app.cell
def _(jnp):
    def attention_with_metric(Q, K, V, g):
        """
        Generalized attention with explicit metric tensor.

        S^{ij} = Q^{ia} g_{ab} K^{jb}
        """
        # Lower the key index: K_lower_{ja} = g_{ab} K^{jb}
        K_lower = jnp.einsum("ab,jb->ja", g, K)

        # Compute scores: S^{ij} = Q^{ia} K_{ja}
        S = jnp.einsum("ia,ja->ij", Q, K_lower)

        # Softmax
        S_stable = S - jnp.max(S, axis=-1, keepdims=True)
        A = jnp.exp(S_stable) / jnp.sum(jnp.exp(S_stable), axis=-1, keepdims=True)

        # Output
        O = jnp.einsum("ij,jb->ib", A, V)

        return O, A, S

    return (attention_with_metric,)


@app.cell
def _(K, Q, V, attention_with_metric, jnp):
    d_k = Q.shape[-1]

    # Standard scaled attention metric
    g_scaled = jnp.eye(d_k) / jnp.sqrt(d_k)

    # Unscaled (identity) metric
    g_identity = jnp.eye(d_k)

    # Compare
    O_scaled, A_scaled, S_scaled = attention_with_metric(Q, K, V, g_scaled)
    O_identity, A_identity, S_identity = attention_with_metric(Q, K, V, g_identity)

    print("Comparison of metrics:")
    print()
    print("Scaled metric (standard attention):")
    print("  Scores:", S_scaled.flatten())
    print("  Weights row 0:", A_scaled[0])
    print()
    print("Identity metric (no scaling):")
    print("  Scores:", S_identity.flatten())
    print("  Weights row 0:", A_identity[0])
    print()
    print("Note: Higher scores with identity metric lead to more peaked attention")
    return (
        A_identity,
        A_scaled,
        O_identity,
        O_scaled,
        S_identity,
        S_scaled,
        d_k,
        g_identity,
        g_scaled,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        The attention mechanism is a sequence of tensor contractions:

        1. **Bilinear form** $S^{ij} = Q^{ia} g_{ab} K^{jb}$ computes similarities
        2. **Gibbs distribution** $A^{ij} = \exp(S^{ij})/Z^i$ normalizes to probabilities
        3. **Contraction** $O^{ib} = A^{ij} V^{jb}$ aggregates values

        Each step has a clear geometric or statistical interpretation:
        - Scores: inner product in metric space
        - Weights: Boltzmann distribution over keys
        - Output: expected value under attention distribution

        Next notebook: explore the softmax as a Gibbs distribution from statistical mechanics.
        """
    )
    return


if __name__ == "__main__":
    app.run()
