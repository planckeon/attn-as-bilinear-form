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
        # 1. Bilinear Forms and Metric Tensors

        This notebook explores the **tensor calculus foundations** of attention mechanisms.

        ## Key Concepts

        1. **Metric tensors** $g_{ab}$ define how we measure similarity
        2. **Bilinear forms** $B(u,v) = u^a g_{ab} v^b$ compute similarity scores
        3. **Index notation** makes tensor operations explicit

        ## Notation

        - **Superscripts** (contravariant): $v^a$, $Q^{ia}$
        - **Subscripts** (covariant): $g_{ab}$, $u_a$
        - **Einstein summation**: repeated indices are summed
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
        ## Metric Tensors

        A **metric tensor** $g_{ab}$ is a symmetric, positive-definite matrix that defines an inner product:

        $$\langle u, v \rangle_g = u^a g_{ab} v^b$$

        Different metrics give different notions of "similarity" between vectors.
        """
    )
    return


@app.cell
def _(jnp):
    def euclidean_metric(d: int):
        """Identity metric: g_{ab} = delta_{ab}"""
        return jnp.eye(d)

    def scaled_euclidean_metric(d: int):
        """Scaled metric used in attention: g_{ab} = (1/sqrt(d)) * delta_{ab}"""
        return jnp.eye(d) / jnp.sqrt(d)

    def diagonal_metric(weights):
        """Diagonal metric with custom weights"""
        return jnp.diag(weights)

    return diagonal_metric, euclidean_metric, scaled_euclidean_metric


@app.cell
def _(euclidean_metric, jnp, scaled_euclidean_metric):
    # Example: 4-dimensional feature space
    d = 4

    g_euclid = euclidean_metric(d)
    g_scaled = scaled_euclidean_metric(d)

    print("Euclidean metric g_{ab} = delta_{ab}:")
    print(g_euclid)
    print()
    print(f"Scaled metric g_{{ab}} = (1/sqrt({d})) * delta_{{ab}}:")
    print(g_scaled)
    print()
    print(f"Scale factor: 1/sqrt({d}) = {1 / jnp.sqrt(d):.4f}")
    return d, g_euclid, g_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Bilinear Forms

        A **bilinear form** is a map $B: V \times W \to \mathbb{R}$ that's linear in both arguments.

        In index notation with metric $g_{ab}$:
        $$B(u, v) = u^a g_{ab} v^b$$

        This is exactly what attention scores compute!
        """
    )
    return


@app.cell
def _(jnp):
    def bilinear_form(u, v, g):
        """
        Compute bilinear form: B(u,v) = u^a g_{ab} v^b

        Using einsum to make the index contraction explicit.
        """
        # First lower the index of v: v_a = g_{ab} v^b
        v_lower = jnp.einsum("ab,b->a", g, v)
        # Then contract with u: u^a v_a
        return jnp.einsum("a,a->", u, v_lower)

    def bilinear_form_batch(Q, K, g):
        """
        Batch bilinear form for attention: S^{ij} = Q^{ia} g_{ab} K^{jb}

        This computes all pairwise similarities.
        """
        # K_lower_{ja} = g_{ab} K^{jb}
        K_lower = jnp.einsum("ab,jb->ja", g, K)
        # S^{ij} = Q^{ia} K_{ja}
        return jnp.einsum("ia,ja->ij", Q, K_lower)

    return bilinear_form, bilinear_form_batch


@app.cell
def _(bilinear_form, g_euclid, g_scaled, jnp):
    # Two vectors in 4D
    u = jnp.array([1.0, 0.0, 0.0, 0.0])
    v = jnp.array([1.0, 1.0, 0.0, 0.0])

    # Compute bilinear forms with different metrics
    B_euclid = bilinear_form(u, v, g_euclid)
    B_scaled = bilinear_form(u, v, g_scaled)

    print(f"Vectors:")
    print(f"  u = {u}")
    print(f"  v = {v}")
    print()
    print(f"Bilinear form B(u,v) = u^a g_{{ab}} v^b:")
    print(f"  Euclidean metric: {B_euclid:.4f}")
    print(f"  Scaled metric:    {B_scaled:.4f}")
    print()
    print(
        f"Note: Scaled = Euclidean / sqrt(d) = {B_euclid:.4f} / {jnp.sqrt(4):.1f} = {B_scaled:.4f}"
    )
    return B_euclid, B_scaled, u, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Connection to Attention Scores

        The attention score between query $Q^{ia}$ and key $K^{ja}$ is:

        $$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

        where $g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$ for scaled dot-product attention.
        """
    )
    return


@app.cell
def _(bilinear_form_batch, jnp, scaled_euclidean_metric):
    # Example: 2 queries, 3 keys, dimension 4
    Q = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Query 1: unit vector along axis 0
            [0.0, 1.0, 0.0, 0.0],  # Query 2: unit vector along axis 1
        ]
    )

    K = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Key 1: same as Query 1
            [0.0, 1.0, 0.0, 0.0],  # Key 2: same as Query 2
            [1.0, 1.0, 0.0, 0.0],  # Key 3: mix of both
        ]
    )

    g = scaled_euclidean_metric(4)

    # Compute attention scores using bilinear form
    S = bilinear_form_batch(Q, K, g)

    print("Queries Q^{ia}:")
    print(Q)
    print()
    print("Keys K^{ja}:")
    print(K)
    print()
    print("Attention Scores S^{ij} = Q^{ia} g_{ab} K^{jb}:")
    print(S)
    return K, Q, S, g


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interpretation of Scores

        Looking at the score matrix $S^{ij}$:

        - **Diagonal elements** (where $i$ matches the pattern in $j$): Higher scores
        - **Off-diagonal elements**: Lower scores where patterns don't match

        Query 1 matches Key 1 and Key 3 (both have component along axis 0).
        Query 2 matches Key 2 and Key 3 (both have component along axis 1).
        """
    )
    return


@app.cell
def _(S, plt):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(S, cmap="Blues", aspect="auto")
    ax.set_xlabel("Key Index (j)")
    ax.set_ylabel("Query Index (i)")
    ax.set_title("Attention Scores $S^{ij}$")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1])

    # Add value annotations
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(
                j,
                i,
                f"{S[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if S[i, j] > 0.3 else "black",
            )

    plt.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()
    fig
    return ax, fig, i, im, j


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Learned Metrics

        We can generalize beyond scaled dot-product by learning a metric:

        $$g_{ab} = W^c_a W_{cb}$$

        This parameterization ensures positive semi-definiteness (since $g = W^T W$).
        """
    )
    return


@app.cell
def _(jax, jnp):
    def learned_metric(W):
        """
        Learned metric: g_{ab} = W^T W

        This ensures positive semi-definiteness.
        """
        return jnp.einsum("ca,cb->ab", W, W)

    # Example: Learn a 4x4 metric from a 2x4 weight matrix
    key = jax.random.PRNGKey(42)
    W = jax.random.normal(key, (2, 4)) * 0.5

    g_learned = learned_metric(W)

    print("Weight matrix W (rank 2):")
    print(W)
    print()
    print("Learned metric g_{ab} = W^T W:")
    print(g_learned)
    print()
    print(f"Note: Rank of metric = min(rows of W, cols of W) = {min(W.shape)}")
    return W, g_learned, key, learned_metric


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Verifying Metric Properties

        A valid metric tensor must be:
        1. **Symmetric**: $g_{ab} = g_{ba}$
        2. **Positive (semi-)definite**: $v^a g_{ab} v^b \geq 0$ for all $v$
        """
    )
    return


@app.cell
def _(g_learned, g_scaled, jnp):
    def verify_metric(g, name):
        """Check metric properties"""
        is_symmetric = jnp.allclose(g, g.T)
        eigenvalues = jnp.linalg.eigvalsh(g)
        is_positive = jnp.all(eigenvalues >= -1e-6)

        print(f"{name}:")
        print(f"  Symmetric: {is_symmetric}")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Positive semi-definite: {is_positive}")
        print()

    verify_metric(g_scaled, "Scaled Euclidean")
    verify_metric(g_learned, "Learned (W^T W)")
    return (verify_metric,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Metric tensors** define the geometry of feature space
        - **Bilinear forms** measure similarity: $B(u,v) = u^a g_{ab} v^b$
        - **Attention scores** are bilinear forms with scaled Euclidean metric
        - **Learned metrics** generalize to arbitrary positive semi-definite similarity

        The next notebook explores how these scores become attention weights through softmax.
        """
    )
    return


if __name__ == "__main__":
    app.run()
