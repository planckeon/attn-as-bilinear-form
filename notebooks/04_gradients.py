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
        # 4. Gradient Derivations and Verification

        This notebook derives attention gradients in **index notation** and verifies them against JAX autodiff.

        ## The Chain Rule in Index Notation

        For scalar loss $L$, the gradient w.r.t. $Q^{kl}$ is:

        $$\frac{\partial L}{\partial Q^{kl}} = \frac{\partial L}{\partial O^{ib}} \frac{\partial O^{ib}}{\partial A^{mn}} \frac{\partial A^{mn}}{\partial S^{pq}} \frac{\partial S^{pq}}{\partial Q^{kl}}$$

        We'll compute each piece and verify the result.
        """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    return jax, jnp, plt


@app.cell
def _(jnp):
    def attention_forward(Q, K, V):
        """
        Full attention forward pass with cached intermediates.

        Returns output and all intermediate tensors for gradient computation.
        """
        d_k = Q.shape[-1]
        scale = 1.0 / jnp.sqrt(d_k)

        # S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale

        # A^{ij} = softmax_j(S^{ij})
        S_max = jnp.max(S, axis=-1, keepdims=True)
        exp_S = jnp.exp(S - S_max)
        A = exp_S / jnp.sum(exp_S, axis=-1, keepdims=True)

        # O^{ib} = A^{ij} V^{jb}
        O = jnp.einsum("ij,jb->ib", A, V)

        return O, {"S": S, "A": A, "scale": scale}

    return (attention_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Through Value Aggregation

        Given $O^{ib} = A^{ij} V^{jb}$:

        $$\frac{\partial O^{ib}}{\partial A^{mn}} = \delta^i_m V^{nb}$$

        $$\frac{\partial O^{ib}}{\partial V^{mn}} = A^{im} \delta^b_n$$

        Therefore:

        $$\frac{\partial L}{\partial A^{mn}} = \frac{\partial L}{\partial O^{mb}} V^{nb} \quad \text{(matrix: } \bar{O} V^T \text{)}$$

        $$\frac{\partial L}{\partial V^{mn}} = A^{im} \frac{\partial L}{\partial O^{in}} \quad \text{(matrix: } A^T \bar{O} \text{)}$$
        """
    )
    return


@app.cell
def _(jnp):
    def grad_output_wrt_A(dL_dO, V):
        """dL/dA = dL/dO @ V^T"""
        return jnp.einsum("ib,jb->ij", dL_dO, V)

    def grad_output_wrt_V(dL_dO, A):
        """dL/dV = A^T @ dL/dO"""
        return jnp.einsum("ij,ib->jb", A, dL_dO)

    return grad_output_wrt_A, grad_output_wrt_V


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Through Softmax

        The softmax Jacobian is:

        $$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m A^{ij} (\delta^j_n - A^{in})$$

        This gives the famous formula:

        $$\frac{\partial L}{\partial S^{mn}} = A^{mn} \left( \frac{\partial L}{\partial A^{mn}} - \sum_j A^{mj} \frac{\partial L}{\partial A^{mj}} \right)$$

        Or: $\bar{S} = A \odot (\bar{A} - \text{rowsum}(A \odot \bar{A}))$
        """
    )
    return


@app.cell
def _(jnp):
    def grad_softmax(dL_dA, A):
        """Gradient through softmax"""
        # sum_j A^{ij} * dL/dA^{ij}
        sum_term = jnp.sum(A * dL_dA, axis=-1, keepdims=True)
        # dL/dS^{ij} = A^{ij} * (dL/dA^{ij} - sum_term^i)
        return A * (dL_dA - sum_term)

    return (grad_softmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Through Score Computation

        Given $S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$:

        $$\frac{\partial S^{ij}}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \delta^i_k K^{jl}$$

        $$\frac{\partial S^{ij}}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \delta^j_k Q^{il}$$

        Therefore:

        $$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{kj}} K^{jl} \quad \text{(matrix: } \frac{1}{\sqrt{d_k}} \bar{S} K \text{)}$$

        $$\frac{\partial L}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S^{ik}} Q^{il} \quad \text{(matrix: } \frac{1}{\sqrt{d_k}} \bar{S}^T Q \text{)}$$
        """
    )
    return


@app.cell
def _(jnp):
    def grad_scores_wrt_Q(dL_dS, K, scale):
        """dL/dQ = scale * dL/dS @ K"""
        return scale * jnp.einsum("ij,jl->il", dL_dS, K)

    def grad_scores_wrt_K(dL_dS, Q, scale):
        """dL/dK = scale * dL/dS^T @ Q"""
        return scale * jnp.einsum("ij,il->jl", dL_dS, Q)

    return grad_scores_wrt_Q, grad_scores_wrt_K


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Complete Manual Backward Pass
        """
    )
    return


@app.cell
def _(
    grad_output_wrt_A,
    grad_output_wrt_V,
    grad_scores_wrt_K,
    grad_scores_wrt_Q,
    grad_softmax,
):
    def attention_backward_manual(dL_dO, Q, K, V, A, scale):
        """
        Complete backward pass using manual gradient formulas.

        Args:
            dL_dO: Upstream gradient w.r.t. output
            Q, K, V: Forward inputs
            A: Cached attention weights
            scale: 1/sqrt(d_k)

        Returns:
            dL_dQ, dL_dK, dL_dV
        """
        # Step 1: dL/dV = A^T @ dL/dO
        dL_dV = grad_output_wrt_V(dL_dO, A)

        # Step 2: dL/dA = dL/dO @ V^T
        dL_dA = grad_output_wrt_A(dL_dO, V)

        # Step 3: dL/dS = gradient through softmax
        dL_dS = grad_softmax(dL_dA, A)

        # Step 4: dL/dQ and dL/dK
        dL_dQ = grad_scores_wrt_Q(dL_dS, K, scale)
        dL_dK = grad_scores_wrt_K(dL_dS, Q, scale)

        return dL_dQ, dL_dK, dL_dV

    return (attention_backward_manual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Verification Against JAX Autodiff

        Now we verify our manual gradients match JAX's automatic differentiation.
        """
    )
    return


@app.cell
def _(attention_backward_manual, attention_forward, jax, jnp):
    # Create test inputs
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    n_q, n_k, d_k, d_v = 4, 6, 8, 8

    Q_test = jax.random.normal(keys[0], (n_q, d_k))
    K_test = jax.random.normal(keys[1], (n_k, d_k))
    V_test = jax.random.normal(keys[2], (n_k, d_v))

    # Forward pass
    O_test, cache = attention_forward(Q_test, K_test, V_test)

    # Define loss function: L = ||O||^2 / 2
    def loss_fn(Q, K, V):
        O, _ = attention_forward(Q, K, V)
        return 0.5 * jnp.sum(O**2)

    # JAX gradients
    dL_dQ_jax, dL_dK_jax, dL_dV_jax = jax.grad(loss_fn, argnums=(0, 1, 2))(Q_test, K_test, V_test)

    # Manual gradients
    dL_dO = O_test  # For L = ||O||^2 / 2, dL/dO = O
    dL_dQ_manual, dL_dK_manual, dL_dV_manual = attention_backward_manual(
        dL_dO, Q_test, K_test, V_test, cache["A"], cache["scale"]
    )

    print("Verification Results:")
    print("=" * 50)
    print(f"dL/dQ match: {jnp.allclose(dL_dQ_jax, dL_dQ_manual, rtol=1e-5)}")
    print(f"  Max diff: {jnp.max(jnp.abs(dL_dQ_jax - dL_dQ_manual)):.2e}")
    print()
    print(f"dL/dK match: {jnp.allclose(dL_dK_jax, dL_dK_manual, rtol=1e-5)}")
    print(f"  Max diff: {jnp.max(jnp.abs(dL_dK_jax - dL_dK_manual)):.2e}")
    print()
    print(f"dL/dV match: {jnp.allclose(dL_dV_jax, dL_dV_manual, rtol=1e-5)}")
    print(f"  Max diff: {jnp.max(jnp.abs(dL_dV_jax - dL_dV_manual)):.2e}")
    return (
        K_test,
        O_test,
        Q_test,
        V_test,
        cache,
        d_k,
        d_v,
        dL_dK_jax,
        dL_dK_manual,
        dL_dO,
        dL_dQ_jax,
        dL_dQ_manual,
        dL_dV_jax,
        dL_dV_manual,
        key,
        keys,
        loss_fn,
        n_k,
        n_q,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualizing Gradient Flow
        """
    )
    return


@app.cell
def _(dL_dK_manual, dL_dQ_manual, dL_dV_manual, plt):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(dL_dQ_manual, cmap="RdBu", aspect="auto")
    axes[0].set_title("$\\partial L / \\partial Q$")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Query")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dL_dK_manual, cmap="RdBu", aspect="auto")
    axes[1].set_title("$\\partial L / \\partial K$")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Key")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(dL_dV_manual, cmap="RdBu", aspect="auto")
    axes[2].set_title("$\\partial L / \\partial V$")
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("Value")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Gradient Flow Through Attention")
    plt.tight_layout()
    fig
    return axes, fig, im0, im1, im2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Norms by Component

        Let's see which gradients are largest:
        """
    )
    return


@app.cell
def _(dL_dK_manual, dL_dQ_manual, dL_dV_manual, jnp, plt):
    norms = {
        "Q": float(jnp.linalg.norm(dL_dQ_manual)),
        "K": float(jnp.linalg.norm(dL_dK_manual)),
        "V": float(jnp.linalg.norm(dL_dV_manual)),
    }

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(norms.keys(), norms.values(), color=["steelblue", "coral", "seagreen"])
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Magnitudes")

    for i, (name, norm) in enumerate(norms.items()):
        ax2.text(i, norm + 0.01 * max(norms.values()), f"{norm:.3f}", ha="center")

    plt.tight_layout()
    fig2
    return ax2, fig2, i, name, norm, norms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        We derived and verified all gradients for attention:

        1. **Value gradient**: $\partial L / \partial V = A^T (\partial L / \partial O)$
        2. **Attention weight gradient**: $\partial L / \partial A = (\partial L / \partial O) V^T$
        3. **Score gradient**: $\partial L / \partial S = A \odot (\partial L / \partial A - \text{rowsum}(A \odot \partial L / \partial A))$
        4. **Query gradient**: $\partial L / \partial Q = \frac{1}{\sqrt{d_k}} (\partial L / \partial S) K$
        5. **Key gradient**: $\partial L / \partial K = \frac{1}{\sqrt{d_k}} (\partial L / \partial S)^T Q$

        All formulas verified against JAX autodiff!
        """
    )
    return


if __name__ == "__main__":
    app.run()
