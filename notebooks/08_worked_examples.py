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
        # 8. Worked Examples

        This notebook provides **complete numerical examples** that you can follow
        step-by-step with pen and paper.

        We'll work through:
        1. A tiny attention computation (2 queries, 3 keys)
        2. Gradient computation by hand
        3. Multi-head with 2 heads
        """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    # Use limited precision for cleaner display
    jnp.set_printoptions(precision=4, suppress=True)

    return jax, jnp, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 1: Minimal Attention (2×3×2)

        **Setup:**
        - 2 queries, 3 keys, dimension 2
        - $d_k = 2$, so $\sqrt{d_k} = \sqrt{2} \approx 1.414$

        ### Input Tensors
        """
    )
    return


@app.cell
def _(jnp):
    # Carefully chosen values for hand computation
    Q = jnp.array(
        [
            [1.0, 0.0],  # Query 0: unit vector along axis 0
            [0.0, 1.0],  # Query 1: unit vector along axis 1
        ]
    )

    K = jnp.array(
        [
            [1.0, 0.0],  # Key 0: matches Query 0
            [0.0, 1.0],  # Key 1: matches Query 1
            [1.0, 1.0],  # Key 2: between both
        ]
    )

    V = jnp.array(
        [
            [1.0, 0.0],  # Value 0
            [0.0, 1.0],  # Value 1
            [0.5, 0.5],  # Value 2
        ]
    )

    print("Input Tensors")
    print("=" * 50)
    print()
    print("Queries Q^{ia} (2 queries, dimension 2):")
    print(Q)
    print()
    print("Keys K^{ja} (3 keys, dimension 2):")
    print(K)
    print()
    print("Values V^{jb} (3 values, dimension 2):")
    print(V)
    return K, Q, V


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Step 1: Score Computation

        $$S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

        For $d_k = 2$: scale factor $= 1/\sqrt{2} \approx 0.707$

        **Manual computation:**
        - $S^{00} = (1 \cdot 1 + 0 \cdot 0) / \sqrt{2} = 1/\sqrt{2} \approx 0.707$
        - $S^{01} = (1 \cdot 0 + 0 \cdot 1) / \sqrt{2} = 0$
        - $S^{02} = (1 \cdot 1 + 0 \cdot 1) / \sqrt{2} = 1/\sqrt{2} \approx 0.707$
        - $S^{10} = (0 \cdot 1 + 1 \cdot 0) / \sqrt{2} = 0$
        - $S^{11} = (0 \cdot 0 + 1 \cdot 1) / \sqrt{2} = 1/\sqrt{2} \approx 0.707$
        - $S^{12} = (0 \cdot 1 + 1 \cdot 1) / \sqrt{2} = 1/\sqrt{2} \approx 0.707$
        """
    )
    return


@app.cell
def _(K, Q, jnp):
    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)

    # Compute scores using einsum
    S = jnp.einsum("ia,ja->ij", Q, K) * scale

    print("Step 1: Score Computation")
    print("=" * 50)
    print()
    print(f"Scale factor: 1/sqrt({d_k}) = {scale:.4f}")
    print()
    print("Raw dot products Q @ K^T:")
    print(jnp.einsum("ia,ja->ij", Q, K))
    print()
    print("Scaled scores S^{ij}:")
    print(S)
    print()
    print("Verification:")
    sqrt2 = jnp.sqrt(2.0)
    print(f"  1/sqrt(2) = {1 / sqrt2:.4f}")
    print(
        f"  S[0,0] = {S[0, 0]:.4f} ✓"
        if jnp.allclose(S[0, 0], 1 / sqrt2)
        else f"  S[0,0] = {S[0, 0]:.4f} ✗"
    )
    return S, d_k, scale, sqrt2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Step 2: Softmax (Row-wise)

        $$A^{ij} = \frac{\exp(S^{ij})}{\sum_{j'} \exp(S^{ij'})}$$

        **For row 0:** $[0.707, 0, 0.707]$
        - $\exp(0.707) \approx 2.028$
        - $\exp(0) = 1$
        - $Z_0 = 2.028 + 1 + 2.028 = 5.056$
        - $A^{0j} = [2.028/5.056, 1/5.056, 2.028/5.056] \approx [0.401, 0.198, 0.401]$

        **For row 1:** $[0, 0.707, 0.707]$
        - Same partition function $Z_1 = 5.056$
        - $A^{1j} = [0.198, 0.401, 0.401]$
        """
    )
    return


@app.cell
def _(S, jnp):
    # Compute softmax
    exp_S = jnp.exp(S)
    Z = jnp.sum(exp_S, axis=-1, keepdims=True)
    A = exp_S / Z

    print("Step 2: Softmax")
    print("=" * 50)
    print()
    print("exp(S):")
    print(exp_S)
    print()
    print("Partition function Z^i:")
    print(Z.squeeze())
    print()
    print("Attention weights A^{ij}:")
    print(A)
    print()
    print("Row sums (should be 1.0):")
    print(jnp.sum(A, axis=-1))
    print()
    print("Manual verification:")
    exp_sqrt2 = jnp.exp(1 / jnp.sqrt(2))
    Z_manual = 2 * exp_sqrt2 + 1
    print(f"  exp(1/sqrt(2)) = {exp_sqrt2:.4f}")
    print(f"  Z = 2*{exp_sqrt2:.4f} + 1 = {Z_manual:.4f}")
    print(f"  A[0,0] = {exp_sqrt2:.4f}/{Z_manual:.4f} = {exp_sqrt2 / Z_manual:.4f}")
    return A, Z, Z_manual, exp_S, exp_sqrt2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Step 3: Value Aggregation

        $$O^{ib} = A^{ij} V^{jb}$$

        **For output 0:**
        $$O^{0b} = 0.401 \cdot V^{0b} + 0.198 \cdot V^{1b} + 0.401 \cdot V^{2b}$$
        $$= 0.401 \cdot [1,0] + 0.198 \cdot [0,1] + 0.401 \cdot [0.5,0.5]$$
        $$= [0.401, 0] + [0, 0.198] + [0.201, 0.201]$$
        $$= [0.602, 0.399]$$
        """
    )
    return


@app.cell
def _(A, V, jnp):
    # Compute output
    O = jnp.einsum("ij,jb->ib", A, V)

    print("Step 3: Value Aggregation")
    print("=" * 50)
    print()
    print("Output O^{ib}:")
    print(O)
    print()
    print("Manual computation for O[0]:")
    print(f"  = {A[0, 0]:.3f}*[1,0] + {A[0, 1]:.3f}*[0,1] + {A[0, 2]:.3f}*[0.5,0.5]")
    term0 = A[0, 0] * V[0]
    term1 = A[0, 1] * V[1]
    term2 = A[0, 2] * V[2]
    print(f"  = {term0} + {term1} + {term2}")
    print(f"  = {term0 + term1 + term2}")
    return O, term0, term1, term2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Complete Forward Pass Summary
        """
    )
    return


@app.cell
def _(A, K, O, Q, S, V, jnp, plt):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    # Q
    im0 = axes[0].imshow(Q, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("Q (2×2)")
    axes[0].set_xlabel("Feature a")
    axes[0].set_ylabel("Query i")
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            axes[0].text(j, i, f"{Q[i, j]:.1f}", ha="center", va="center", fontsize=10)

    # S
    im1 = axes[1].imshow(S, cmap="RdBu_r", aspect="auto")
    axes[1].set_title("Scores S (2×3)")
    axes[1].set_xlabel("Key j")
    axes[1].set_ylabel("Query i")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            axes[1].text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=10)

    # A
    im2 = axes[2].imshow(A, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
    axes[2].set_title("Weights A (2×3)")
    axes[2].set_xlabel("Key j")
    axes[2].set_ylabel("Query i")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            axes[2].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=10)

    # O
    im3 = axes[3].imshow(O, cmap="viridis", aspect="auto")
    axes[3].set_title("Output O (2×2)")
    axes[3].set_xlabel("Value dim b")
    axes[3].set_ylabel("Query i")
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            axes[3].text(
                j, i, f"{O[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white"
            )

    plt.suptitle("Attention Forward Pass: Q → S → A → O", fontsize=12)
    plt.tight_layout()
    fig
    return axes, fig, i, im0, im1, im2, im3, j


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Example 2: Gradient Computation

        Now let's compute gradients for backpropagation.

        Suppose we have a simple loss:
        $$L = \sum_{i,b} O^{ib} = \text{sum of all outputs}$$

        Then $\frac{\partial L}{\partial O^{ib}} = 1$ for all $i, b$.
        """
    )
    return


@app.cell
def _(A, O, V, jnp):
    # Upstream gradient (dL/dO)
    dL_dO = jnp.ones_like(O)

    print("Gradient Computation")
    print("=" * 50)
    print()
    print("Loss: L = sum(O)")
    print(f"L = {jnp.sum(O):.4f}")
    print()
    print("Upstream gradient dL/dO:")
    print(dL_dO)
    return (dL_dO,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gradient w.r.t. Values

        $$\frac{\partial L}{\partial V^{kl}} = \frac{\partial L}{\partial O^{ib}} \frac{\partial O^{ib}}{\partial V^{kl}}$$

        Since $O^{ib} = A^{ij} V^{jb}$:
        $$\frac{\partial O^{ib}}{\partial V^{kl}} = A^{ik} \delta^b_l$$

        Therefore:
        $$\frac{\partial L}{\partial V^{kl}} = \sum_i A^{ik} \frac{\partial L}{\partial O^{il}} = A^{ik} \bar{O}^{il}$$

        In matrix form: $\frac{\partial L}{\partial V} = A^T \cdot \frac{\partial L}{\partial O}$
        """
    )
    return


@app.cell
def _(A, dL_dO, jnp):
    # Gradient w.r.t. V
    dL_dV = jnp.einsum("ij,ib->jb", A, dL_dO)  # A^T @ dL_dO

    print("Gradient w.r.t. Values")
    print("=" * 50)
    print()
    print("Formula: dL/dV = A^T @ dL/dO")
    print()
    print("A^T:")
    print(A.T)
    print()
    print("dL/dV:")
    print(dL_dV)
    print()
    print("Manual verification for dL/dV[0,0]:")
    print(f"  = A[0,0]*dL_dO[0,0] + A[1,0]*dL_dO[1,0]")
    print(f"  = {A[0, 0]:.4f}*{dL_dO[0, 0]:.1f} + {A[1, 0]:.4f}*{dL_dO[1, 0]:.1f}")
    print(f"  = {A[0, 0] * dL_dO[0, 0] + A[1, 0] * dL_dO[1, 0]:.4f}")
    return (dL_dV,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gradient w.r.t. Attention Weights

        $$\frac{\partial L}{\partial A^{mn}} = \frac{\partial L}{\partial O^{ib}} \frac{\partial O^{ib}}{\partial A^{mn}}$$

        Since $O^{ib} = A^{ij} V^{jb}$:
        $$\frac{\partial O^{ib}}{\partial A^{mn}} = \delta^i_m V^{nb}$$

        Therefore:
        $$\frac{\partial L}{\partial A^{mn}} = \bar{O}^{mb} V^{nb} = \bar{O}^m \cdot V^n$$

        In matrix form: $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} \cdot V^T$
        """
    )
    return


@app.cell
def _(V, dL_dO, jnp):
    # Gradient w.r.t. A
    dL_dA = jnp.einsum("ib,jb->ij", dL_dO, V)  # dL_dO @ V^T

    print("Gradient w.r.t. Attention Weights")
    print("=" * 50)
    print()
    print("Formula: dL/dA = dL/dO @ V^T")
    print()
    print("V^T:")
    print(V.T)
    print()
    print("dL/dA:")
    print(dL_dA)
    return (dL_dA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gradient Through Softmax

        This is the trickiest part. The softmax Jacobian is:

        $$\frac{\partial A^{ij}}{\partial S^{mn}} = \delta^i_m \cdot A^{ij}(\delta^j_n - A^{in})$$

        The gradient through softmax is:

        $$\frac{\partial L}{\partial S^{ij}} = A^{ij} \left( \bar{A}^{ij} - \sum_{j'} A^{ij'} \bar{A}^{ij'} \right)$$

        where $\bar{A} = \frac{\partial L}{\partial A}$.

        This can be written as: $\frac{\partial L}{\partial S} = A \odot (\bar{A} - \text{rowsum}(A \odot \bar{A}))$
        """
    )
    return


@app.cell
def _(A, dL_dA, jnp):
    # Gradient through softmax
    # dL/dS = A * (dL/dA - sum_j(A * dL/dA))

    # Element-wise product
    A_dA = A * dL_dA

    # Row sums
    row_sums = jnp.sum(A_dA, axis=-1, keepdims=True)

    # Gradient w.r.t. scores
    dL_dS = A * (dL_dA - row_sums)

    print("Gradient Through Softmax")
    print("=" * 50)
    print()
    print("A * dL/dA (element-wise):")
    print(A_dA)
    print()
    print("Row sums of A * dL/dA:")
    print(row_sums.squeeze())
    print()
    print("dL/dA - rowsum:")
    print(dL_dA - row_sums)
    print()
    print("dL/dS = A * (dL/dA - rowsum):")
    print(dL_dS)
    return A_dA, dL_dS, row_sums


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gradient w.r.t. Queries and Keys

        Finally:

        $$\frac{\partial L}{\partial Q^{kl}} = \frac{1}{\sqrt{d_k}} \sum_j \frac{\partial L}{\partial S^{kj}} K^{jl}$$

        $$\frac{\partial L}{\partial K^{kl}} = \frac{1}{\sqrt{d_k}} \sum_i \frac{\partial L}{\partial S^{ik}} Q^{il}$$

        In matrix form:
        - $\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} \cdot K$
        - $\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S}^T \cdot Q$
        """
    )
    return


@app.cell
def _(K, Q, dL_dS, jnp, scale):
    # Gradient w.r.t. Q
    dL_dQ = scale * jnp.einsum("ij,ja->ia", dL_dS, K)

    # Gradient w.r.t. K
    dL_dK = scale * jnp.einsum("ij,ia->ja", dL_dS, Q)

    print("Gradients w.r.t. Q and K")
    print("=" * 50)
    print()
    print(f"Scale factor: 1/sqrt(d_k) = {scale:.4f}")
    print()
    print("dL/dQ = (1/sqrt(d_k)) * dL/dS @ K:")
    print(dL_dQ)
    print()
    print("dL/dK = (1/sqrt(d_k)) * dL/dS^T @ Q:")
    print(dL_dK)
    return dL_dK, dL_dQ


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Verify with JAX Autodiff
        """
    )
    return


@app.cell
def _(K, Q, V, dL_dK, dL_dQ, dL_dV, jax, jnp):
    def attention_loss(Q, K, V):
        """Attention followed by sum loss."""
        d_k = Q.shape[-1]
        S = jnp.einsum("ia,ja->ij", Q, K) / jnp.sqrt(d_k)
        exp_S = jnp.exp(S - jnp.max(S, axis=-1, keepdims=True))
        A = exp_S / jnp.sum(exp_S, axis=-1, keepdims=True)
        O = jnp.einsum("ij,jb->ib", A, V)
        return jnp.sum(O)

    # Compute gradients with JAX
    grads = jax.grad(attention_loss, argnums=(0, 1, 2))(Q, K, V)
    dL_dQ_jax, dL_dK_jax, dL_dV_jax = grads

    print("Verification with JAX Autodiff")
    print("=" * 50)
    print()
    print("dL/dQ matches:", jnp.allclose(dL_dQ, dL_dQ_jax, atol=1e-5))
    print(f"  Manual: {dL_dQ.flatten()}")
    print(f"  JAX:    {dL_dQ_jax.flatten()}")
    print()
    print("dL/dK matches:", jnp.allclose(dL_dK, dL_dK_jax, atol=1e-5))
    print(f"  Manual: {dL_dK.flatten()}")
    print(f"  JAX:    {dL_dK_jax.flatten()}")
    print()
    print("dL/dV matches:", jnp.allclose(dL_dV, dL_dV_jax, atol=1e-5))
    print(f"  Manual: {dL_dV.flatten()}")
    print(f"  JAX:    {dL_dV_jax.flatten()}")
    return attention_loss, dL_dK_jax, dL_dQ_jax, dL_dV_jax, grads


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Summary of Gradient Formulas

        | Gradient | Formula | Matrix Form |
        |----------|---------|-------------|
        | $\partial L / \partial V$ | $\sum_i A^{ik} \bar{O}^{il}$ | $A^T \bar{O}$ |
        | $\partial L / \partial A$ | $\bar{O}^{mb} V^{nb}$ | $\bar{O} V^T$ |
        | $\partial L / \partial S$ | $A^{ij}(\bar{A}^{ij} - \sum_{j'} A^{ij'}\bar{A}^{ij'})$ | $A \odot (\bar{A} - \text{rowsum}(A \odot \bar{A}))$ |
        | $\partial L / \partial Q$ | $\frac{1}{\sqrt{d_k}} \bar{S}^{ij} K^{jl}$ | $\frac{1}{\sqrt{d_k}} \bar{S} K$ |
        | $\partial L / \partial K$ | $\frac{1}{\sqrt{d_k}} \bar{S}^{ij} Q^{il}$ | $\frac{1}{\sqrt{d_k}} \bar{S}^T Q$ |

        All gradients verified against JAX autodiff!
        """
    )
    return


if __name__ == "__main__":
    app.run()
