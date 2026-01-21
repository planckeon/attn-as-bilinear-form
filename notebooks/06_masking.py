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
        # 6. Attention Masking

        This notebook explores **attention masks**, which control which positions can attend
        to which other positions.

        ## Key Use Cases

        1. **Causal masking**: Prevent attending to future tokens (autoregressive models)
        2. **Padding masks**: Handle variable-length sequences
        3. **Sparse patterns**: Efficient attention for long sequences

        ## Mask Convention

        - `True` = attend (keep)
        - `False` = mask out (set score to $-\infty$ before softmax)
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
        ## Causal Masking

        In autoregressive models (like GPT), position $i$ can only attend to positions $j \leq i$.

        This is represented by a **lower triangular** mask:

        $$M^{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{otherwise} \end{cases}$$
        """
    )
    return


@app.cell
def _(jnp):
    def causal_mask(n):
        """
        Create a causal (lower triangular) mask.

        Position i can attend to positions j <= i.
        """
        return jnp.tril(jnp.ones((n, n), dtype=bool))

    def visualize_mask(mask, title="Mask"):
        """Create ASCII visualization of a mask."""
        rows = []
        for i in range(mask.shape[0]):
            row = "".join(["#" if mask[i, j] else "." for j in range(mask.shape[1])])
            rows.append(f"  {i}: {row}")
        return f"{title}\n" + "\n".join(rows)

    return causal_mask, visualize_mask


@app.cell
def _(causal_mask, visualize_mask):
    # Create a causal mask for sequence length 6
    n = 6
    mask_causal = causal_mask(n)

    print(visualize_mask(mask_causal, "Causal Mask (# = attend, . = masked)"))
    print()
    print("Position 0: can only attend to itself")
    print("Position 5: can attend to all positions 0-5")
    return mask_causal, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applying Masks to Attention

        Masks are applied to attention scores before softmax:

        $$\tilde{S}^{ij} = \begin{cases} S^{ij} & \text{if } M^{ij} = 1 \\ -\infty & \text{if } M^{ij} = 0 \end{cases}$$

        After softmax, masked positions get weight $\approx 0$:
        $$A^{ij} = \text{softmax}_j(\tilde{S}^{ij})$$
        """
    )
    return


@app.cell
def _(jnp):
    def apply_mask(scores, mask, fill_value=-1e9):
        """Apply boolean mask to attention scores."""
        return jnp.where(mask, scores, fill_value)

    def masked_attention(Q, K, V, mask):
        """Attention with masking."""
        d_k = Q.shape[-1]

        # Compute scores
        S = jnp.einsum("ia,ja->ij", Q, K) / jnp.sqrt(d_k)

        # Apply mask
        S_masked = apply_mask(S, mask)

        # Softmax
        S_stable = S_masked - jnp.max(S_masked, axis=-1, keepdims=True)
        exp_S = jnp.exp(S_stable)
        A = exp_S / jnp.sum(exp_S, axis=-1, keepdims=True)

        # Output
        O = jnp.einsum("ij,jb->ib", A, V)

        return O, A, S, S_masked

    return apply_mask, masked_attention


@app.cell
def _(causal_mask, jnp, masked_attention):
    # Example: 4 positions, dimension 3
    n_ex = 4
    d_ex = 3

    # Simple pattern: each position has a distinct embedding
    Q_ex = jnp.eye(n_ex, d_ex)
    K_ex = jnp.eye(n_ex, d_ex)
    V_ex = jnp.arange(n_ex * d_ex).reshape(n_ex, d_ex).astype(float)

    # Create causal mask
    mask_ex = causal_mask(n_ex)

    # Run masked attention
    O_ex, A_ex, S_ex, S_masked_ex = masked_attention(Q_ex, K_ex, V_ex, mask_ex)

    print("Attention with Causal Masking")
    print("=" * 50)
    print()
    print("Raw Scores S^{ij}:")
    print(S_ex)
    print()
    print("Masked Scores (masked -> -inf):")
    print(jnp.where(S_masked_ex < -1e8, float("-inf"), S_masked_ex))
    print()
    print("Attention Weights A^{ij}:")
    print(A_ex)
    print()
    print("Note: Each row sums to 1, but only attends to valid positions")
    return (
        A_ex,
        K_ex,
        O_ex,
        Q_ex,
        S_ex,
        S_masked_ex,
        V_ex,
        d_ex,
        mask_ex,
        n_ex,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Padding Masks

        For batched sequences of different lengths, we mask out padding tokens:
        """
    )
    return


@app.cell
def _(jnp):
    def padding_mask(lengths, max_len):
        """
        Create padding mask from sequence lengths.

        Args:
            lengths: Array of sequence lengths, shape (batch,)
            max_len: Maximum sequence length

        Returns:
            Boolean mask of shape (batch, max_len)
        """
        positions = jnp.arange(max_len)[None, :]  # (1, max_len)
        return positions < lengths[:, None]  # (batch, max_len)

    def attention_mask_from_padding(q_lengths, k_lengths, max_q, max_k):
        """
        Create 2D attention mask from padding masks.

        Position i can attend to position j if both are valid (not padding).
        """
        q_mask = padding_mask(q_lengths, max_q)  # (batch, max_q)
        k_mask = padding_mask(k_lengths, max_k)  # (batch, max_k)

        # Outer product: valid if both query and key are valid
        return q_mask[:, :, None] & k_mask[:, None, :]

    return attention_mask_from_padding, padding_mask


@app.cell
def _(jnp, padding_mask):
    # Example: batch of 3 sequences with lengths 3, 5, 4
    max_len = 6
    lengths = jnp.array([3, 5, 4])

    pad_mask = padding_mask(lengths, max_len)

    print("Padding Mask")
    print("=" * 50)
    print()
    print(f"Sequence lengths: {lengths}")
    print(f"Max length: {max_len}")
    print()
    for b in range(len(lengths)):
        mask_str = "".join(["#" if pad_mask[b, j] else "." for j in range(max_len)])
        print(f"  Batch {b} (len={lengths[b]}): {mask_str}")
    return b, lengths, mask_str, max_len, pad_mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Combined Causal + Padding Mask

        For autoregressive models with variable-length sequences:
        """
    )
    return


@app.cell
def _(jnp):
    def causal_padding_mask(lengths, max_len):
        """
        Combine causal and padding masks.

        Position i can attend to position j if:
        1. j <= i (causal)
        2. Both i and j are not padding
        """
        batch = lengths.shape[0]

        # Causal mask (shared across batch)
        causal = jnp.tril(jnp.ones((max_len, max_len), dtype=bool))

        # Padding mask
        positions = jnp.arange(max_len)
        pad_mask = positions[None, :] < lengths[:, None]  # (batch, max_len)

        # Combine: causal AND both positions valid
        combined = (
            causal[None, :, :]
            & pad_mask[:, :, None]  # query valid
            & pad_mask[:, None, :]  # key valid
        )

        return combined

    return (causal_padding_mask,)


@app.cell
def _(causal_padding_mask, jnp):
    # Example with different sequence lengths
    lengths_cp = jnp.array([4, 6, 3])
    max_len_cp = 6

    combined_mask = causal_padding_mask(lengths_cp, max_len_cp)

    print("Combined Causal + Padding Mask")
    print("=" * 50)
    print()
    for b in range(len(lengths_cp)):
        print(f"Batch {b} (length={lengths_cp[b]}):")
        for i in range(max_len_cp):
            row = "".join(["#" if combined_mask[b, i, j] else "." for j in range(max_len_cp)])
            print(f"  {i}: {row}")
        print()
    return b, combined_mask, i, lengths_cp, max_len_cp, row


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sparse Attention Patterns

        For long sequences, full attention is expensive ($O(n^2)$).
        Sparse patterns reduce this while maintaining model quality.
        """
    )
    return


@app.cell
def _(jnp):
    def local_attention_mask(n, window):
        """
        Local (sliding window) attention.

        Position i attends to positions in [i - window//2, i + window//2].
        """
        i_idx = jnp.arange(n)[:, None]
        j_idx = jnp.arange(n)[None, :]
        return jnp.abs(i_idx - j_idx) <= window // 2

    def strided_attention_mask(n, stride):
        """
        Strided attention pattern.

        Position i attends to positions j where j % stride == i % stride.
        """
        i_idx = jnp.arange(n)[:, None]
        j_idx = jnp.arange(n)[None, :]
        return (i_idx % stride) == (j_idx % stride)

    def block_sparse_mask(n, block_size):
        """
        Block-sparse attention.

        Positions can only attend within their block.
        """
        i_idx = jnp.arange(n)[:, None]
        j_idx = jnp.arange(n)[None, :]
        return (i_idx // block_size) == (j_idx // block_size)

    def global_local_mask(n, window, global_tokens):
        """
        Combined global + local attention (Longformer-style).

        First `global_tokens` attend everywhere;
        other positions use local window.
        """
        # Local attention for all
        local = local_attention_mask(n, window)

        # Global tokens
        global_mask = jnp.zeros((n, n), dtype=bool)
        global_mask = global_mask.at[:global_tokens, :].set(True)  # Global rows
        global_mask = global_mask.at[:, :global_tokens].set(True)  # Global cols

        return local | global_mask

    return (
        block_sparse_mask,
        global_local_mask,
        local_attention_mask,
        strided_attention_mask,
    )


@app.cell
def _(
    block_sparse_mask,
    global_local_mask,
    jnp,
    local_attention_mask,
    strided_attention_mask,
    visualize_mask,
):
    n_sparse = 12

    masks = {
        "Local (window=5)": local_attention_mask(n_sparse, 5),
        "Strided (stride=3)": strided_attention_mask(n_sparse, 3),
        "Block (size=4)": block_sparse_mask(n_sparse, 4),
        "Global+Local (g=2, w=3)": global_local_mask(n_sparse, 3, 2),
    }

    print("Sparse Attention Patterns")
    print("=" * 50)
    print()

    for name, mask in masks.items():
        sparsity = 1 - jnp.mean(mask)
        print(visualize_mask(mask, f"{name} (sparsity={100 * sparsity:.1f}%)"))
        print()
    return mask, masks, n_sparse, name, sparsity


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualization
        """
    )
    return


@app.cell
def _(masks, plt):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, (name, mask) in enumerate(masks.items()):
        ax = axes[idx]
        im = ax.imshow(mask, cmap="Blues", aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    plt.suptitle("Sparse Attention Patterns", fontsize=14)
    plt.tight_layout()
    fig
    return ax, axes, fig, idx, im, mask, name


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Causal with Window (Sliding Window Causal)

        Combine causal constraint with local window for efficient autoregressive attention:
        """
    )
    return


@app.cell
def _(jnp, visualize_mask):
    def causal_window_mask(n, window):
        """
        Causal mask with limited window.

        Position i attends to [max(0, i-window+1), i].
        """
        # Causal: j <= i
        causal = jnp.tril(jnp.ones((n, n), dtype=bool))

        # Window: i - j < window
        i_idx = jnp.arange(n)[:, None]
        j_idx = jnp.arange(n)[None, :]
        window_cond = (i_idx - j_idx) < window

        return causal & window_cond

    # Example
    n_cw = 10
    for window in [3, 5, 10]:
        mask = causal_window_mask(n_cw, window)
        sparsity = 1 - jnp.mean(mask)
        print(visualize_mask(mask, f"Causal Window={window} (sparsity={100 * sparsity:.1f}%)"))
        print()
    return causal_window_mask, mask, n_cw, sparsity, window


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Computational Savings

        Sparsity reduces the number of attention computations:
        """
    )
    return


@app.cell
def _(jnp, masks):
    print("Computational Comparison")
    print("=" * 50)
    print()
    print(f"Sequence length: {12}")
    print(f"Full attention FLOPs: {12 * 12} = n^2")
    print()

    for name, mask in masks.items():
        active = jnp.sum(mask)
        full = mask.shape[0] * mask.shape[1]
        savings = 100 * (1 - active / full)
        print(f"{name}:")
        print(f"  Active positions: {active}/{full} ({100 * active / full:.1f}%)")
        print(f"  FLOP savings: {savings:.1f}%")
        print()
    return active, full, mask, name, savings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        Attention masks control information flow:

        1. **Causal**: Lower triangular, for autoregressive models
        2. **Padding**: Handle variable-length sequences in batches
        3. **Sparse**: Trade some expressiveness for efficiency

        The mask modifies attention as:
        $$A^{ij} = \text{softmax}_j\left(S^{ij} + M^{ij}\right)$$

        where $M^{ij} = 0$ (attend) or $M^{ij} = -\infty$ (mask out).

        Next notebook: Hopfield networks and the connection to associative memory.
        """
    )
    return


if __name__ == "__main__":
    app.run()
