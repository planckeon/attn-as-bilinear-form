"""
Attention masking utilities.

Masking is essential for:
1. Causal attention (autoregressive models) - prevent attending to future
2. Padding masks - handle variable-length sequences
3. Sparse patterns - efficient attention approximations

Mask convention:
    True = attend (keep)
    False = mask out (set to -inf before softmax)
"""

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Causal Masks (Autoregressive)
# =============================================================================


def causal_mask(n: int) -> Array:
    """
    Create a causal (lower triangular) mask.

    Prevents position i from attending to positions j > i.
    Used in autoregressive models like GPT.

    Visual (4x4):
        [[T, F, F, F],
         [T, T, F, F],
         [T, T, T, F],
         [T, T, T, T]]

    Args:
        n: Sequence length

    Returns:
        Boolean mask of shape (n, n)
    """
    return jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))


def causal_mask_with_window(n: int, window: int) -> Array:
    """
    Causal mask with limited attention window.

    Position i can only attend to positions in [max(0, i-window+1), i].
    Useful for long sequences where full attention is too expensive.

    Args:
        n: Sequence length
        window: Attention window size

    Returns:
        Boolean mask of shape (n, n)
    """
    # Start with causal mask
    mask = causal_mask(n)

    # Create window mask: |i - j| < window
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]
    window_mask = (i_idx - j_idx) < window

    return mask & window_mask


# =============================================================================
# Padding Masks
# =============================================================================


def padding_mask(lengths: Array, max_len: int) -> Array:
    """
    Create padding mask from sequence lengths.

    For variable-length sequences, mask out padding tokens.

    Args:
        lengths: Array of sequence lengths, shape (batch,)
        max_len: Maximum sequence length

    Returns:
        Boolean mask of shape (batch, max_len)
    """
    positions = jnp.arange(max_len)[None, :]  # (1, max_len)
    return positions < lengths[:, None]  # (batch, max_len)


def attention_mask_from_padding(
    query_lengths: Array,
    key_lengths: Array,
    max_q: int,
    max_k: int,
) -> Array:
    """
    Create 2D attention mask from padding masks.

    Position i can attend to position j if both are valid (not padding).

    Args:
        query_lengths: Query sequence lengths, shape (batch,)
        key_lengths: Key sequence lengths, shape (batch,)
        max_q: Maximum query length
        max_k: Maximum key length

    Returns:
        Boolean mask of shape (batch, max_q, max_k)
    """
    q_mask = padding_mask(query_lengths, max_q)  # (batch, max_q)
    k_mask = padding_mask(key_lengths, max_k)  # (batch, max_k)

    # Outer product: valid if both query and key positions are valid
    return q_mask[:, :, None] & k_mask[:, None, :]


# =============================================================================
# Combined Masks
# =============================================================================


def causal_padding_mask(lengths: Array, max_len: int) -> Array:
    """
    Combine causal and padding masks.

    Used in autoregressive models with variable-length sequences.

    Args:
        lengths: Sequence lengths, shape (batch,)
        max_len: Maximum sequence length

    Returns:
        Boolean mask of shape (batch, max_len, max_len)
    """
    # Causal mask (shared across batch)
    causal = causal_mask(max_len)  # (max_len, max_len)

    # Padding mask
    pad_mask = padding_mask(lengths, max_len)  # (batch, max_len)

    # Combine: need both causal and non-padding
    # Query padding: (batch, max_len, 1)
    # Key padding: (batch, 1, max_len)
    combined = causal[None, :, :] & pad_mask[:, :, None] & pad_mask[:, None, :]

    return combined


# =============================================================================
# Sparse Patterns
# =============================================================================


def local_attention_mask(n: int, window: int) -> Array:
    """
    Local attention mask (sliding window).

    Position i attends to positions in [i - window//2, i + window//2].
    Unlike causal, this is bidirectional.

    Args:
        n: Sequence length
        window: Total window size (should be odd for symmetry)

    Returns:
        Boolean mask of shape (n, n)
    """
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]

    half_window = window // 2
    return jnp.abs(i_idx - j_idx) <= half_window


def strided_attention_mask(n: int, stride: int) -> Array:
    """
    Strided attention mask.

    Position i attends to positions j where j % stride == i % stride.
    Creates a pattern where positions attend to every stride-th token.

    Args:
        n: Sequence length
        stride: Stride value

    Returns:
        Boolean mask of shape (n, n)
    """
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]

    return (i_idx % stride) == (j_idx % stride)


def block_sparse_mask(n: int, block_size: int) -> Array:
    """
    Block-sparse attention mask.

    Divides sequence into blocks of size block_size.
    Positions can only attend within their block.

    Args:
        n: Sequence length
        block_size: Size of each block

    Returns:
        Boolean mask of shape (n, n)
    """
    i_idx = jnp.arange(n)[:, None]
    j_idx = jnp.arange(n)[None, :]

    return (i_idx // block_size) == (j_idx // block_size)


def global_local_mask(n: int, window: int, global_tokens: int) -> Array:
    """
    Combined global and local attention (like Longformer).

    First `global_tokens` positions attend to all positions (global).
    Other positions use local attention with the given window.

    Args:
        n: Sequence length
        window: Local attention window size
        global_tokens: Number of global attention tokens at the start

    Returns:
        Boolean mask of shape (n, n)
    """
    # Local attention for all
    local = local_attention_mask(n, window)

    # Global tokens can attend everywhere
    global_mask = jnp.zeros((n, n), dtype=jnp.bool_)
    global_mask = global_mask.at[:global_tokens, :].set(True)  # Global rows
    global_mask = global_mask.at[:, :global_tokens].set(True)  # Global cols

    return local | global_mask


# =============================================================================
# Mask Utilities
# =============================================================================


def apply_mask(scores: Array, mask: Array, fill_value: float = -1e9) -> Array:
    """
    Apply boolean mask to attention scores.

    Sets masked positions to fill_value (large negative for softmax).

    Args:
        scores: Attention scores of shape (..., n_q, n_k)
        mask: Boolean mask of shape (..., n_q, n_k)
        fill_value: Value for masked positions

    Returns:
        Masked scores
    """
    return jnp.where(mask, scores, fill_value)


def mask_to_additive(mask: Array, fill_value: float = -1e9) -> Array:
    """
    Convert boolean mask to additive mask.

    Returns 0 where True, fill_value where False.
    Can be added directly to scores: scores + additive_mask

    Args:
        mask: Boolean mask
        fill_value: Value for masked positions

    Returns:
        Additive mask
    """
    return jnp.where(mask, 0.0, fill_value)


def visualize_mask(mask: Array) -> str:
    """
    Create ASCII visualization of a mask.

    Args:
        mask: Boolean mask of shape (n_q, n_k)

    Returns:
        String representation
    """
    rows = []
    for i in range(mask.shape[0]):
        row = "".join(["#" if mask[i, j] else "." for j in range(mask.shape[1])])
        rows.append(row)
    return "\n".join(rows)
