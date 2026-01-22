"""
Multi-head attention implementation.

Multi-head attention extends single-head by:
1. Learning multiple projection matrices (heads)
2. Computing attention independently in each head
3. Concatenating and projecting the results

Index conventions:
    h: head index (H heads)
    i: query sequence position
    j: key/value sequence position
    a: input embedding dimension (d_model)
    b: per-head key/query dimension (d_k = d_model / H)
    c: per-head value dimension (d_v = d_model / H)
"""

import jax.numpy as jnp
from jax import Array

from .softmax import softmax_rows

# =============================================================================
# Multi-Head Attention Core
# =============================================================================


def multihead_attention(
    Q: Array,
    K: Array,
    V: Array,
    W_Q: Array,
    W_K: Array,
    W_V: Array,
    W_O: Array,
    mask: Array | None = None,
    return_weights: bool = False,
) -> Array | tuple[Array, Array]:
    """
    Multi-head attention (Vaswani et al., 2017).

    For each head h:
        Q_h = X @ W_Q^h      (project queries)
        K_h = X @ W_K^h      (project keys)
        V_h = X @ W_V^h      (project values)
        head_h = Attention(Q_h, K_h, V_h)

    Then concatenate and project:
        output = Concat(head_1, ..., head_H) @ W_O

    In index notation with head index h:
        Q^{hia} = Q^{ib} W_Q^{hba}
        K^{hja} = K^{jb} W_K^{hba}
        V^{hjc} = V^{jb} W_V^{hbc}
        S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
        A^{hij} = softmax_j(S^{hij})
        O^{hic} = A^{hij} V^{hjc}
        Y^{ia} = O^{hic} W_O^{hca}  (sum over h, c)

    Args:
        Q: Query input of shape (n_q, d_model)
        K: Key input of shape (n_k, d_model)
        V: Value input of shape (n_k, d_model)
        W_Q: Query projections of shape (H, d_model, d_k)
        W_K: Key projections of shape (H, d_model, d_k)
        W_V: Value projections of shape (H, d_model, d_v)
        W_O: Output projection of shape (H, d_v, d_model)
        mask: Optional attention mask of shape (n_q, n_k)
        return_weights: If True, also return attention weights

    Returns:
        Output of shape (n_q, d_model), optionally with weights
    """
    W_Q.shape[0]
    d_k = W_Q.shape[2]

    # Project Q, K, V for all heads
    # Q^{hia} = Q^{ib} W_Q^{hba}
    Q_h = jnp.einsum("ib,hba->hia", Q, W_Q)  # (H, n_q, d_k)
    K_h = jnp.einsum("jb,hba->hja", K, W_K)  # (H, n_k, d_k)
    V_h = jnp.einsum("jb,hbc->hjc", V, W_V)  # (H, n_k, d_v)

    # Compute attention scores: S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
    S = jnp.einsum("hia,hja->hij", Q_h, K_h) / jnp.sqrt(d_k)  # (H, n_q, n_k)

    # Apply mask if provided
    if mask is not None:
        S = jnp.where(mask[None, :, :], S, -1e9)

    # Softmax over keys: A^{hij} = softmax_j(S^{hij})
    A = softmax_rows(S)  # (H, n_q, n_k)

    # Weighted sum of values: O^{hic} = A^{hij} V^{hjc}
    O = jnp.einsum("hij,hjc->hic", A, V_h)  # (H, n_q, d_v)

    # Concatenate and project: Y^{ia} = O^{hic} W_O^{hca}
    Y = jnp.einsum("hic,hca->ia", O, W_O)  # (n_q, d_model)

    if return_weights:
        return Y, A
    return Y


def multihead_attention_batched(
    Q: Array,
    K: Array,
    V: Array,
    W_Q: Array,
    W_K: Array,
    W_V: Array,
    W_O: Array,
    mask: Array | None = None,
) -> Array:
    """
    Batched multi-head attention.

    Adds batch dimension b:
        Q: (batch, n_q, d_model)
        K: (batch, n_k, d_model)
        V: (batch, n_k, d_model)

    Args:
        Q: Query input of shape (batch, n_q, d_model)
        K: Key input of shape (batch, n_k, d_model)
        V: Value input of shape (batch, n_k, d_model)
        W_Q, W_K, W_V, W_O: Projection matrices (shared across batch)
        mask: Optional mask of shape (batch, n_q, n_k) or (n_q, n_k)

    Returns:
        Output of shape (batch, n_q, d_model)
    """
    d_k = W_Q.shape[2]

    # Project: Q_h^{bhia} = Q^{bid} W_Q^{hda}
    # (b=batch, i=seq, d=d_model, h=head, a=d_k)
    Q_h = jnp.einsum("bid,hda->bhia", Q, W_Q)
    K_h = jnp.einsum("bjd,hda->bhja", K, W_K)
    V_h = jnp.einsum("bjd,hdc->bhjc", V, W_V)

    # Scores: S^{bhij} = Q^{bhia} K^{bhja} / sqrt(d_k)
    S = jnp.einsum("bhia,bhja->bhij", Q_h, K_h) / jnp.sqrt(d_k)

    # Mask
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]  # Broadcast
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]  # Add head dim
        S = jnp.where(mask, S, -1e9)

    # Attention: A^{bhij} = softmax_j(S^{bhij})
    A = softmax_rows(S)

    # Output: O^{bhic} = A^{bhij} V^{bhjc}
    O = jnp.einsum("bhij,bhjc->bhic", A, V_h)

    # Project: Y^{bid} = O^{bhic} W_O^{hcd}
    Y = jnp.einsum("bhic,hcd->bid", O, W_O)

    return Y


# =============================================================================
# Weight Initialization
# =============================================================================


def init_multihead_weights(
    key,
    d_model: int,
    num_heads: int,
    d_k: int | None = None,
    d_v: int | None = None,
) -> dict[str, Array]:
    """
    Initialize multi-head attention weights.

    Uses Xavier/Glorot initialization for stable gradients.

    Args:
        key: JAX random key
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_k: Per-head key/query dimension (default: d_model // num_heads)
        d_v: Per-head value dimension (default: d_model // num_heads)

    Returns:
        Dictionary with W_Q, W_K, W_V, W_O
    """
    import jax.random as random

    if d_k is None:
        d_k = d_model // num_heads
    if d_v is None:
        d_v = d_model // num_heads

    keys = random.split(key, 4)

    # Xavier initialization scale
    scale_qk = jnp.sqrt(2.0 / (d_model + d_k))
    scale_v = jnp.sqrt(2.0 / (d_model + d_v))
    scale_o = jnp.sqrt(2.0 / (num_heads * d_v + d_model))

    return {
        "W_Q": random.normal(keys[0], (num_heads, d_model, d_k)) * scale_qk,
        "W_K": random.normal(keys[1], (num_heads, d_model, d_k)) * scale_qk,
        "W_V": random.normal(keys[2], (num_heads, d_model, d_v)) * scale_v,
        "W_O": random.normal(keys[3], (num_heads, d_v, d_model)) * scale_o,
    }


# =============================================================================
# Analysis Functions
# =============================================================================


def head_diversity(A: Array) -> float:
    """
    Measure diversity of attention patterns across heads.

    Uses average pairwise cosine distance between head attention patterns.
    Higher values indicate more diverse heads.

    Args:
        A: Attention weights of shape (H, n_q, n_k)

    Returns:
        Diversity score in [0, 1]
    """
    H = A.shape[0]

    # Flatten attention patterns: (H, n_q * n_k)
    A_flat = A.reshape(H, -1)

    # Normalize to unit vectors
    A_norm = A_flat / (jnp.linalg.norm(A_flat, axis=-1, keepdims=True) + 1e-8)

    # Pairwise cosine similarities
    similarities = jnp.einsum("hi,ji->hj", A_norm, A_norm)

    # Average off-diagonal (exclude self-similarity)
    mask = 1 - jnp.eye(H)
    avg_similarity = jnp.sum(similarities * mask) / (H * (H - 1))

    # Convert to diversity (1 - similarity)
    return float(1 - avg_similarity)


def head_attention_entropy(A: Array) -> Array:
    """
    Compute attention entropy for each head.

    Args:
        A: Attention weights of shape (H, n_q, n_k)

    Returns:
        Mean entropy per head of shape (H,)
    """
    from .softmax import entropy

    # Entropy for each (head, query) pair
    H_per_query = entropy(A)  # (H, n_q)

    # Average over queries
    return jnp.mean(H_per_query, axis=-1)  # (H,)


from typing import Any


def decompose_multihead(
    Q: Array,
    K: Array,
    V: Array,
    W_Q: Array,
    W_K: Array,
    W_V: Array,
    W_O: Array,
) -> dict[str, Any]:
    """
    Decompose multi-head attention for analysis.

    Args:
        Q, K, V: Input tensors
        W_Q, W_K, W_V, W_O: Projection matrices

    Returns:
        Dictionary with all intermediate tensors
    """
    d_k = W_Q.shape[2]

    # Projections
    Q_h = jnp.einsum("ib,hba->hia", Q, W_Q)
    K_h = jnp.einsum("jb,hba->hja", K, W_K)
    V_h = jnp.einsum("jb,hbc->hjc", V, W_V)

    # Scores
    S = jnp.einsum("hia,hja->hij", Q_h, K_h) / jnp.sqrt(d_k)

    # Attention weights
    A = softmax_rows(S)

    # Per-head outputs
    O_h = jnp.einsum("hij,hjc->hic", A, V_h)

    # Final output
    Y = jnp.einsum("hic,hca->ia", O_h, W_O)

    return {
        "Q_heads": Q_h,
        "K_heads": K_h,
        "V_heads": V_h,
        "scores": S,
        "weights": A,
        "head_outputs": O_h,
        "output": Y,
        "head_diversity": head_diversity(A),
        "head_entropy": head_attention_entropy(A),
    }
