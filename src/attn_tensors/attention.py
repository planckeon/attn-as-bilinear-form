"""
Attention mechanism forward pass with explicit tensor operations.

This module implements scaled dot-product attention using einsum notation
that directly corresponds to the index notation in the theory.

Index conventions:
- i: query sequence position (n_q positions)
- j: key/value sequence position (n_k positions)
- a, b: feature/embedding dimensions
- h: attention head index
"""

import jax.numpy as jnp
from jax import Array

from .bilinear import bilinear_form_batch
from .softmax import gibbs_distribution, softmax_rows

# =============================================================================
# Core Attention Operations
# =============================================================================


def attention_scores(Q: Array, K: Array, scale: bool = True) -> Array:
    """
    Compute raw attention scores: S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)

    In index notation with implicit metric:
        S^{ij} = Q^{ia} g_{ab} K^{jb}

    where g_{ab} = (1/sqrt(d_k)) * delta_{ab} for scaled attention.

    Args:
        Q: Query tensor of shape (n_q, d_k)
        K: Key tensor of shape (n_k, d_k)
        scale: Whether to apply 1/sqrt(d_k) scaling

    Returns:
        Score matrix S^{ij} of shape (n_q, n_k)
    """
    d_k = Q.shape[-1]

    # S^{ij} = Q^{ia} K^{ja} (contraction over feature index a)
    S = jnp.einsum("ia,ja->ij", Q, K)

    if scale:
        S = S / jnp.sqrt(d_k)

    return S


def attention_scores_with_metric(Q: Array, K: Array, g: Array) -> Array:
    """
    Compute attention scores with explicit metric tensor.

    S^{ij} = Q^{ia} g_{ab} K^{jb}

    This generalizes standard attention to arbitrary bilinear forms.

    Args:
        Q: Query tensor of shape (n_q, d_k)
        K: Key tensor of shape (n_k, d_k)
        g: Metric tensor g_{ab} of shape (d_k, d_k)

    Returns:
        Score matrix S^{ij} of shape (n_q, n_k)
    """
    return bilinear_form_batch(Q, K, g)


def attention_weights(S: Array, mask: Array | None = None, temperature: float = 1.0) -> Array:
    """
    Convert scores to attention weights via softmax (Gibbs distribution).

    A^{ij} = exp(S^{ij} / T) / Z^i

    where Z^i = sum_j exp(S^{ij} / T) is the partition function.

    Args:
        S: Score matrix of shape (n_q, n_k)
        mask: Optional boolean mask, True = keep, False = mask out
        temperature: Temperature parameter T (default 1.0)

    Returns:
        Attention weight matrix A^{ij} of shape (n_q, n_k)
    """
    return gibbs_distribution(S, mask=mask, temperature=temperature)


def attention_output(A: Array, V: Array) -> Array:
    """
    Compute attention output by weighted sum of values.

    O^{ib} = A^{ij} V^{jb}

    Args:
        A: Attention weights of shape (n_q, n_k)
        V: Value tensor of shape (n_k, d_v)

    Returns:
        Output tensor O^{ib} of shape (n_q, d_v)
    """
    return jnp.einsum("ij,jb->ib", A, V)


# =============================================================================
# Full Attention Functions
# =============================================================================


def scaled_dot_product_attention(
    Q: Array,
    K: Array,
    V: Array,
    mask: Array | None = None,
    temperature: float = 1.0,
    return_weights: bool = False,
) -> Array | tuple[Array, Array]:
    """
    Full scaled dot-product attention (Vaswani et al., 2017).

    Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    In index notation:
        S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)     # Scores
        A^{ij} = exp(S^{ij}) / sum_k exp(S^{ik})  # Weights (Gibbs)
        O^{ib} = A^{ij} V^{jb}                    # Output

    Args:
        Q: Queries of shape (n_q, d_k)
        K: Keys of shape (n_k, d_k)
        V: Values of shape (n_k, d_v)
        mask: Optional attention mask
        temperature: Softmax temperature
        return_weights: If True, also return attention weights

    Returns:
        Output of shape (n_q, d_v), optionally with weights
    """
    # Step 1: Compute scores S^{ij}
    S = attention_scores(Q, K, scale=True)

    # Step 2: Apply softmax to get weights A^{ij}
    A = attention_weights(S, mask=mask, temperature=temperature)

    # Step 3: Compute output O^{ib}
    O = attention_output(A, V)

    if return_weights:
        return O, A
    return O


def bilinear_attention(
    Q: Array,
    K: Array,
    V: Array,
    g: Array,
    mask: Array | None = None,
    temperature: float = 1.0,
    return_weights: bool = False,
) -> Array | tuple[Array, Array]:
    """
    Generalized attention with explicit metric tensor.

    S^{ij} = Q^{ia} g_{ab} K^{jb}
    A^{ij} = softmax_j(S^{ij})
    O^{ib} = A^{ij} V^{jb}

    This allows for learned or structured similarity metrics beyond
    the standard scaled dot product.

    Args:
        Q: Queries of shape (n_q, d_k)
        K: Keys of shape (n_k, d_k)
        V: Values of shape (n_k, d_v)
        g: Metric tensor g_{ab} of shape (d_k, d_k)
        mask: Optional attention mask
        temperature: Softmax temperature
        return_weights: If True, also return attention weights

    Returns:
        Output of shape (n_q, d_v), optionally with weights
    """
    # Step 1: Compute scores with metric
    S = attention_scores_with_metric(Q, K, g)

    # Step 2: Apply softmax
    A = attention_weights(S, mask=mask, temperature=temperature)

    # Step 3: Compute output
    O = attention_output(A, V)

    if return_weights:
        return O, A
    return O


# =============================================================================
# Batched Attention
# =============================================================================


def batched_attention(
    Q: Array,
    K: Array,
    V: Array,
    mask: Array | None = None,
) -> Array:
    """
    Batched scaled dot-product attention.

    Adds batch dimension b:
        S^{bij} = Q^{bia} K^{bja} / sqrt(d_k)
        A^{bij} = softmax_j(S^{bij})
        O^{bib} = A^{bij} V^{bjc}

    Args:
        Q: Queries of shape (batch, n_q, d_k)
        K: Keys of shape (batch, n_k, d_k)
        V: Values of shape (batch, n_k, d_v)
        mask: Optional mask of shape (batch, n_q, n_k) or (n_q, n_k)

    Returns:
        Output of shape (batch, n_q, d_v)
    """
    d_k = Q.shape[-1]

    # S^{bij} = Q^{bia} K^{bja} / sqrt(d_k)
    S = jnp.einsum("bia,bja->bij", Q, K) / jnp.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        S = jnp.where(mask, S, -1e9)

    # A^{bij} = softmax_j(S^{bij})
    A = softmax_rows(S)

    # O^{bic} = A^{bij} V^{bjc}
    O = jnp.einsum("bij,bjc->bic", A, V)

    return O


# =============================================================================
# Attention Decomposition (for analysis)
# =============================================================================


def decompose_attention(
    Q: Array,
    K: Array,
    V: Array,
) -> dict[str, Array]:
    """
    Decompose attention into its constituent tensor operations.

    Returns all intermediate tensors for analysis and visualization.

    Args:
        Q: Queries of shape (n_q, d_k)
        K: Keys of shape (n_k, d_k)
        V: Values of shape (n_k, d_v)

    Returns:
        Dictionary with:
            - scores: Raw attention scores S^{ij}
            - weights: Normalized attention weights A^{ij}
            - output: Final output O^{ib}
            - partition: Partition function Z^i
            - scale: Scaling factor 1/sqrt(d_k)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)

    # Raw scores (unscaled)
    S_raw = jnp.einsum("ia,ja->ij", Q, K)

    # Scaled scores
    S = S_raw * scale

    # Partition function Z^i = sum_j exp(S^{ij})
    Z = jnp.sum(jnp.exp(S), axis=-1, keepdims=True)

    # Attention weights
    A = jnp.exp(S) / Z

    # Output
    O = jnp.einsum("ij,jb->ib", A, V)

    return {
        "scores_raw": S_raw,
        "scores": S,
        "weights": A,
        "output": O,
        "partition": Z.squeeze(-1),
        "scale": scale,
    }
