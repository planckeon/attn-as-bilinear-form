"""
Bilinear forms and metric tensors for attention mechanisms.

This module implements the tensor calculus foundations:
- Metric tensors g_{ab} for measuring similarity
- Index raising/lowering operations
- Bilinear form computations: B(u,v) = u^a g_{ab} v^b

Notation conventions:
- Superscripts (contravariant): u^a, Q^{ia}
- Subscripts (covariant): g_{ab}, u_a
- Einstein summation: repeated indices are summed
"""

from typing import Literal

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Metric Tensors
# =============================================================================


def euclidean_metric(d: int) -> Array:
    """
    Euclidean (identity) metric: g_{ab} = delta_{ab}

    This is the simplest metric, giving standard dot product.

    Args:
        d: Dimension of the feature space

    Returns:
        Identity matrix of shape (d, d)
    """
    return jnp.eye(d)


def scaled_euclidean_metric(d: int, scale: float | None = None) -> Array:
    """
    Scaled Euclidean metric: g_{ab} = (1/sqrt(d)) * delta_{ab}

    This is the metric used in standard attention (Vaswani et al., 2017).
    The 1/sqrt(d_k) scaling prevents dot products from growing too large
    in high dimensions, keeping softmax gradients in a good range.

    Args:
        d: Dimension of the feature space
        scale: Custom scale factor. If None, uses 1/sqrt(d)

    Returns:
        Scaled identity matrix of shape (d, d)
    """
    if scale is None:
        scale = float(1.0 / jnp.sqrt(d))
    return scale * jnp.eye(d)


def learned_metric(W: Array) -> Array:
    """
    Learned bilinear metric: g_{ab} = W^T W

    This ensures the metric is positive semi-definite.
    The metric can be parameterized by a lower-triangular matrix W
    for a unique Cholesky-like decomposition.

    Args:
        W: Weight matrix of shape (r, d) where r is the rank

    Returns:
        Metric tensor of shape (d, d)
    """
    return jnp.einsum("ra,rb->ab", W, W)


def diagonal_metric(diag: Array) -> Array:
    """
    Diagonal metric: g_{ab} = diag(sigma)_{ab}

    This gives axis-aligned feature importance weighting.

    Args:
        diag: Diagonal elements of shape (d,), must be positive

    Returns:
        Diagonal metric of shape (d, d)
    """
    return jnp.diag(diag)


# =============================================================================
# Index Operations
# =============================================================================


def lower_index(v_upper: Array, g: Array) -> Array:
    """
    Lower a contravariant index: v_a = g_{ab} v^b

    Converts a vector with upper index to one with lower index.

    Args:
        v_upper: Contravariant vector v^a of shape (..., d)
        g: Metric tensor g_{ab} of shape (d, d)

    Returns:
        Covariant vector v_a of shape (..., d)
    """
    return jnp.einsum("ab,...b->...a", g, v_upper)


def raise_index(v_lower: Array, g_inv: Array) -> Array:
    """
    Raise a covariant index: v^a = g^{ab} v_b

    Converts a vector with lower index to one with upper index.

    Args:
        v_lower: Covariant vector v_a of shape (..., d)
        g_inv: Inverse metric g^{ab} of shape (d, d)

    Returns:
        Contravariant vector v^a of shape (..., d)
    """
    return jnp.einsum("ab,...b->...a", g_inv, v_lower)


def metric_inverse(g: Array) -> Array:
    """
    Compute inverse metric: g^{ab} such that g^{ac}g_{cb} = delta^a_b

    Args:
        g: Metric tensor g_{ab} of shape (d, d)

    Returns:
        Inverse metric g^{ab} of shape (d, d)
    """
    return jnp.linalg.inv(g)


# =============================================================================
# Bilinear Forms
# =============================================================================


def bilinear_form(u: Array, v: Array, g: Array) -> Array:
    """
    Compute bilinear form: B(u, v) = u^a g_{ab} v^b

    This is the core operation in attention: computing similarity
    between query and key vectors under a given metric.

    Args:
        u: First vector(s) u^a of shape (..., d)
        v: Second vector(s) v^b of shape (..., d)
        g: Metric tensor g_{ab} of shape (d, d)

    Returns:
        Scalar(s) representing the bilinear form value
    """
    # First lower the index of v: v_a = g_{ab} v^b
    v_lower = jnp.einsum("ab,...b->...a", g, v)
    # Then contract with u: u^a v_a
    return jnp.einsum("...a,...a->...", u, v_lower)


def bilinear_form_batch(Q: Array, K: Array, g: Array) -> Array:
    """
    Batch bilinear form for attention scores: S^{ij} = Q^{ia} g_{ab} K^{jb}

    Computes all pairwise bilinear forms between query positions i
    and key positions j.

    Args:
        Q: Queries of shape (n_q, d_k) with indices Q^{ia}
        K: Keys of shape (n_k, d_k) with indices K^{ja}
        g: Metric tensor g_{ab} of shape (d_k, d_k)

    Returns:
        Attention scores S^{ij} of shape (n_q, n_k)
    """
    # K_lower_{ja} = g_{ab} K^{jb}
    K_lower = jnp.einsum("ab,jb->ja", g, K)
    # S^{ij} = Q^{ia} K_{ja}
    return jnp.einsum("ia,ja->ij", Q, K_lower)


def quadratic_form(v: Array, g: Array) -> Array:
    """
    Compute quadratic form: Q(v) = v^a g_{ab} v^b = ||v||_g^2

    This is the squared norm of v under the metric g.

    Args:
        v: Vector v^a of shape (..., d)
        g: Metric tensor g_{ab} of shape (d, d)

    Returns:
        Scalar(s) representing the squared norm
    """
    return bilinear_form(v, v, g)


# =============================================================================
# Utility Functions
# =============================================================================


def inner_product(
    u: Array, v: Array, metric_type: Literal["euclidean", "scaled"] = "scaled"
) -> Array:
    """
    Compute inner product with standard metrics.

    Convenience function for the most common cases.

    Args:
        u: First vector(s) of shape (..., d)
        v: Second vector(s) of shape (..., d)
        metric_type: "euclidean" for identity, "scaled" for 1/sqrt(d) scaling

    Returns:
        Inner product value(s)
    """
    d = u.shape[-1]
    if metric_type == "euclidean":
        g = euclidean_metric(d)
    else:
        g = scaled_euclidean_metric(d)
    return bilinear_form(u, v, g)


def verify_metric_properties(g: Array, tol: float = 1e-6) -> dict[str, bool | float]:
    """
    Verify that a matrix satisfies metric tensor properties.

    A valid metric tensor must be:
    1. Symmetric: g_{ab} = g_{ba}
    2. Positive definite: v^a g_{ab} v^b > 0 for all v != 0

    Args:
        g: Candidate metric tensor of shape (d, d)
        tol: Tolerance for numerical comparisons

    Returns:
        Dictionary with verification results
    """
    # Check symmetry
    is_symmetric = jnp.allclose(g, g.T, atol=tol)

    # Check positive definiteness via eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(g)
    is_positive_definite = jnp.all(eigenvalues > -tol)

    return {
        "symmetric": bool(is_symmetric),
        "positive_definite": bool(is_positive_definite),
        "min_eigenvalue": float(jnp.min(eigenvalues)),
        "valid_metric": bool(is_symmetric and is_positive_definite),
    }
