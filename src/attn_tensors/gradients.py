"""
Manual gradient derivations for attention mechanisms.

This module implements the gradients derived by hand using index notation,
then verifies them against JAX's autodiff. This serves both as:
1. Verification that our derivations are correct
2. Educational material showing the explicit gradient formulas

Gradient naming convention:
    dL_dX = partial L / partial X (upstream gradient from loss L)

Index conventions:
    i: query position
    j: key/value position
    a, b: feature dimensions
    k, l: general indices in gradient expressions
"""

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial


# =============================================================================
# Score Gradients: dL/dQ, dL/dK from dL/dS
# =============================================================================


def grad_scores_wrt_Q(dL_dS: Array, K: Array, scale: float) -> Array:
    """
    Gradient of loss w.r.t. queries, given gradient w.r.t. scores.

    Given: S^{ij} = (1/sqrt(d_k)) Q^{ia} K^{ja}

    Derivation:
        dS^{ij}/dQ^{kl} = (1/sqrt(d_k)) * delta^i_k * delta^a_l * K^{ja}
                       = (1/sqrt(d_k)) * delta^i_k * K^{jl}

    Therefore:
        dL/dQ^{kl} = dL/dS^{ij} * dS^{ij}/dQ^{kl}
                   = (1/sqrt(d_k)) * dL/dS^{kj} * K^{jl}

    In matrix form: dL/dQ = (1/sqrt(d_k)) * (dL/dS) @ K

    Args:
        dL_dS: Gradient w.r.t. scores, shape (n_q, n_k)
        K: Keys, shape (n_k, d_k)
        scale: Scaling factor (1/sqrt(d_k))

    Returns:
        Gradient w.r.t. Q, shape (n_q, d_k)
    """
    # dL/dQ^{il} = scale * dL/dS^{ij} * K^{jl}
    return scale * jnp.einsum("ij,jl->il", dL_dS, K)


def grad_scores_wrt_K(dL_dS: Array, Q: Array, scale: float) -> Array:
    """
    Gradient of loss w.r.t. keys, given gradient w.r.t. scores.

    Given: S^{ij} = (1/sqrt(d_k)) Q^{ia} K^{ja}

    Derivation:
        dS^{ij}/dK^{kl} = (1/sqrt(d_k)) * Q^{ia} * delta^j_k * delta^a_l
                       = (1/sqrt(d_k)) * delta^j_k * Q^{il}

    Therefore:
        dL/dK^{kl} = dL/dS^{ij} * dS^{ij}/dK^{kl}
                   = (1/sqrt(d_k)) * dL/dS^{ik} * Q^{il}

    In matrix form: dL/dK = (1/sqrt(d_k)) * (dL/dS)^T @ Q

    Args:
        dL_dS: Gradient w.r.t. scores, shape (n_q, n_k)
        Q: Queries, shape (n_q, d_k)
        scale: Scaling factor (1/sqrt(d_k))

    Returns:
        Gradient w.r.t. K, shape (n_k, d_k)
    """
    # dL/dK^{jl} = scale * dL/dS^{ij} * Q^{il}
    return scale * jnp.einsum("ij,il->jl", dL_dS, Q)


# =============================================================================
# Softmax Gradients: dL/dS from dL/dA
# =============================================================================


def grad_softmax(dL_dA: Array, A: Array) -> Array:
    """
    Gradient through softmax: dL/dS from dL/dA.

    Given: A^{ij} = softmax_j(S^{ij}) = exp(S^{ij}) / sum_k exp(S^{ik})

    The Jacobian is:
        dA^{ij}/dS^{mn} = delta^i_m * A^{ij} * (delta^j_n - A^{in})

    Therefore:
        dL/dS^{mn} = sum_{ij} dL/dA^{ij} * dA^{ij}/dS^{mn}
                   = dL/dA^{mj} * A^{mj} * (delta^j_n - A^{mn})
                   = A^{mn} * (dL/dA^{mn} - sum_j A^{mj} * dL/dA^{mj})

    In compact form:
        dL/dS = A * (dL/dA - sum_j(A * dL/dA))

    Args:
        dL_dA: Gradient w.r.t. attention weights, shape (n_q, n_k)
        A: Attention weights, shape (n_q, n_k)

    Returns:
        Gradient w.r.t. scores, shape (n_q, n_k)
    """
    # Compute sum_j A^{ij} * dL/dA^{ij} for each i
    sum_term = jnp.sum(A * dL_dA, axis=-1, keepdims=True)  # (n_q, 1)

    # dL/dS^{ij} = A^{ij} * (dL/dA^{ij} - sum_term^i)
    return A * (dL_dA - sum_term)


# =============================================================================
# Value Gradients: dL/dA, dL/dV from dL/dO
# =============================================================================


def grad_output_wrt_A(dL_dO: Array, V: Array) -> Array:
    """
    Gradient w.r.t. attention weights from output gradient.

    Given: O^{ib} = A^{ij} V^{jb}

    Derivation:
        dO^{ib}/dA^{mn} = delta^i_m * delta^j_n * V^{jb}
                       = delta^i_m * V^{nb}

    Therefore:
        dL/dA^{mn} = dL/dO^{ib} * dO^{ib}/dA^{mn}
                   = dL/dO^{mb} * V^{nb}

    In matrix form: dL/dA = dL/dO @ V^T

    Args:
        dL_dO: Gradient w.r.t. output, shape (n_q, d_v)
        V: Values, shape (n_k, d_v)

    Returns:
        Gradient w.r.t. A, shape (n_q, n_k)
    """
    # dL/dA^{ij} = dL/dO^{ib} * V^{jb}
    return jnp.einsum("ib,jb->ij", dL_dO, V)


def grad_output_wrt_V(dL_dO: Array, A: Array) -> Array:
    """
    Gradient w.r.t. values from output gradient.

    Given: O^{ib} = A^{ij} V^{jb}

    Derivation:
        dO^{ib}/dV^{mn} = A^{ij} * delta^j_m * delta^b_n
                       = A^{im} * delta^b_n

    Therefore:
        dL/dV^{mn} = dL/dO^{ib} * dO^{ib}/dV^{mn}
                   = dL/dO^{in} * A^{im}

    In matrix form: dL/dV = A^T @ dL/dO

    Args:
        dL_dO: Gradient w.r.t. output, shape (n_q, d_v)
        A: Attention weights, shape (n_q, n_k)

    Returns:
        Gradient w.r.t. V, shape (n_k, d_v)
    """
    # dL/dV^{jb} = A^{ij} * dL/dO^{ib}
    return jnp.einsum("ij,ib->jb", A, dL_dO)


# =============================================================================
# Full Attention Backward Pass
# =============================================================================


def attention_backward(
    dL_dO: Array,
    Q: Array,
    K: Array,
    V: Array,
    A: Array,
    S: Array,
) -> tuple[Array, Array, Array]:
    """
    Full backward pass through attention, using manual gradient formulas.

    Forward pass:
        S^{ij} = Q^{ia} K^{ja} / sqrt(d_k)
        A^{ij} = softmax_j(S^{ij})
        O^{ib} = A^{ij} V^{jb}

    Backward pass:
        dL/dV = A^T @ dL/dO
        dL/dA = dL/dO @ V^T
        dL/dS = A * (dL/dA - rowsum(A * dL/dA))
        dL/dQ = (1/sqrt(d_k)) * dL/dS @ K
        dL/dK = (1/sqrt(d_k)) * (dL/dS)^T @ Q

    Args:
        dL_dO: Upstream gradient w.r.t. output, shape (n_q, d_v)
        Q: Queries, shape (n_q, d_k)
        K: Keys, shape (n_k, d_k)
        V: Values, shape (n_k, d_v)
        A: Cached attention weights, shape (n_q, n_k)
        S: Cached scores (not strictly needed, but for completeness)

    Returns:
        Tuple of (dL/dQ, dL/dK, dL/dV)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)

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


# =============================================================================
# Verification Against JAX Autodiff
# =============================================================================


def verify_gradients(
    Q: Array,
    K: Array,
    V: Array,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, bool]:
    """
    Verify manual gradients against JAX autodiff.

    Uses a simple L2 loss on the output for verification:
        L = sum_{ib} O^{ib}^2 / 2

    Args:
        Q: Queries, shape (n_q, d_k)
        K: Keys, shape (n_k, d_k)
        V: Values, shape (n_k, d_v)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with verification results for each gradient
    """
    from .attention import scaled_dot_product_attention

    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)

    # Forward pass with cached intermediates
    S = jnp.einsum("ia,ja->ij", Q, K) * scale
    A = jax.nn.softmax(S, axis=-1)
    O = jnp.einsum("ij,jb->ib", A, V)

    # Simple loss: L = ||O||^2 / 2
    def loss_fn(Q, K, V):
        O = scaled_dot_product_attention(Q, K, V)
        return 0.5 * jnp.sum(O**2)

    # Get JAX gradients
    dL_dQ_jax, dL_dK_jax, dL_dV_jax = jax.grad(loss_fn, argnums=(0, 1, 2))(Q, K, V)

    # Get manual gradients
    # For L = ||O||^2 / 2, we have dL/dO = O
    dL_dO = O
    dL_dQ_manual, dL_dK_manual, dL_dV_manual = attention_backward(dL_dO, Q, K, V, A, S)

    # Compare
    results = {
        "dL_dQ": bool(jnp.allclose(dL_dQ_jax, dL_dQ_manual, rtol=rtol, atol=atol)),
        "dL_dK": bool(jnp.allclose(dL_dK_jax, dL_dK_manual, rtol=rtol, atol=atol)),
        "dL_dV": bool(jnp.allclose(dL_dV_jax, dL_dV_manual, rtol=rtol, atol=atol)),
    }
    results["all_correct"] = all(results.values())

    return results


def gradient_numerical_check(
    Q: Array,
    K: Array,
    V: Array,
    eps: float = 1e-5,
) -> dict[str, Array]:
    """
    Numerical gradient check using finite differences.

    For each parameter, compute:
        dL/dP ≈ (L(P + eps) - L(P - eps)) / (2 * eps)

    This is slower but provides an independent verification.

    Args:
        Q: Queries, shape (n_q, d_k)
        K: Keys, shape (n_k, d_k)
        V: Values, shape (n_k, d_v)
        eps: Finite difference step size

    Returns:
        Dictionary with numerical gradients and comparison errors
    """
    from .attention import scaled_dot_product_attention

    def loss_fn(Q, K, V):
        O = scaled_dot_product_attention(Q, K, V)
        return 0.5 * jnp.sum(O**2)

    # Numerical gradient for Q
    dL_dQ_num = jnp.zeros_like(Q)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            Q_plus = Q.at[i, j].add(eps)
            Q_minus = Q.at[i, j].add(-eps)
            dL_dQ_num = dL_dQ_num.at[i, j].set(
                (loss_fn(Q_plus, K, V) - loss_fn(Q_minus, K, V)) / (2 * eps)
            )

    # Get analytic gradient
    dL_dQ_analytic = jax.grad(loss_fn, argnums=0)(Q, K, V)

    return {
        "numerical": dL_dQ_num,
        "analytic": dL_dQ_analytic,
        "max_error": float(jnp.max(jnp.abs(dL_dQ_num - dL_dQ_analytic))),
        "rel_error": float(
            jnp.max(jnp.abs(dL_dQ_num - dL_dQ_analytic) / (jnp.abs(dL_dQ_analytic) + 1e-8))
        ),
    }


# =============================================================================
# Individual Gradient Components (for education/analysis)
# =============================================================================


def gradient_flow_analysis(
    Q: Array,
    K: Array,
    V: Array,
) -> dict[str, Array]:
    """
    Analyze gradient flow through attention for understanding.

    Returns all intermediate gradients and their statistics.

    Args:
        Q, K, V: Attention inputs

    Returns:
        Dictionary with gradient tensors and statistics
    """
    from .attention import scaled_dot_product_attention

    d_k = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d_k)

    # Forward pass
    S = jnp.einsum("ia,ja->ij", Q, K) * scale
    A = jax.nn.softmax(S, axis=-1)
    O = jnp.einsum("ij,jb->ib", A, V)

    # Backward with L = ||O||^2 / 2
    dL_dO = O  # dL/dO = O

    # Each step of backprop
    dL_dV = grad_output_wrt_V(dL_dO, A)
    dL_dA = grad_output_wrt_A(dL_dO, V)
    dL_dS = grad_softmax(dL_dA, A)
    dL_dQ = grad_scores_wrt_Q(dL_dS, K, scale)
    dL_dK = grad_scores_wrt_K(dL_dS, Q, scale)

    return {
        # Forward quantities
        "S": S,
        "A": A,
        "O": O,
        # Gradients
        "dL_dO": dL_dO,
        "dL_dV": dL_dV,
        "dL_dA": dL_dA,
        "dL_dS": dL_dS,
        "dL_dQ": dL_dQ,
        "dL_dK": dL_dK,
        # Statistics
        "grad_norms": {
            "dL_dQ": float(jnp.linalg.norm(dL_dQ)),
            "dL_dK": float(jnp.linalg.norm(dL_dK)),
            "dL_dV": float(jnp.linalg.norm(dL_dV)),
        },
    }
