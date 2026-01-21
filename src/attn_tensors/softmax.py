"""
Softmax as Gibbs distribution from statistical mechanics.

The softmax function has deep connections to statistical physics:
- It's the Boltzmann/Gibbs distribution from equilibrium thermodynamics
- The temperature parameter controls the entropy of the distribution
- The partition function normalizes probabilities

Notation:
- S^{ij}: Energy/score (negative energy for high probability)
- T: Temperature parameter (beta = 1/T is inverse temperature)
- Z^i: Partition function (normalization)
- A^{ij}: Probability distribution (attention weights)
"""

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Core Softmax Operations
# =============================================================================


def softmax_rows(x: Array) -> Array:
    """
    Apply softmax along the last axis (rows).

    softmax(x)_j = exp(x_j) / sum_k exp(x_k)

    Uses the log-sum-exp trick for numerical stability:
        softmax(x) = softmax(x - max(x))

    Args:
        x: Input tensor of shape (..., n)

    Returns:
        Softmax probabilities of shape (..., n)
    """
    # Subtract max for numerical stability
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def log_softmax(x: Array) -> Array:
    """
    Compute log-softmax for numerical stability.

    log_softmax(x)_j = x_j - log(sum_k exp(x_k))

    Args:
        x: Input tensor of shape (..., n)

    Returns:
        Log probabilities of shape (..., n)
    """
    x_max = jnp.max(x, axis=-1, keepdims=True)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(x - x_max), axis=-1, keepdims=True)) + x_max
    return x - log_sum_exp


# =============================================================================
# Gibbs Distribution (Statistical Mechanics View)
# =============================================================================


def gibbs_distribution(
    energies: Array,
    temperature: float = 1.0,
    mask: Array | None = None,
) -> Array:
    """
    Compute the Gibbs/Boltzmann distribution.

    In statistical mechanics, for a system with energy levels E_j,
    the probability of state j at temperature T is:

        P(j) = exp(-E_j / T) / Z

    where Z = sum_j exp(-E_j / T) is the partition function.

    For attention, we use S = -E (scores are negative energies):

        A^{ij} = exp(S^{ij} / T) / Z^i

    Args:
        energies: Score matrix S^{ij} of shape (..., n)
                  (positive = preferred, like negative energy)
        temperature: Temperature T > 0
        mask: Boolean mask, True = keep, False = mask (set to -inf)

    Returns:
        Probability distribution of shape (..., n)
    """
    # Scale by temperature
    scaled = energies / temperature

    # Apply mask (set masked positions to -inf so exp(-inf) = 0)
    if mask is not None:
        scaled = jnp.where(mask, scaled, -jnp.inf)

    return softmax_rows(scaled)


def partition_function(energies: Array, temperature: float = 1.0) -> Array:
    """
    Compute the partition function Z = sum_j exp(S_j / T).

    The partition function is fundamental in statistical mechanics:
    - It normalizes the Gibbs distribution
    - Its log gives the free energy: F = -T log Z
    - Derivatives give thermodynamic quantities

    Args:
        energies: Score/energy values of shape (..., n)
        temperature: Temperature parameter

    Returns:
        Partition function Z of shape (...)
    """
    scaled = energies / temperature
    # Use log-sum-exp for stability, then exp
    x_max = jnp.max(scaled, axis=-1, keepdims=True)
    return jnp.sum(jnp.exp(scaled - x_max), axis=-1) * jnp.exp(x_max.squeeze(-1))


def free_energy(energies: Array, temperature: float = 1.0) -> Array:
    """
    Compute the free energy F = -T log Z.

    The free energy combines energy and entropy:
        F = <E> - T * S

    where <E> is the expected energy and S is the entropy.

    Args:
        energies: Score/energy values of shape (..., n)
        temperature: Temperature parameter

    Returns:
        Free energy F of shape (...)
    """
    scaled = energies / temperature
    # log Z using log-sum-exp
    x_max = jnp.max(scaled, axis=-1)
    log_Z = jnp.log(jnp.sum(jnp.exp(scaled - x_max[..., None]), axis=-1)) + x_max
    return -temperature * log_Z


# =============================================================================
# Entropy and Information Theory
# =============================================================================


def entropy(probs: Array, eps: float = 1e-12) -> Array:
    """
    Compute the Shannon entropy of a distribution.

    H = -sum_j P_j log P_j

    In statistical mechanics, this is related to the Gibbs entropy.

    Args:
        probs: Probability distribution of shape (..., n)
        eps: Small constant for numerical stability

    Returns:
        Entropy H of shape (...)
    """
    # Clip probabilities to avoid log(0)
    p = jnp.clip(probs, eps, 1.0)
    return -jnp.sum(probs * jnp.log(p), axis=-1)


def attention_entropy(A: Array, eps: float = 1e-12) -> Array:
    """
    Compute entropy of attention weights for each query.

    H^i = -sum_j A^{ij} log A^{ij}

    High entropy = diffuse attention (uniform over keys)
    Low entropy = focused attention (peaked on few keys)

    Args:
        A: Attention weights of shape (n_q, n_k)
        eps: Numerical stability constant

    Returns:
        Per-query entropy of shape (n_q,)
    """
    return entropy(A, eps)


def max_entropy(n: int) -> float:
    """
    Compute maximum entropy for n states (uniform distribution).

    H_max = log(n)

    Args:
        n: Number of states

    Returns:
        Maximum entropy value
    """
    return float(jnp.log(n))


def normalized_entropy(A: Array, eps: float = 1e-12) -> Array:
    """
    Compute normalized entropy in [0, 1].

    H_norm = H / log(n)

    0 = delta distribution (all attention on one key)
    1 = uniform distribution (equal attention on all keys)

    Args:
        A: Attention weights of shape (..., n)
        eps: Numerical stability constant

    Returns:
        Normalized entropy in [0, 1]
    """
    n = A.shape[-1]
    H = entropy(A, eps)
    return H / jnp.log(n)


# =============================================================================
# Temperature Effects
# =============================================================================


def temperature_sweep(
    energies: Array,
    temperatures: Array,
) -> Array:
    """
    Compute Gibbs distributions at multiple temperatures.

    Useful for understanding the temperature dependence:
    - T -> 0: argmax (hard attention)
    - T = 1: standard softmax
    - T -> inf: uniform distribution

    Args:
        energies: Score values of shape (n,)
        temperatures: Array of temperatures of shape (n_temps,)

    Returns:
        Distributions of shape (n_temps, n)
    """
    # Shape: (n_temps, n)
    scaled = energies[None, :] / temperatures[:, None]
    return softmax_rows(scaled)


def effective_number_of_states(probs: Array) -> Array:
    """
    Compute effective number of states (perplexity).

    n_eff = exp(H) = exp(-sum_j P_j log P_j)

    This gives the number of states that "effectively" contribute
    to the distribution. For attention:
    - n_eff = 1: attending to exactly one key
    - n_eff = n: uniform attention over all keys

    Args:
        probs: Probability distribution of shape (..., n)

    Returns:
        Effective number of states
    """
    H = entropy(probs)
    return jnp.exp(H)


# =============================================================================
# Softmax Jacobian (for gradient derivations)
# =============================================================================


def softmax_jacobian(probs: Array) -> Array:
    """
    Compute the Jacobian of softmax: dA_j / dS_k

    The Jacobian is:
        dA_j/dS_k = A_j (delta_{jk} - A_k)

    This is fundamental for backpropagation through attention.

    Args:
        probs: Softmax output A of shape (n,)

    Returns:
        Jacobian matrix of shape (n, n)
    """
    n = probs.shape[0]
    # Outer product: A_j * A_k
    outer = jnp.outer(probs, probs)
    # Diagonal: A_j * delta_{jk}
    diag = jnp.diag(probs)
    # J_{jk} = A_j (delta_{jk} - A_k)
    return diag - outer


def softmax_jacobian_batched(probs: Array) -> Array:
    """
    Batch version of softmax Jacobian.

    For attention weights A^{ij}, compute:
        dA^{ij}/dS^{ik} = A^{ij} (delta_{jk} - A^{ik})

    Args:
        probs: Attention weights of shape (n_q, n_k)

    Returns:
        Jacobian tensor of shape (n_q, n_k, n_k)
    """
    # probs: (n_q, n_k)
    # Output: (n_q, n_k, n_k) where [i, j, k] = dA^{ij}/dS^{ik}

    n_q, n_k = probs.shape

    # Diagonal part: A^{ij} * delta_{jk}
    diag = jnp.einsum("ij,jk->ijk", probs, jnp.eye(n_k))

    # Outer product part: A^{ij} * A^{ik}
    outer = jnp.einsum("ij,ik->ijk", probs, probs)

    return diag - outer
