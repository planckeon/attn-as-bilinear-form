"""
Hopfield network connections to attention.

Modern Hopfield networks (Ramsauer et al., 2020) establish a deep connection
between attention mechanisms and associative memory. Key insights:

1. Attention is energy-based: softmax over scores is a Gibbs distribution
2. Update rule: new state = softmax(beta * patterns^T @ state) @ patterns
3. This is exactly the attention mechanism!
4. Memory capacity scales exponentially with dimension (vs linearly for classical)

Reference: "Hopfield Networks is All You Need" (arXiv:2008.02217)
"""

import jax.numpy as jnp
from jax import Array

from .softmax import gibbs_distribution

# =============================================================================
# Classical Hopfield Networks (for comparison)
# =============================================================================


def classical_hopfield_energy(state: Array, patterns: Array) -> float:
    """
    Classical Hopfield network energy.

    E(x) = -0.5 * sum_{i,j} W_{ij} x_i x_j
         = -0.5 * x^T W x

    where W = (1/N) * sum_mu xi_mu xi_mu^T (Hebbian learning)

    Args:
        state: Binary state vector x of shape (N,) with values in {-1, +1}
        patterns: Stored patterns of shape (M, N)

    Returns:
        Energy value (scalar)
    """
    N = state.shape[0]

    # Hebbian weight matrix: W = (1/N) * sum_mu xi_mu @ xi_mu^T
    W = jnp.einsum("mi,mj->ij", patterns, patterns) / N

    # Energy: E = -0.5 * x^T W x
    return float(-0.5 * jnp.einsum("i,ij,j->", state, W, state))


def classical_hopfield_update(state: Array, patterns: Array) -> Array:
    """
    Asynchronous update for classical Hopfield network.

    Update rule: x_i <- sign(sum_j W_{ij} x_j)

    This finds fixed points that are (hopefully) stored patterns.

    Args:
        state: Current state of shape (N,)
        patterns: Stored patterns of shape (M, N)

    Returns:
        Updated state (one full sweep)
    """
    N = state.shape[0]
    W = jnp.einsum("mi,mj->ij", patterns, patterns) / N

    # h_i = sum_j W_{ij} x_j
    h = jnp.einsum("ij,j->i", W, state)

    return jnp.sign(h)


# =============================================================================
# Modern Hopfield Networks (Exponential Capacity)
# =============================================================================


def modern_hopfield_energy(state: Array, patterns: Array, beta: float = 1.0) -> float:
    """
    Modern Hopfield network energy (Ramsauer et al., 2020).

    E(x) = -lse(beta * patterns @ x) + 0.5 * ||x||^2 + const

    where lse is the log-sum-exp function:
        lse(z) = log(sum_mu exp(z_mu))

    The key insight: lse is smooth maximum, so energy is minimized
    when x aligns with one of the patterns.

    Args:
        state: Continuous state vector x of shape (d,)
        patterns: Stored patterns of shape (M, d)
        beta: Inverse temperature (higher = sharper memory)

    Returns:
        Energy value
    """
    # Similarities: s_mu = patterns_mu @ x
    similarities = jnp.einsum("ma,a->m", patterns, state)  # (M,)

    # Log-sum-exp: smoothed maximum
    lse = jnp.max(beta * similarities) + jnp.log(
        jnp.sum(jnp.exp(beta * similarities - jnp.max(beta * similarities)))
    )

    # Energy: -lse + 0.5 * ||x||^2
    return float(-lse / beta + 0.5 * jnp.sum(state**2))


def modern_hopfield_update(state: Array, patterns: Array, beta: float = 1.0) -> Array:
    """
    Update rule for modern Hopfield network.

    x_new = patterns^T @ softmax(beta * patterns @ x)

    This is EXACTLY the attention mechanism!
    - Query: x (current state)
    - Keys: patterns
    - Values: patterns
    - Temperature: 1/beta

    Args:
        state: Current state of shape (d,)
        patterns: Stored patterns of shape (M, d)
        beta: Inverse temperature

    Returns:
        Updated state of shape (d,)
    """
    # Similarities (attention scores)
    similarities = jnp.einsum("ma,a->m", patterns, state)  # (M,)

    # Attention weights (Gibbs distribution)
    weights = gibbs_distribution(similarities, temperature=1.0 / beta)  # (M,)

    # Weighted combination (attention output)
    new_state = jnp.einsum("m,ma->a", weights, patterns)  # (d,)

    return new_state


def hopfield_retrieve(
    query: Array,
    patterns: Array,
    beta: float = 1.0,
    max_iterations: int = 10,
    tol: float = 1e-6,
) -> tuple[Array, int]:
    """
    Retrieve a stored pattern given a query.

    Iterates the Hopfield update until convergence.

    Args:
        query: Initial query/probe of shape (d,)
        patterns: Stored patterns of shape (M, d)
        beta: Inverse temperature
        max_iterations: Maximum update iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (retrieved pattern, number of iterations)
    """
    state = query

    for i in range(max_iterations):
        new_state = modern_hopfield_update(state, patterns, beta)

        # Check convergence
        if jnp.max(jnp.abs(new_state - state)) < tol:
            return new_state, i + 1

        state = new_state

    return state, max_iterations


# =============================================================================
# Attention as Hopfield
# =============================================================================


def attention_as_hopfield(
    queries: Array,
    keys: Array,
    values: Array,
    beta: float | None = None,
) -> Array:
    """
    Interpret attention as Hopfield network retrieval.

    The mapping:
    - Keys (K): Stored patterns for similarity computation
    - Values (V): Stored patterns for retrieval
    - Queries (Q): Probe states
    - Output: Retrieved patterns

    Attention(Q, K, V) = softmax(beta * Q @ K^T) @ V

    This is parallel Hopfield retrieval for all queries simultaneously,
    with separate keys and values (hetero-associative memory).

    Args:
        queries: Query states of shape (n_q, d)
        keys: Key patterns of shape (n_k, d)
        values: Value patterns of shape (n_k, d_v)
        beta: Inverse temperature (default: 1/sqrt(d))

    Returns:
        Retrieved values of shape (n_q, d_v)
    """
    d = queries.shape[-1]
    if beta is None:
        beta = float(1.0 / jnp.sqrt(d))

    # Similarities
    scores = jnp.einsum("qa,ka->qk", queries, keys) * beta

    # Retrieval weights
    weights = gibbs_distribution(scores, temperature=1.0)

    # Retrieved values
    return jnp.einsum("qk,kv->qv", weights, values)


# =============================================================================
# Memory Capacity Analysis
# =============================================================================


def classical_capacity(N: int) -> float:
    """
    Theoretical capacity of classical Hopfield network.

    M_max ≈ 0.14 * N

    where N is the dimension and M is the number of patterns.

    Args:
        N: Pattern dimension

    Returns:
        Maximum number of reliably stored patterns
    """
    return 0.14 * N


def modern_capacity(d: int, epsilon: float = 0.01) -> float:
    """
    Theoretical capacity of modern Hopfield network.

    M_max ≈ exp(0.5 * d) for continuous patterns

    The capacity scales exponentially with dimension!

    Args:
        d: Pattern dimension
        epsilon: Error tolerance

    Returns:
        Approximate maximum capacity
    """
    # From Ramsauer et al., 2020: M = O(d^(d/2))
    # Simplified exponential approximation
    return float(jnp.exp(0.5 * d))


def separation_quality(patterns: Array) -> float:
    """
    Measure how well-separated stored patterns are.

    Well-separated patterns lead to reliable retrieval.
    Uses minimum pairwise distance.

    Args:
        patterns: Stored patterns of shape (M, d)

    Returns:
        Minimum normalized distance between any two patterns
    """
    M = patterns.shape[0]

    # Pairwise distances
    diff = patterns[:, None, :] - patterns[None, :, :]  # (M, M, d)
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (M, M)

    # Mask diagonal (self-distance) - use where to avoid inf * 0 = nan
    diagonal_mask = jnp.eye(M, dtype=jnp.bool_)
    distances = jnp.where(diagonal_mask, jnp.inf, distances)

    min_distance = jnp.min(distances)

    # Normalize by average pattern norm
    avg_norm = jnp.mean(jnp.linalg.norm(patterns, axis=-1))

    return float(min_distance / avg_norm)


# =============================================================================
# Visualization Helpers
# =============================================================================


def energy_landscape_1d(
    patterns: Array,
    axis: int = 0,
    n_points: int = 100,
    beta: float = 1.0,
) -> tuple[Array, Array]:
    """
    Compute energy along one axis for visualization.

    Args:
        patterns: Stored patterns of shape (M, d)
        axis: Which dimension to vary
        n_points: Number of evaluation points
        beta: Inverse temperature

    Returns:
        Tuple of (x values, energy values)
    """
    d = patterns.shape[1]

    # Create probe states varying along one axis
    x = jnp.linspace(-2, 2, n_points)
    states = jnp.zeros((n_points, d))
    states = states.at[:, axis].set(x)

    # Compute energy at each point
    energies = jnp.array([modern_hopfield_energy(s, patterns, beta) for s in states])

    return x, energies


def retrieval_basin(
    pattern_idx: int,
    patterns: Array,
    n_samples: int = 100,
    noise_scale: float = 0.5,
    beta: float = 1.0,
    key=None,
) -> float:
    """
    Estimate the basin of attraction for a stored pattern.

    Measures what fraction of noisy queries retrieve the correct pattern.

    Args:
        pattern_idx: Index of pattern to test
        patterns: All stored patterns of shape (M, d)
        n_samples: Number of random queries
        noise_scale: Scale of noise added to pattern
        beta: Inverse temperature
        key: JAX random key

    Returns:
        Fraction of successful retrievals
    """
    import jax.random as random

    if key is None:
        key = random.PRNGKey(0)

    target = patterns[pattern_idx]

    # Generate noisy queries
    noise = random.normal(key, (n_samples, patterns.shape[1])) * noise_scale
    queries = target + noise

    # Retrieve for each query
    successes = 0
    for q in queries:
        retrieved, _ = hopfield_retrieve(q, patterns, beta)

        # Check if closest to target pattern
        distances = jnp.sum((retrieved - patterns) ** 2, axis=-1)
        closest = jnp.argmin(distances)

        if closest == pattern_idx:
            successes += 1

    return successes / n_samples
