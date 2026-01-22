"""Shared test fixtures for attn-tensors.

This file is auto-loaded by pytest - use for fixtures only.
Shared helper functions and strategies are in helpers.py.
"""

import jax.random as random
import pytest
from hypothesis import Phase, Verbosity, settings

# =============================================================================
# Hypothesis Configuration
# =============================================================================

# Configure hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=100,
    phases=[Phase.generate, Phase.target],
    deadline=None,  # Disable deadline for JAX JIT compilation
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
)
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
)

# Load CI profile by default (can override with HYPOTHESIS_PROFILE env var)
settings.load_profile("ci")


# =============================================================================
# JAX Fixtures
# =============================================================================


@pytest.fixture
def rng_key():
    """Base random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def rng_keys(rng_key):
    """Multiple random keys for tests needing several random sources."""
    return random.split(rng_key, 10)


# =============================================================================
# Sample Data Fixtures - Parameterized
# =============================================================================


@pytest.fixture(
    params=[
        (2, 3, 4),  # Tiny: 2 queries, 3 keys, dim 4
        (4, 6, 8),  # Small: 4 queries, 6 keys, dim 8
        (8, 12, 16),  # Medium: 8 queries, 12 keys, dim 16
    ]
)
def sample_qkv(request, rng_key):
    """Parameterized Q, K, V tensors of various sizes."""
    n_q, n_k, d = request.param
    keys = random.split(rng_key, 3)
    return {
        "Q": random.normal(keys[0], (n_q, d)),
        "K": random.normal(keys[1], (n_k, d)),
        "V": random.normal(keys[2], (n_k, d)),
        "n_q": n_q,
        "n_k": n_k,
        "d": d,
    }


@pytest.fixture(params=[4, 8, 16])
def sample_dimension(request):
    """Common dimensions for testing."""
    return request.param


@pytest.fixture(params=[2, 4, 8])
def num_heads(request):
    """Number of attention heads for multi-head tests."""
    return request.param


@pytest.fixture
def sample_multihead_config(rng_key):
    """Standard multi-head attention configuration."""
    return {
        "d_model": 32,
        "num_heads": 4,
        "d_k": 8,
        "d_v": 8,
        "n_seq": 6,
        "key": rng_key,
    }


# =============================================================================
# Backend Detection
# =============================================================================


def has_mlx():
    """Check if MLX is available (Apple Silicon only)."""
    try:
        import mlx  # noqa: F401

        return True
    except ImportError:
        return False


# Skip decorator for MLX-only tests
requires_mlx = pytest.mark.skipif(
    not has_mlx(), reason="MLX not available (requires Apple Silicon)"
)
