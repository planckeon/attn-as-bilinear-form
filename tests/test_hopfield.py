"""Tests for Hopfield network module."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from attn_tensors.hopfield import (
    attention_as_hopfield,
    classical_capacity,
    classical_hopfield_energy,
    classical_hopfield_update,
    energy_landscape_1d,
    hopfield_retrieve,
    modern_capacity,
    modern_hopfield_energy,
    modern_hopfield_update,
    retrieval_basin,
    separation_quality,
)

from .helpers import (
    assert_allclose,
    assert_finite,
    assert_shape,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_patterns(rng_key):
    """Simple orthogonal patterns for testing."""
    # Create somewhat separated patterns
    patterns = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return patterns


@pytest.fixture
def random_patterns(rng_key):
    """Random patterns for testing."""
    M, d = 5, 16  # 5 patterns of dimension 16
    return random.normal(rng_key, (M, d))


# =============================================================================
# Classical Hopfield Tests
# =============================================================================


class TestClassicalHopfieldEnergy:
    """Tests for classical_hopfield_energy function."""

    def test_returns_scalar(self, simple_patterns):
        """Energy should be a scalar."""
        state = jnp.array([1.0, 0.0, 0.0, 0.0])
        E = classical_hopfield_energy(state, simple_patterns)

        assert jnp.isscalar(E) or E.shape == (), "Energy should be scalar"

    def test_stored_pattern_low_energy(self, simple_patterns):
        """Stored patterns should have lower energy than random states."""
        # Energy at stored pattern
        E_stored = classical_hopfield_energy(simple_patterns[0], simple_patterns)

        # Energy at random state
        random_state = jnp.array([0.5, 0.5, 0.5, 0.5])
        E_random = classical_hopfield_energy(random_state, simple_patterns)

        # Stored patterns should be local minima
        assert E_stored <= E_random + 0.1, "Stored patterns should have low energy"

    def test_energy_finite(self, random_patterns, rng_key):
        """Energy should always be finite."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        E = classical_hopfield_energy(state, random_patterns)

        assert jnp.isfinite(E), "Energy should be finite"


class TestClassicalHopfieldUpdate:
    """Tests for classical_hopfield_update function."""

    def test_output_shape(self, simple_patterns):
        """Updated state should have same shape as input."""
        state = jnp.array([1.0, -1.0, 1.0, -1.0])
        new_state = classical_hopfield_update(state, simple_patterns)

        assert_shape(new_state, state.shape, "updated state")

    def test_output_is_binary(self, simple_patterns):
        """Updated state should be in {-1, 0, +1}^N (0 when h=0 exactly)."""
        state = jnp.array([0.5, -0.3, 0.8, -0.1])
        new_state = classical_hopfield_update(state, simple_patterns)

        # All values should be -1, 0, or +1 (sign function returns 0 for h=0)
        assert jnp.all((new_state == 1.0) | (new_state == -1.0) | (new_state == 0.0))

    def test_fixed_point_for_patterns(self, rng_key):
        """Stored patterns should be (approximately) fixed points."""
        # Create simple patterns
        patterns = jnp.array(
            [
                [1.0, 1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0],
            ]
        )

        # Update from pattern should return similar pattern
        updated = classical_hopfield_update(patterns[0], patterns)

        # May not be exact due to interference, but should be similar
        correlation = (
            jnp.dot(patterns[0], updated) / jnp.linalg.norm(patterns[0]) / jnp.linalg.norm(updated)
        )
        assert correlation > 0.5, "Pattern should be close to fixed point"


# =============================================================================
# Modern Hopfield Tests
# =============================================================================


class TestModernHopfieldEnergy:
    """Tests for modern_hopfield_energy function."""

    def test_returns_scalar(self, random_patterns, rng_key):
        """Energy should be a scalar."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        E = modern_hopfield_energy(state, random_patterns)

        assert jnp.isscalar(E) or E.shape == (), "Energy should be scalar"

    def test_energy_finite(self, random_patterns, rng_key):
        """Energy should be finite for bounded inputs."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        E = modern_hopfield_energy(state, random_patterns, beta=1.0)

        assert jnp.isfinite(E), "Energy should be finite"

    def test_pattern_is_local_minimum(self, simple_patterns):
        """Stored patterns should be local energy minima."""
        pattern = simple_patterns[0]
        beta = 10.0  # High beta for sharper energy wells

        # Energy at pattern
        E_pattern = modern_hopfield_energy(pattern, simple_patterns, beta)

        # Energy at slightly perturbed state
        perturbed = pattern + 0.1 * jnp.array([0.1, 0.1, 0.1, 0.1])
        E_perturbed = modern_hopfield_energy(perturbed, simple_patterns, beta)

        assert E_pattern <= E_perturbed + 0.01, "Pattern should be local minimum"

    def test_beta_effect(self, random_patterns, rng_key):
        """Higher beta should give sharper energy wells."""
        state = random.normal(rng_key, (random_patterns.shape[1],))

        E_low_beta = modern_hopfield_energy(state, random_patterns, beta=0.1)
        E_high_beta = modern_hopfield_energy(state, random_patterns, beta=10.0)

        # Both should be finite
        assert jnp.isfinite(E_low_beta)
        assert jnp.isfinite(E_high_beta)


class TestModernHopfieldUpdate:
    """Tests for modern_hopfield_update function."""

    def test_output_shape(self, random_patterns, rng_key):
        """Updated state should have same shape as input."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        new_state = modern_hopfield_update(state, random_patterns)

        assert_shape(new_state, state.shape, "updated state")

    def test_output_finite(self, random_patterns, rng_key):
        """Updated state should be finite."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        new_state = modern_hopfield_update(state, random_patterns)

        assert_finite(new_state, "updated state")

    def test_converges_to_pattern(self, simple_patterns):
        """Update should converge toward nearest pattern."""
        # Start near pattern 0
        query = simple_patterns[0] + 0.1 * jnp.array([0.1, 0.1, 0.1, 0.1])
        beta = 10.0

        # Multiple updates
        state = query
        for _ in range(5):
            state = modern_hopfield_update(state, simple_patterns, beta)

        # Should be closer to pattern 0
        dist_0 = jnp.sum((state - simple_patterns[0]) ** 2)
        dist_1 = jnp.sum((state - simple_patterns[1]) ** 2)
        dist_2 = jnp.sum((state - simple_patterns[2]) ** 2)

        assert dist_0 < dist_1 and dist_0 < dist_2, "Should converge to nearest pattern"

    def test_is_attention(self, random_patterns, rng_key):
        """Modern Hopfield update is exactly attention."""
        state = random.normal(rng_key, (random_patterns.shape[1],))
        beta = 2.0

        # Hopfield update
        new_state = modern_hopfield_update(state, random_patterns, beta)

        # Manual attention computation
        similarities = jnp.einsum("ma,a->m", random_patterns, state)
        weights = jax.nn.softmax(beta * similarities)
        expected = jnp.einsum("m,ma->a", weights, random_patterns)

        assert_allclose(new_state, expected, err_msg="Hopfield update should equal attention")


class TestHopfieldRetrieve:
    """Tests for hopfield_retrieve function."""

    def test_returns_state_and_iterations(self, simple_patterns):
        """Should return retrieved state and iteration count."""
        query = simple_patterns[0] * 0.9
        result, iterations = hopfield_retrieve(query, simple_patterns)

        assert_shape(result, query.shape, "retrieved state")
        assert isinstance(iterations, (int, np.integer, jnp.ndarray))

    def test_exact_pattern_retrieval(self, simple_patterns):
        """Exact pattern query should retrieve in few iterations."""
        query = simple_patterns[0]
        result, iterations = hopfield_retrieve(query, simple_patterns, beta=10.0)

        assert iterations <= 3, f"Exact pattern should converge quickly, got {iterations}"

    def test_noisy_retrieval(self, simple_patterns, rng_key):
        """Should retrieve correct pattern from noisy query."""
        noise = random.normal(rng_key, simple_patterns[0].shape) * 0.2
        query = simple_patterns[0] + noise
        beta = 10.0

        result, _ = hopfield_retrieve(query, simple_patterns, beta=beta)

        # Should be closest to pattern 0
        distances = jnp.sum((result - simple_patterns) ** 2, axis=-1)
        closest = jnp.argmin(distances)

        assert closest == 0, f"Should retrieve pattern 0, got {closest}"

    def test_respects_max_iterations(self, random_patterns, rng_key):
        """Should not exceed max_iterations."""
        query = random.normal(rng_key, (random_patterns.shape[1],))
        max_iter = 5

        _, iterations = hopfield_retrieve(query, random_patterns, max_iterations=max_iter)

        assert iterations <= max_iter


# =============================================================================
# Attention as Hopfield Tests
# =============================================================================


class TestAttentionAsHopfield:
    """Tests for attention_as_hopfield function."""

    def test_output_shape(self, rng_key):
        """Output should have shape (n_q, d_v)."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d, d_v = 4, 6, 8, 10

        queries = random.normal(keys[0], (n_q, d))
        key_patterns = random.normal(keys[1], (n_k, d))
        value_patterns = random.normal(keys[2], (n_k, d_v))

        output = attention_as_hopfield(queries, key_patterns, value_patterns)
        assert_shape(output, (n_q, d_v), "attention as hopfield output")

    def test_matches_standard_attention(self, rng_key):
        """Should match standard attention with appropriate beta."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 6, 8

        queries = random.normal(keys[0], (n_q, d))
        key_patterns = random.normal(keys[1], (n_k, d))
        value_patterns = random.normal(keys[2], (n_k, d))

        # Hopfield attention
        output_hopfield = attention_as_hopfield(queries, key_patterns, value_patterns)

        # Standard scaled dot-product attention
        from attn_tensors.attention import scaled_dot_product_attention

        output_standard = scaled_dot_product_attention(queries, key_patterns, value_patterns)

        assert_allclose(output_hopfield, output_standard, err_msg="Should match standard attention")

    def test_custom_beta(self, rng_key):
        """Custom beta should change attention sharpness."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 3, 5, 8

        queries = random.normal(keys[0], (n_q, d))
        key_patterns = random.normal(keys[1], (n_k, d))
        value_patterns = random.normal(keys[2], (n_k, d))

        out_low = attention_as_hopfield(queries, key_patterns, value_patterns, beta=0.1)
        out_high = attention_as_hopfield(queries, key_patterns, value_patterns, beta=10.0)

        # Outputs should be different
        assert not jnp.allclose(out_low, out_high), "Different beta should give different outputs"


# =============================================================================
# Capacity Tests
# =============================================================================


class TestCapacity:
    """Tests for capacity functions."""

    def test_classical_capacity_linear(self):
        """Classical capacity should scale linearly with dimension."""
        N1 = 100
        N2 = 200

        cap1 = classical_capacity(N1)
        cap2 = classical_capacity(N2)

        # Should roughly double
        assert 1.8 < (cap2 / cap1) < 2.2

    def test_classical_capacity_formula(self):
        """Should follow M_max ≈ 0.14 * N."""
        N = 100
        expected = 0.14 * N
        actual = classical_capacity(N)

        assert_allclose(actual, expected)

    def test_modern_capacity_exponential(self):
        """Modern capacity should scale exponentially with dimension."""
        d1 = 10
        d2 = 20

        cap1 = float(modern_capacity(d1))
        cap2 = float(modern_capacity(d2))

        # Should be much larger (roughly exp(5) ≈ 148 times)
        assert cap2 / cap1 > 100


class TestSeparationQuality:
    """Tests for separation_quality function."""

    def test_returns_scalar(self, random_patterns):
        """Should return a scalar."""
        quality = separation_quality(random_patterns)

        assert jnp.isscalar(quality) or quality.shape == ()

    def test_orthogonal_patterns_high_quality(self):
        """Orthogonal patterns should have high separation."""
        patterns = jnp.eye(4)  # Perfectly orthogonal
        quality = separation_quality(patterns)

        assert quality > 1.0, f"Orthogonal patterns should have high separation, got {quality}"

    def test_identical_patterns_zero_quality(self):
        """Identical patterns should have zero separation."""
        pattern = jnp.array([[1.0, 0.0, 0.0]])
        patterns = jnp.repeat(pattern, 3, axis=0)
        quality = separation_quality(patterns)

        assert quality < 0.01, f"Identical patterns should have zero separation, got {quality}"


# =============================================================================
# Energy Landscape Tests
# =============================================================================


class TestEnergyLandscape1D:
    """Tests for energy_landscape_1d function."""

    def test_output_shapes(self, simple_patterns):
        """Should return x values and energies of same length."""
        x, energies = energy_landscape_1d(simple_patterns, axis=0, n_points=50)

        assert len(x) == 50
        assert len(energies) == 50

    def test_energies_finite(self, simple_patterns):
        """All energies should be finite."""
        x, energies = energy_landscape_1d(simple_patterns, axis=0, n_points=20)

        assert jnp.all(jnp.isfinite(energies)), "Energies should be finite"

    def test_different_axes(self, random_patterns):
        """Should work for different axes."""
        d = random_patterns.shape[1]

        for axis in range(min(d, 3)):
            x, energies = energy_landscape_1d(random_patterns, axis=axis, n_points=10)
            assert len(energies) == 10


class TestRetrievalBasin:
    """Tests for retrieval_basin function."""

    def test_returns_fraction(self, simple_patterns, rng_key):
        """Should return a fraction in [0, 1]."""
        success_rate = retrieval_basin(
            pattern_idx=0, patterns=simple_patterns, n_samples=10, noise_scale=0.1, key=rng_key
        )

        assert 0.0 <= success_rate <= 1.0

    def test_low_noise_high_success(self, simple_patterns, rng_key):
        """Low noise should give high success rate."""
        success_rate = retrieval_basin(
            pattern_idx=0,
            patterns=simple_patterns,
            n_samples=20,
            noise_scale=0.01,
            beta=10.0,
            key=rng_key,
        )

        assert success_rate > 0.8, f"Low noise should give high success, got {success_rate}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in Hopfield networks."""

    def test_single_pattern(self, rng_key):
        """Should work with single pattern."""
        pattern = random.normal(rng_key, (1, 8))
        query = pattern[0] + 0.1

        result, _ = hopfield_retrieve(query, pattern)
        assert_finite(result, "single pattern retrieval")

    def test_high_dimensional(self, rng_key):
        """Should work with high-dimensional patterns."""
        M, d = 10, 128
        patterns = random.normal(rng_key, (M, d))
        query = patterns[0] + random.normal(random.split(rng_key, 1)[0], (d,)) * 0.1

        result, _ = hopfield_retrieve(query, patterns, beta=1.0)
        assert_finite(result, "high-dimensional retrieval")

    def test_many_patterns(self, rng_key):
        """Should work with many patterns."""
        M, d = 50, 32
        patterns = random.normal(rng_key, (M, d))
        query = patterns[0]

        result, _ = hopfield_retrieve(query, patterns, beta=1.0, max_iterations=20)
        assert_finite(result, "many patterns retrieval")

    def test_extreme_beta(self, simple_patterns, rng_key):
        """Should handle extreme beta values."""
        query = random.normal(rng_key, (simple_patterns.shape[1],))

        # Very low beta
        result_low, _ = hopfield_retrieve(query, simple_patterns, beta=0.01)
        assert_finite(result_low, "low beta retrieval")

        # Very high beta
        result_high, _ = hopfield_retrieve(query, simple_patterns, beta=100.0)
        assert_finite(result_high, "high beta retrieval")
