"""Tests for softmax.py - Softmax, entropy, and statistical mechanics."""

import jax.numpy as jnp
import jax.random as random
import pytest
from hypothesis import assume, given

from attn_tensors.softmax import (
    attention_entropy,
    effective_number_of_states,
    entropy,
    free_energy,
    gibbs_distribution,
    log_softmax,
    max_entropy,
    normalized_entropy,
    partition_function,
    softmax_jacobian,
    softmax_jacobian_batched,
    softmax_rows,
    temperature_sweep,
)

from .helpers import (
    ATOL,
    assert_allclose,
    assert_finite,
    assert_nonnegative,
    assert_probability_distribution,
    assert_shape,
    positive_floats,
    softmax_input,
)

# =============================================================================
# Softmax Basic Tests
# =============================================================================


class TestSoftmaxRows:
    """Tests for softmax_rows function."""

    def test_sums_to_one_1d(self, rng_key):
        """1D softmax should sum to 1."""
        x = random.normal(rng_key, (5,))
        probs = softmax_rows(x)
        assert_allclose(jnp.sum(probs), 1.0, err_msg="Softmax doesn't sum to 1")

    def test_sums_to_one_2d(self, rng_key):
        """2D softmax should have rows summing to 1."""
        x = random.normal(rng_key, (3, 5))
        probs = softmax_rows(x)
        row_sums = jnp.sum(probs, axis=-1)
        assert_allclose(row_sums, jnp.ones(3), err_msg="Rows don't sum to 1")

    def test_nonnegative(self, rng_key):
        """Softmax output should be non-negative."""
        x = random.normal(rng_key, (10,))
        probs = softmax_rows(x)
        assert_nonnegative(probs, "softmax")

    def test_equal_inputs_uniform(self):
        """Equal inputs should give uniform distribution."""
        x = jnp.ones(5) * 3.0
        probs = softmax_rows(x)
        expected = jnp.ones(5) / 5
        assert_allclose(probs, expected)

    def test_shape_preserved(self, rng_key):
        """Output shape should match input shape."""
        x = random.normal(rng_key, (3, 4, 5))
        probs = softmax_rows(x)
        assert_shape(probs, (3, 4, 5))

    @given(softmax_input())
    def test_fuzz_sums_to_one(self, x):
        """Fuzz: softmax always sums to 1."""
        probs = softmax_rows(x)
        assert_allclose(jnp.sum(probs), 1.0, rtol=1e-3, atol=1e-4)

    @given(softmax_input())
    def test_fuzz_nonnegative(self, x):
        """Fuzz: softmax is always non-negative."""
        probs = softmax_rows(x)
        assert jnp.all(probs >= 0)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of softmax."""

    def test_large_positive_inputs(self):
        """Large positive inputs shouldn't overflow."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        probs = softmax_rows(x)

        assert_finite(probs, "softmax(large positive)")
        assert_probability_distribution(probs)

    def test_large_negative_inputs(self):
        """Large negative inputs shouldn't underflow to all zeros."""
        x = jnp.array([-1000.0, -1001.0, -1002.0])
        probs = softmax_rows(x)

        assert_finite(probs, "softmax(large negative)")
        assert_probability_distribution(probs)

    def test_mixed_extreme_inputs(self):
        """Mixed extreme inputs should be handled."""
        x = jnp.array([-1000.0, 0.0, 1000.0])
        probs = softmax_rows(x)

        assert_finite(probs, "softmax(mixed extreme)")
        # The largest should dominate
        assert probs[2] > 0.99

    def test_very_close_inputs(self):
        """Very close inputs should give near-uniform output."""
        x = jnp.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10])
        probs = softmax_rows(x)

        assert_finite(probs, "softmax(very close)")
        # Should be nearly uniform
        assert jnp.max(probs) - jnp.min(probs) < 0.01

    @pytest.mark.parametrize("magnitude", [100, 500, 1000])
    def test_stability_at_magnitudes(self, magnitude):
        """Test stability at various magnitudes."""
        x = jnp.array([0.0, 1.0, 2.0]) + magnitude
        probs = softmax_rows(x)

        assert_finite(probs, f"softmax at magnitude {magnitude}")
        assert_probability_distribution(probs)


# =============================================================================
# Log Softmax Tests
# =============================================================================


class TestLogSoftmax:
    """Tests for log_softmax function."""

    def test_consistency_with_softmax(self, rng_key):
        """log_softmax should equal log(softmax)."""
        x = random.normal(rng_key, (5,))

        log_probs = log_softmax(x)
        expected = jnp.log(softmax_rows(x))

        assert_allclose(log_probs, expected)

    def test_logsumexp_normalization(self, rng_key):
        """exp(log_softmax(x)) should sum to 1."""
        x = random.normal(rng_key, (5,))
        log_probs = log_softmax(x)
        probs = jnp.exp(log_probs)

        assert_allclose(jnp.sum(probs), 1.0)

    def test_large_inputs_stable(self):
        """Log softmax should be stable for large inputs."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        log_probs = log_softmax(x)

        assert_finite(log_probs, "log_softmax(large)")
        # All log probs should be negative
        assert jnp.all(log_probs <= 0)


# =============================================================================
# Gibbs Distribution Tests
# =============================================================================


class TestGibbsDistribution:
    """Tests for gibbs_distribution (softmax with temperature)."""

    def test_temperature_one_equals_softmax(self, rng_key):
        """Temperature=1 should equal standard softmax."""
        x = random.normal(rng_key, (5,))

        gibbs = gibbs_distribution(x, temperature=1.0)
        softmax = softmax_rows(x)

        assert_allclose(gibbs, softmax)

    def test_low_temperature_peaks(self, rng_key):
        """Low temperature should make distribution more peaked."""
        x = jnp.array([1.0, 2.0, 3.0])

        high_T = gibbs_distribution(x, temperature=10.0)
        low_T = gibbs_distribution(x, temperature=0.1)

        # Low T should have higher max
        assert jnp.max(low_T) > jnp.max(high_T)

    def test_high_temperature_uniform(self):
        """High temperature should approach uniform."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        probs = gibbs_distribution(x, temperature=100.0)

        # Should be nearly uniform
        assert jnp.max(probs) - jnp.min(probs) < 0.1

    def test_temperature_zero_limit(self):
        """Very low temperature should approach one-hot at argmax."""
        x = jnp.array([1.0, 5.0, 2.0])
        probs = gibbs_distribution(x, temperature=0.01)

        # Should be nearly one-hot at index 1
        assert probs[1] > 0.99

    def test_with_mask(self):
        """Masked positions should get zero probability."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.array([True, True, False, True])

        probs = gibbs_distribution(x, mask=mask)

        assert probs[2] < 1e-6, "Masked position should have ~0 probability"
        # Remaining should sum to 1
        assert_allclose(probs[0] + probs[1] + probs[3], 1.0, atol=1e-4)

    @given(softmax_input(), positive_floats)
    def test_fuzz_temperature(self, x, temperature):
        """Fuzz: any positive temperature should give valid distribution."""
        assume(len(x) >= 2)
        probs = gibbs_distribution(x, temperature=temperature)

        assert_finite(probs, "gibbs")
        assert_nonnegative(probs, "gibbs")
        assert_allclose(jnp.sum(probs), 1.0, rtol=1e-3, atol=1e-4)


# =============================================================================
# Partition Function Tests
# =============================================================================


class TestPartitionFunction:
    """Tests for partition_function (Z = sum exp(E/T))."""

    def test_positive(self, rng_key):
        """Partition function should always be positive."""
        energies = random.normal(rng_key, (5,))
        Z = partition_function(energies)
        assert Z > 0, "Partition function should be positive"

    def test_uniform_energies(self):
        """Uniform energies: Z = n * exp(E)."""
        E = 2.0
        n = 5
        energies = jnp.ones(n) * E
        Z = partition_function(energies, temperature=1.0)
        expected = n * jnp.exp(E)
        assert_allclose(Z, expected)

    @given(softmax_input())
    def test_fuzz_positive(self, energies):
        """Fuzz: Z > 0 always."""
        Z = partition_function(energies)
        assert Z > 0


# =============================================================================
# Entropy Tests
# =============================================================================


class TestEntropy:
    """Tests for entropy function."""

    def test_uniform_is_max(self):
        """Uniform distribution should have maximum entropy."""
        n = 8
        probs = jnp.ones(n) / n
        H = entropy(probs)
        H_max = max_entropy(n)
        assert_allclose(H, H_max)

    def test_one_hot_is_zero(self):
        """One-hot distribution should have zero entropy."""
        probs = jnp.array([0.0, 1.0, 0.0, 0.0])
        H = entropy(probs)
        assert_allclose(H, 0.0, atol=1e-6)

    def test_nonnegative(self, rng_key):
        """Entropy should be non-negative."""
        x = random.normal(rng_key, (10,))
        probs = softmax_rows(x)
        H = entropy(probs)
        assert H >= -ATOL, f"Entropy is negative: {H}"

    def test_bounded_by_log_n(self, rng_key):
        """Entropy should be bounded by log(n)."""
        n = 10
        x = random.normal(rng_key, (n,))
        probs = softmax_rows(x)
        H = entropy(probs)
        H_max = max_entropy(n)
        assert H <= H_max + ATOL, f"Entropy {H} exceeds max {H_max}"

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_max_entropy_values(self, n):
        """max_entropy should return log(n)."""
        H_max = max_entropy(n)
        expected = jnp.log(n)
        assert_allclose(H_max, expected)


# =============================================================================
# Normalized Entropy Tests
# =============================================================================


class TestNormalizedEntropy:
    """Tests for normalized_entropy function."""

    def test_range_zero_to_one(self, rng_key):
        """Normalized entropy should be in [0, 1]."""
        x = random.normal(rng_key, (10,))
        probs = softmax_rows(x)
        H_norm = normalized_entropy(probs)

        assert H_norm >= -ATOL, f"Normalized entropy < 0: {H_norm}"
        assert H_norm <= 1.0 + ATOL, f"Normalized entropy > 1: {H_norm}"

    def test_uniform_is_one(self):
        """Uniform distribution should have normalized entropy = 1."""
        probs = jnp.ones(10) / 10
        H_norm = normalized_entropy(probs)
        assert_allclose(H_norm, 1.0)

    def test_peaked_is_near_zero(self):
        """Peaked distribution should have normalized entropy near 0."""
        probs = jnp.array([0.99, 0.005, 0.005])
        H_norm = normalized_entropy(probs)
        assert H_norm < 0.1


# =============================================================================
# Attention Entropy Tests
# =============================================================================


class TestAttentionEntropy:
    """Tests for attention_entropy function."""

    def test_shape(self, rng_key):
        """Should return entropy per query position."""
        x = random.normal(rng_key, (4, 6))
        A = softmax_rows(x)
        H = attention_entropy(A)
        assert_shape(H, (4,), "attention_entropy")

    def test_bounded(self, rng_key):
        """Each entropy should be bounded by log(n_keys)."""
        n_q, n_k = 4, 6
        x = random.normal(rng_key, (n_q, n_k))
        A = softmax_rows(x)
        H = attention_entropy(A)

        H_max = max_entropy(n_k)
        assert jnp.all(H <= H_max + ATOL)
        assert jnp.all(H >= -ATOL)


# =============================================================================
# Temperature Sweep Tests
# =============================================================================


class TestTemperatureSweep:
    """Tests for temperature_sweep function."""

    def test_output_shape(self, rng_key):
        """Output should have shape (n_temps, n_energies)."""
        energies = random.normal(rng_key, (5,))
        temperatures = jnp.array([0.1, 1.0, 10.0])

        probs = temperature_sweep(energies, temperatures)
        assert_shape(probs, (3, 5), "temperature_sweep")

    def test_each_row_valid_distribution(self, rng_key):
        """Each temperature should give valid distribution."""
        energies = random.normal(rng_key, (5,))
        temperatures = jnp.array([0.1, 0.5, 1.0, 2.0, 10.0])

        probs = temperature_sweep(energies, temperatures)

        for i in range(len(temperatures)):
            assert_probability_distribution(probs[i], axis=-1)


# =============================================================================
# Effective Number of States Tests
# =============================================================================


class TestEffectiveNumberOfStates:
    """Tests for effective_number_of_states (perplexity)."""

    def test_uniform_equals_n(self):
        """Uniform distribution should have n effective states."""
        n = 10
        probs = jnp.ones(n) / n
        eff_n = effective_number_of_states(probs)
        assert_allclose(eff_n, n)

    def test_one_hot_equals_one(self):
        """One-hot should have ~1 effective state."""
        probs = jnp.array([0.0, 1.0, 0.0, 0.0])
        eff_n = effective_number_of_states(probs)
        assert_allclose(eff_n, 1.0, atol=1e-4)

    def test_bounded_by_n(self, rng_key):
        """Effective states should be <= n."""
        n = 10
        x = random.normal(rng_key, (n,))
        probs = softmax_rows(x)
        eff_n = effective_number_of_states(probs)

        assert eff_n <= n + ATOL
        assert eff_n >= 1.0 - ATOL


# =============================================================================
# Softmax Jacobian Tests
# =============================================================================


class TestSoftmaxJacobian:
    """Tests for softmax_jacobian function."""

    def test_shape(self, rng_key):
        """Jacobian should be (n, n)."""
        x = random.normal(rng_key, (5,))
        probs = softmax_rows(x)
        J = softmax_jacobian(probs)
        assert_shape(J, (5, 5), "softmax_jacobian")

    def test_row_sums_zero(self, rng_key):
        """Rows of Jacobian should sum to zero."""
        x = random.normal(rng_key, (5,))
        probs = softmax_rows(x)
        J = softmax_jacobian(probs)

        row_sums = jnp.sum(J, axis=-1)
        assert_allclose(row_sums, jnp.zeros(5), err_msg="Jacobian rows don't sum to 0")

    def test_diagonal_formula(self, rng_key):
        """Diagonal: J[i,i] = p[i] * (1 - p[i])."""
        x = random.normal(rng_key, (4,))
        probs = softmax_rows(x)
        J = softmax_jacobian(probs)

        expected_diag = probs * (1 - probs)
        actual_diag = jnp.diag(J)
        assert_allclose(actual_diag, expected_diag)

    def test_off_diagonal_formula(self, rng_key):
        """Off-diagonal: J[i,j] = -p[i] * p[j]."""
        x = random.normal(rng_key, (3,))
        probs = softmax_rows(x)
        J = softmax_jacobian(probs)

        # Check a specific off-diagonal
        assert_allclose(J[0, 1], -probs[0] * probs[1])
        assert_allclose(J[1, 2], -probs[1] * probs[2])

    @given(softmax_input(n=4))
    def test_fuzz_row_sums_zero(self, x):
        """Fuzz: Jacobian rows always sum to zero."""
        probs = softmax_rows(x)
        J = softmax_jacobian(probs)
        row_sums = jnp.sum(J, axis=-1)
        assert_allclose(row_sums, jnp.zeros(4), rtol=1e-3, atol=1e-4)


class TestSoftmaxJacobianBatched:
    """Tests for softmax_jacobian_batched function."""

    def test_shape(self, rng_key):
        """Batched Jacobian should be (batch, n, n)."""
        x = random.normal(rng_key, (3, 5))
        probs = softmax_rows(x)
        J = softmax_jacobian_batched(probs)
        assert_shape(J, (3, 5, 5), "softmax_jacobian_batched")

    def test_matches_unbatched(self, rng_key):
        """Batched should match applying unbatched to each row."""
        x = random.normal(rng_key, (3, 4))
        probs = softmax_rows(x)

        J_batched = softmax_jacobian_batched(probs)

        for i in range(3):
            J_single = softmax_jacobian(probs[i])
            assert_allclose(J_batched[i], J_single)


# =============================================================================
# Free Energy Tests
# =============================================================================


class TestFreeEnergy:
    """Tests for free_energy function."""

    def test_formula(self, rng_key):
        """F = -T * log(Z)."""
        energies = random.normal(rng_key, (5,))
        T = 2.0

        F = free_energy(energies, temperature=T)
        Z = partition_function(energies, temperature=T)
        expected = -T * jnp.log(Z)

        assert_allclose(F, expected)

    def test_finite(self, rng_key):
        """Free energy should be finite."""
        energies = random.normal(rng_key, (10,))
        F = free_energy(energies, temperature=1.0)
        assert_finite(jnp.array([F]), "free_energy")


# =============================================================================
# Edge Cases
# =============================================================================


class TestSoftmaxEdgeCases:
    """Edge case tests for softmax module."""

    def test_length_one(self):
        """Single element should give probability 1."""
        x = jnp.array([5.0])
        probs = softmax_rows(x)
        assert_allclose(probs, jnp.array([1.0]))

    def test_length_two(self):
        """Two elements with known values."""
        x = jnp.array([0.0, 1.0])
        probs = softmax_rows(x)

        # exp(0) / (exp(0) + exp(1)) = 1 / (1 + e)
        expected_0 = 1.0 / (1.0 + jnp.exp(1.0))
        expected_1 = jnp.exp(1.0) / (1.0 + jnp.exp(1.0))
        assert_allclose(probs, jnp.array([expected_0, expected_1]))

    def test_all_zeros(self):
        """All zeros should give uniform."""
        x = jnp.zeros(5)
        probs = softmax_rows(x)
        assert_allclose(probs, jnp.ones(5) / 5)

    def test_3d_input(self, rng_key):
        """Should work with 3D input."""
        x = random.normal(rng_key, (2, 3, 4))
        probs = softmax_rows(x)

        assert_shape(probs, (2, 3, 4))
        # Each innermost vector should sum to 1
        sums = jnp.sum(probs, axis=-1)
        assert_allclose(sums, jnp.ones((2, 3)))
