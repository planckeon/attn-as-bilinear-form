"""Tests for bilinear.py - Metric tensors and bilinear forms."""

import jax.numpy as jnp
import jax.random as random
import pytest
from hypothesis import given
from hypothesis import strategies as st

from attn_tensors.bilinear import (
    bilinear_form,
    bilinear_form_batch,
    diagonal_metric,
    euclidean_metric,
    inner_product,
    learned_metric,
    lower_index,
    metric_inverse,
    quadratic_form,
    raise_index,
    scaled_euclidean_metric,
    verify_metric_properties,
)

from .helpers import (
    ATOL,
    assert_allclose,
    assert_positive_definite,
    assert_positive_semidefinite,
    assert_shape,
    assert_symmetric,
    valid_metric,
)

# =============================================================================
# Euclidean Metric Tests
# =============================================================================


class TestEuclideanMetric:
    """Tests for euclidean_metric function."""

    @pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
    def test_is_identity(self, d):
        """Euclidean metric should be identity matrix."""
        g = euclidean_metric(d)
        assert_allclose(g, jnp.eye(d))

    @pytest.mark.parametrize("d", [1, 2, 4, 8])
    def test_shape(self, d):
        """Euclidean metric should have shape (d, d)."""
        g = euclidean_metric(d)
        assert_shape(g, (d, d), "euclidean_metric")

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_is_symmetric(self, d):
        """Euclidean metric should be symmetric."""
        g = euclidean_metric(d)
        assert_symmetric(g, "euclidean_metric")

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_is_positive_definite(self, d):
        """Euclidean metric should be positive definite."""
        g = euclidean_metric(d)
        assert_positive_definite(g, "euclidean_metric")


# =============================================================================
# Scaled Euclidean Metric Tests
# =============================================================================


class TestScaledEuclideanMetric:
    """Tests for scaled_euclidean_metric function."""

    @pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
    def test_default_scaling(self, d):
        """Default scaling should be 1/sqrt(d)."""
        g = scaled_euclidean_metric(d)
        expected = jnp.eye(d) / jnp.sqrt(d)
        assert_allclose(g, expected)

    @pytest.mark.parametrize("d,scale", [(4, 0.5), (8, 2.0), (16, 0.25)])
    def test_custom_scaling(self, d, scale):
        """Custom scaling should be applied correctly."""
        g = scaled_euclidean_metric(d, scale=scale)
        expected = jnp.eye(d) * scale
        assert_allclose(g, expected)

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_is_symmetric(self, d):
        """Scaled metric should be symmetric."""
        g = scaled_euclidean_metric(d)
        assert_symmetric(g, "scaled_euclidean_metric")

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_is_positive_definite(self, d):
        """Scaled metric should be positive definite."""
        g = scaled_euclidean_metric(d)
        assert_positive_definite(g, "scaled_euclidean_metric")


# =============================================================================
# Learned Metric Tests
# =============================================================================


class TestLearnedMetric:
    """Tests for learned_metric function (g = W^T W)."""

    def test_from_identity(self):
        """Identity W should give identity metric."""
        W = jnp.eye(4)
        g = learned_metric(W)
        assert_allclose(g, jnp.eye(4))

    def test_from_random_is_positive_semidefinite(self, rng_key):
        """Random W should give positive semi-definite metric."""
        W = random.normal(rng_key, (5, 4))
        g = learned_metric(W)
        assert_positive_semidefinite(g, "learned_metric")

    def test_is_symmetric(self, rng_key):
        """Learned metric should be symmetric."""
        W = random.normal(rng_key, (3, 4))
        g = learned_metric(W)
        assert_symmetric(g, "learned_metric")

    def test_shape(self, rng_key):
        """Learned metric should have shape (d, d) where d = W.shape[1]."""
        W = random.normal(rng_key, (5, 8))
        g = learned_metric(W)
        assert_shape(g, (8, 8), "learned_metric")

    @given(st.integers(1, 8), st.integers(1, 8))
    def test_fuzz_shapes(self, m, n):
        """Fuzz test: various W shapes should work."""
        key = random.PRNGKey(42)
        W = random.normal(key, (m, n))
        g = learned_metric(W)
        assert_shape(g, (n, n), "learned_metric")
        assert_symmetric(g, "learned_metric")


# =============================================================================
# Diagonal Metric Tests
# =============================================================================


class TestDiagonalMetric:
    """Tests for diagonal_metric function."""

    def test_from_ones(self):
        """Ones vector should give identity metric."""
        diag = jnp.ones(4)
        g = diagonal_metric(diag)
        assert_allclose(g, jnp.eye(4))

    def test_from_custom(self):
        """Custom diagonal should be on diagonal."""
        diag = jnp.array([1.0, 2.0, 3.0])
        g = diagonal_metric(diag)
        expected = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        assert_allclose(g, expected)

    def test_is_symmetric(self, rng_key):
        """Diagonal metric should be symmetric."""
        diag = jnp.abs(random.normal(rng_key, (5,))) + 0.1
        g = diagonal_metric(diag)
        assert_symmetric(g, "diagonal_metric")

    def test_positive_diag_is_positive_definite(self, rng_key):
        """Positive diagonal should give positive definite metric."""
        diag = jnp.abs(random.normal(rng_key, (5,))) + 0.1
        g = diagonal_metric(diag)
        assert_positive_definite(g, "diagonal_metric")


# =============================================================================
# Index Raising/Lowering Tests
# =============================================================================


class TestIndexOperations:
    """Tests for raise_index and lower_index functions."""

    def test_lower_then_raise_identity(self, rng_key):
        """lower then raise should return original vector."""
        d = 4
        v = random.normal(rng_key, (d,))
        g = scaled_euclidean_metric(d)
        g_inv = metric_inverse(g)

        v_lower = lower_index(v, g)
        v_back = raise_index(v_lower, g_inv)
        assert_allclose(v_back, v, err_msg="raise(lower(v)) != v")

    def test_raise_then_lower_identity(self, rng_key):
        """raise then lower should return original vector."""
        d = 4
        v = random.normal(rng_key, (d,))
        g = scaled_euclidean_metric(d)
        g_inv = metric_inverse(g)

        v_upper = raise_index(v, g_inv)
        v_back = lower_index(v_upper, g)
        assert_allclose(v_back, v, err_msg="lower(raise(v)) != v")

    def test_lower_index_formula(self, rng_key):
        """lower_index should compute v_a = g_{ab} v^b."""
        d = 3
        v = random.normal(rng_key, (d,))
        g = jnp.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])

        v_lower = lower_index(v, g)
        expected = jnp.einsum("ab,b->a", g, v)
        assert_allclose(v_lower, expected)

    @given(valid_metric(d=4))
    def test_fuzz_lower_raise_inverse(self, g):
        """Fuzz: lower then raise should be identity for any valid metric."""
        key = random.PRNGKey(123)
        v = random.normal(key, (4,))
        g_inv = metric_inverse(g)

        v_lower = lower_index(v, g)
        v_back = raise_index(v_lower, g_inv)
        assert_allclose(v_back, v, rtol=1e-3, atol=1e-4)


# =============================================================================
# Metric Inverse Tests
# =============================================================================


class TestMetricInverse:
    """Tests for metric_inverse function."""

    def test_inverse_of_identity(self):
        """Inverse of identity should be identity."""
        g = jnp.eye(4)
        g_inv = metric_inverse(g)
        assert_allclose(g_inv, jnp.eye(4))

    def test_inverse_of_scaled(self):
        """Inverse of scaled identity should have reciprocal scale."""
        scale = 2.0
        g = jnp.eye(4) * scale
        g_inv = metric_inverse(g)
        assert_allclose(g_inv, jnp.eye(4) / scale)

    def test_g_times_g_inv_is_identity(self, rng_key):
        """g @ g^{-1} should be identity."""
        W = random.normal(rng_key, (4, 4))
        g = W.T @ W + jnp.eye(4) * 0.1  # Ensure positive definite
        g_inv = metric_inverse(g)

        product = g @ g_inv
        assert_allclose(product, jnp.eye(4), err_msg="g @ g_inv != I")


# =============================================================================
# Bilinear Form Tests
# =============================================================================


class TestBilinearForm:
    """Tests for bilinear_form function."""

    def test_euclidean_is_dot_product(self, rng_key):
        """Bilinear form with Euclidean metric should be dot product."""
        keys = random.split(rng_key, 2)
        u = random.normal(keys[0], (4,))
        v = random.normal(keys[1], (4,))
        g = euclidean_metric(4)

        result = bilinear_form(u, v, g)
        expected = jnp.dot(u, v)
        assert_allclose(result, expected)

    def test_symmetry(self, rng_key):
        """B(u, v) should equal B(v, u) for symmetric metric."""
        keys = random.split(rng_key, 2)
        u = random.normal(keys[0], (4,))
        v = random.normal(keys[1], (4,))
        g = scaled_euclidean_metric(4)

        buv = bilinear_form(u, v, g)
        bvu = bilinear_form(v, u, g)
        assert_allclose(buv, bvu, err_msg="B(u,v) != B(v,u)")

    def test_linearity_first_arg(self, rng_key):
        """B(au + bv, w) = aB(u, w) + bB(v, w)."""
        keys = random.split(rng_key, 3)
        u = random.normal(keys[0], (4,))
        v = random.normal(keys[1], (4,))
        w = random.normal(keys[2], (4,))
        g = scaled_euclidean_metric(4)
        a, b = 2.0, 3.0

        lhs = bilinear_form(a * u + b * v, w, g)
        rhs = a * bilinear_form(u, w, g) + b * bilinear_form(v, w, g)
        assert_allclose(lhs, rhs, err_msg="Bilinear form not linear in first arg")

    def test_linearity_second_arg(self, rng_key):
        """B(u, av + bw) = aB(u, v) + bB(u, w)."""
        keys = random.split(rng_key, 3)
        u = random.normal(keys[0], (4,))
        v = random.normal(keys[1], (4,))
        w = random.normal(keys[2], (4,))
        g = scaled_euclidean_metric(4)
        a, b = 2.0, 3.0

        lhs = bilinear_form(u, a * v + b * w, g)
        rhs = a * bilinear_form(u, v, g) + b * bilinear_form(u, w, g)
        assert_allclose(lhs, rhs, err_msg="Bilinear form not linear in second arg")

    @given(valid_metric(d=4))
    def test_fuzz_symmetry(self, g):
        """Fuzz: B(u, v) = B(v, u) for any valid metric."""
        key = random.PRNGKey(42)
        keys = random.split(key, 2)
        u = random.normal(keys[0], (4,))
        v = random.normal(keys[1], (4,))

        buv = bilinear_form(u, v, g)
        bvu = bilinear_form(v, u, g)
        assert_allclose(buv, bvu, rtol=1e-3, atol=1e-4)


# =============================================================================
# Batch Bilinear Form Tests
# =============================================================================


class TestBilinearFormBatch:
    """Tests for bilinear_form_batch function."""

    def test_shape(self, rng_key):
        """Output shape should be (n_q, n_k)."""
        keys = random.split(rng_key, 2)
        Q = random.normal(keys[0], (3, 4))
        K = random.normal(keys[1], (5, 4))
        g = scaled_euclidean_metric(4)

        S = bilinear_form_batch(Q, K, g)
        assert_shape(S, (3, 5), "bilinear_form_batch")

    def test_equals_loop(self, rng_key):
        """Batch should equal looping over pairs."""
        keys = random.split(rng_key, 2)
        Q = random.normal(keys[0], (3, 4))
        K = random.normal(keys[1], (5, 4))
        g = scaled_euclidean_metric(4)

        S_batch = bilinear_form_batch(Q, K, g)

        # Manual loop
        S_loop = jnp.zeros((3, 5))
        for i in range(3):
            for j in range(5):
                S_loop = S_loop.at[i, j].set(bilinear_form(Q[i], K[j], g))

        assert_allclose(S_batch, S_loop, err_msg="Batch != loop")

    def test_with_identity_metric(self, rng_key):
        """With identity metric, should equal Q @ K^T."""
        keys = random.split(rng_key, 2)
        Q = random.normal(keys[0], (3, 4))
        K = random.normal(keys[1], (5, 4))
        g = euclidean_metric(4)

        S = bilinear_form_batch(Q, K, g)
        expected = Q @ K.T
        assert_allclose(S, expected)


# =============================================================================
# Quadratic Form Tests
# =============================================================================


class TestQuadraticForm:
    """Tests for quadratic_form function."""

    def test_nonnegative(self, rng_key):
        """Quadratic form should be non-negative for positive definite metric."""
        v = random.normal(rng_key, (4,))
        g = scaled_euclidean_metric(4)

        q = quadratic_form(v, g)
        assert q >= -ATOL, f"Quadratic form is negative: {q}"

    def test_zero_vector(self):
        """Quadratic form of zero vector should be zero."""
        v = jnp.zeros(4)
        g = scaled_euclidean_metric(4)

        q = quadratic_form(v, g)
        assert_allclose(q, 0.0)

    def test_equals_bilinear_form(self, rng_key):
        """Q(v) should equal B(v, v)."""
        v = random.normal(rng_key, (4,))
        g = scaled_euclidean_metric(4)

        q = quadratic_form(v, g)
        b = bilinear_form(v, v, g)
        assert_allclose(q, b)


# =============================================================================
# Verify Metric Properties Tests
# =============================================================================


class TestVerifyMetricProperties:
    """Tests for verify_metric_properties function."""

    def test_valid_metric_passes(self, rng_key):
        """Valid metric should pass verification."""
        W = random.normal(rng_key, (4, 4))
        g = W.T @ W + jnp.eye(4) * 0.1

        result = verify_metric_properties(g)
        assert result["valid_metric"], f"Valid metric failed verification: {result}"
        assert result["symmetric"]
        assert result["positive_definite"]

    def test_asymmetric_fails(self):
        """Asymmetric matrix should fail verification."""
        g = jnp.array([[1.0, 0.5], [0.3, 1.0]])  # Not symmetric

        result = verify_metric_properties(g)
        assert not result["valid_metric"], "Asymmetric matrix should fail"
        assert not result["symmetric"]

    def test_negative_eigenvalue_fails(self):
        """Matrix with negative eigenvalue should fail."""
        g = jnp.array([[1.0, 0.0], [0.0, -1.0]])  # Negative eigenvalue

        result = verify_metric_properties(g)
        assert not result["valid_metric"], "Negative definite matrix should fail"
        assert not result["positive_definite"]
        assert result["min_eigenvalue"] < 0


# =============================================================================
# Inner Product Tests
# =============================================================================


class TestInnerProduct:
    """Tests for inner_product convenience function."""

    def test_euclidean_mode(self, rng_key):
        """Euclidean inner product should equal dot product."""
        keys = random.split(rng_key, 2)
        u = random.normal(keys[0], (8,))
        v = random.normal(keys[1], (8,))

        result = inner_product(u, v, metric_type="euclidean")
        expected = jnp.dot(u, v)
        assert_allclose(result, expected)

    def test_scaled_mode(self, rng_key):
        """Scaled inner product should equal dot product / sqrt(d)."""
        keys = random.split(rng_key, 2)
        d = 8
        u = random.normal(keys[0], (d,))
        v = random.normal(keys[1], (d,))

        result = inner_product(u, v, metric_type="scaled")
        expected = jnp.dot(u, v) / jnp.sqrt(d)
        assert_allclose(result, expected)

    def test_default_is_scaled(self, rng_key):
        """Default metric_type should be 'scaled'."""
        keys = random.split(rng_key, 2)
        u = random.normal(keys[0], (8,))
        v = random.normal(keys[1], (8,))

        result_default = inner_product(u, v)
        result_scaled = inner_product(u, v, metric_type="scaled")
        assert_allclose(result_default, result_scaled)

    def test_symmetry(self, rng_key):
        """Inner product should be symmetric."""
        keys = random.split(rng_key, 2)
        u = random.normal(keys[0], (8,))
        v = random.normal(keys[1], (8,))

        result_uv = inner_product(u, v)
        result_vu = inner_product(v, u)
        assert_allclose(result_uv, result_vu)

    def test_bilinearity(self, rng_key):
        """Inner product should be bilinear."""
        keys = random.split(rng_key, 3)
        u = random.normal(keys[0], (8,))
        v = random.normal(keys[1], (8,))
        w = random.normal(keys[2], (8,))
        a, b = 2.5, 1.5

        # Linearity in first argument
        lhs = inner_product(a * u + b * v, w)
        rhs = a * inner_product(u, w) + b * inner_product(v, w)
        assert_allclose(lhs, rhs, err_msg="Not linear in first arg")

    def test_different_dimensions(self):
        """Should work for various dimensions."""
        for d in [1, 2, 4, 8, 16, 32]:
            u = jnp.ones(d)
            v = jnp.ones(d)

            result_euc = inner_product(u, v, metric_type="euclidean")
            result_scaled = inner_product(u, v, metric_type="scaled")

            assert_allclose(result_euc, float(d))
            assert_allclose(result_scaled, float(d) / jnp.sqrt(d))


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for bilinear module."""

    def test_dimension_one(self):
        """1x1 metrics should work."""
        g = euclidean_metric(1)
        assert_shape(g, (1, 1))
        assert_allclose(g, jnp.array([[1.0]]))

    def test_very_small_values(self):
        """Very small values shouldn't cause issues."""
        u = jnp.array([1e-10, 1e-10])
        v = jnp.array([1e-10, 1e-10])
        g = euclidean_metric(2)

        result = bilinear_form(u, v, g)
        assert jnp.isfinite(result)

    def test_very_large_values(self):
        """Very large values shouldn't overflow."""
        u = jnp.array([1e5, 1e5])
        v = jnp.array([1e5, 1e5])
        g = euclidean_metric(2)

        result = bilinear_form(u, v, g)
        assert jnp.isfinite(result)
