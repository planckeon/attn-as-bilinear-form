"""Tests for gradients module - manual vs autodiff verification."""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from hypothesis import given, settings
import numpy as np

from attn_tensors.gradients import (
    grad_scores_wrt_Q,
    grad_scores_wrt_K,
    grad_softmax,
    grad_output_wrt_A,
    grad_output_wrt_V,
    attention_backward,
    verify_gradients,
    gradient_flow_analysis,
    gradient_numerical_check,
)
from attn_tensors.attention import scaled_dot_product_attention

from .helpers import (
    RTOL,
    ATOL,
    assert_allclose,
    assert_shape,
    assert_finite,
    qkv_tensors,
    small_floats,
)


# =============================================================================
# Individual Gradient Component Tests
# =============================================================================


class TestGradScoresWrtQ:
    """Tests for gradient of scores w.r.t. queries."""

    def test_basic_shape(self, sample_qkv):
        """Gradient should have same shape as Q."""
        Q, K = sample_qkv["Q"], sample_qkv["K"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        dL_dS = jnp.ones((n_q, n_k))
        scale = 1.0 / jnp.sqrt(d)

        dL_dQ = grad_scores_wrt_Q(dL_dS, K, scale)
        assert_shape(dL_dQ, (n_q, d), "dL_dQ")

    def test_formula_correctness(self, rng_key):
        """Test that dL/dQ = scale * dL/dS @ K."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 3, 5, 8

        dL_dS = random.normal(keys[0], (n_q, n_k))
        K = random.normal(keys[1], (n_k, d))
        scale = 0.5

        result = grad_scores_wrt_Q(dL_dS, K, scale)
        expected = scale * (dL_dS @ K)

        assert_allclose(result, expected, err_msg="dL_dQ formula incorrect")

    def test_against_jax_autodiff(self, rng_key):
        """Verify against JAX autodiff."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 4, 6, 8

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        scale = 1.0 / jnp.sqrt(d)

        # Manual gradient
        dL_dS = jnp.ones((n_q, n_k))  # Assume dL/dS = 1 for simplicity
        dL_dQ_manual = grad_scores_wrt_Q(dL_dS, K, scale)

        # JAX autodiff
        def score_fn(Q):
            return jnp.sum(scale * jnp.einsum("ia,ja->ij", Q, K))

        dL_dQ_jax = jax.grad(score_fn)(Q)

        assert_allclose(dL_dQ_manual, dL_dQ_jax, err_msg="Manual vs JAX mismatch for dL_dQ")


class TestGradScoresWrtK:
    """Tests for gradient of scores w.r.t. keys."""

    def test_basic_shape(self, sample_qkv):
        """Gradient should have same shape as K."""
        Q, K = sample_qkv["Q"], sample_qkv["K"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        dL_dS = jnp.ones((n_q, n_k))
        scale = 1.0 / jnp.sqrt(d)

        dL_dK = grad_scores_wrt_K(dL_dS, Q, scale)
        assert_shape(dL_dK, (n_k, d), "dL_dK")

    def test_formula_correctness(self, rng_key):
        """Test that dL/dK = scale * dL/dS^T @ Q."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 3, 5, 8

        dL_dS = random.normal(keys[0], (n_q, n_k))
        Q = random.normal(keys[1], (n_q, d))
        scale = 0.5

        result = grad_scores_wrt_K(dL_dS, Q, scale)
        expected = scale * (dL_dS.T @ Q)

        assert_allclose(result, expected, err_msg="dL_dK formula incorrect")

    def test_against_jax_autodiff(self, rng_key):
        """Verify against JAX autodiff."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 4, 6, 8

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        scale = 1.0 / jnp.sqrt(d)

        dL_dS = jnp.ones((n_q, n_k))
        dL_dK_manual = grad_scores_wrt_K(dL_dS, Q, scale)

        def score_fn(K):
            return jnp.sum(scale * jnp.einsum("ia,ja->ij", Q, K))

        dL_dK_jax = jax.grad(score_fn)(K)

        assert_allclose(dL_dK_manual, dL_dK_jax, err_msg="Manual vs JAX mismatch for dL_dK")


class TestGradSoftmax:
    """Tests for gradient through softmax."""

    def test_basic_shape(self, rng_key):
        """Gradient should have same shape as input."""
        n_q, n_k = 4, 6
        S = random.normal(rng_key, (n_q, n_k))
        A = jax.nn.softmax(S, axis=-1)
        dL_dA = jnp.ones((n_q, n_k))

        dL_dS = grad_softmax(dL_dA, A)
        assert_shape(dL_dS, (n_q, n_k), "dL_dS")

    def test_zero_gradient_when_uniform(self, rng_key):
        """When dL_dA is uniform per row and A is uniform, gradient should be zero."""
        n_q, n_k = 3, 5
        A = jnp.ones((n_q, n_k)) / n_k  # Uniform distribution
        dL_dA = jnp.ones((n_q, n_k))  # Uniform upstream gradient

        dL_dS = grad_softmax(dL_dA, A)

        # Each row: dL_dS = A * (dL_dA - sum(A * dL_dA)) = A * (1 - 1) = 0
        assert_allclose(
            dL_dS, jnp.zeros((n_q, n_k)), err_msg="Uniform case should give zero gradient"
        )

    def test_against_jax_autodiff(self, rng_key):
        """Verify against JAX autodiff."""
        n_q, n_k = 4, 6
        S = random.normal(rng_key, (n_q, n_k))
        A = jax.nn.softmax(S, axis=-1)

        # Use a simple loss function
        target = jax.nn.softmax(random.normal(random.split(rng_key, 1)[0], (n_q, n_k)), axis=-1)

        def loss_fn(S):
            A = jax.nn.softmax(S, axis=-1)
            return jnp.sum((A - target) ** 2)

        # JAX gradient
        dL_dS_jax = jax.grad(loss_fn)(S)

        # Manual gradient
        dL_dA = 2 * (A - target)
        dL_dS_manual = grad_softmax(dL_dA, A)

        assert_allclose(dL_dS_manual, dL_dS_jax, err_msg="Manual vs JAX mismatch for softmax grad")


class TestGradOutputWrtA:
    """Tests for gradient of output w.r.t. attention weights."""

    def test_basic_shape(self, sample_qkv):
        """Gradient should have shape (n_q, n_k)."""
        V = sample_qkv["V"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        dL_dO = jnp.ones((n_q, d))
        dL_dA = grad_output_wrt_A(dL_dO, V)

        assert_shape(dL_dA, (n_q, n_k), "dL_dA")

    def test_formula_correctness(self, rng_key):
        """Test that dL/dA = dL/dO @ V^T."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 3, 5, 8

        dL_dO = random.normal(keys[0], (n_q, d))
        V = random.normal(keys[1], (n_k, d))

        result = grad_output_wrt_A(dL_dO, V)
        expected = dL_dO @ V.T

        assert_allclose(result, expected, err_msg="dL_dA formula incorrect")


class TestGradOutputWrtV:
    """Tests for gradient of output w.r.t. values."""

    def test_basic_shape(self, sample_qkv):
        """Gradient should have same shape as V."""
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        dL_dO = jnp.ones((n_q, d))
        A = jnp.ones((n_q, n_k)) / n_k

        dL_dV = grad_output_wrt_V(dL_dO, A)
        assert_shape(dL_dV, (n_k, d), "dL_dV")

    def test_formula_correctness(self, rng_key):
        """Test that dL/dV = A^T @ dL/dO."""
        keys = random.split(rng_key, 2)
        n_q, n_k, d = 3, 5, 8

        dL_dO = random.normal(keys[0], (n_q, d))
        A = jax.nn.softmax(random.normal(keys[1], (n_q, n_k)), axis=-1)

        result = grad_output_wrt_V(dL_dO, A)
        expected = A.T @ dL_dO

        assert_allclose(result, expected, err_msg="dL_dV formula incorrect")


# =============================================================================
# Full Backward Pass Tests
# =============================================================================


class TestAttentionBackward:
    """Tests for full attention backward pass."""

    def test_output_shapes(self, sample_qkv):
        """All gradients should have correct shapes."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        # Forward pass to get intermediates
        scale = 1.0 / jnp.sqrt(d)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale
        A = jax.nn.softmax(S, axis=-1)
        O = jnp.einsum("ij,jb->ib", A, V)

        dL_dO = jnp.ones_like(O)
        dL_dQ, dL_dK, dL_dV = attention_backward(dL_dO, Q, K, V, A, S)

        assert_shape(dL_dQ, (n_q, d), "dL_dQ")
        assert_shape(dL_dK, (n_k, d), "dL_dK")
        assert_shape(dL_dV, (n_k, d), "dL_dV")

    def test_against_jax_autodiff(self, sample_qkv):
        """Verify full backward pass against JAX autodiff."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        # Forward pass
        scale = 1.0 / jnp.sqrt(d)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale
        A = jax.nn.softmax(S, axis=-1)
        O = jnp.einsum("ij,jb->ib", A, V)

        # Simple L2 loss
        def loss_fn(Q, K, V):
            out = scaled_dot_product_attention(Q, K, V)
            return 0.5 * jnp.sum(out**2)

        # JAX gradients
        dL_dQ_jax, dL_dK_jax, dL_dV_jax = jax.grad(loss_fn, argnums=(0, 1, 2))(Q, K, V)

        # Manual gradients
        dL_dO = O  # dL/dO = O for L = 0.5 * ||O||^2
        dL_dQ_manual, dL_dK_manual, dL_dV_manual = attention_backward(dL_dO, Q, K, V, A, S)

        assert_allclose(dL_dQ_manual, dL_dQ_jax, err_msg="dL_dQ mismatch")
        assert_allclose(dL_dK_manual, dL_dK_jax, err_msg="dL_dK mismatch")
        assert_allclose(dL_dV_manual, dL_dV_jax, err_msg="dL_dV mismatch")

    @given(qkv_tensors())
    @settings(deadline=None)
    def test_gradients_always_finite(self, qkv):
        """Gradients should always be finite for bounded inputs."""
        Q, K, V = qkv["Q"], qkv["K"], qkv["V"]
        d = qkv["d"]

        scale = 1.0 / jnp.sqrt(d)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale
        A = jax.nn.softmax(S, axis=-1)
        O = jnp.einsum("ij,jb->ib", A, V)

        dL_dO = O
        dL_dQ, dL_dK, dL_dV = attention_backward(dL_dO, Q, K, V, A, S)

        assert_finite(dL_dQ, "dL_dQ")
        assert_finite(dL_dK, "dL_dK")
        assert_finite(dL_dV, "dL_dV")


# =============================================================================
# Verification Function Tests
# =============================================================================


class TestVerifyGradients:
    """Tests for the verify_gradients function."""

    def test_returns_all_keys(self, sample_qkv):
        """Should return verification for all gradients."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        results = verify_gradients(Q, K, V)

        expected_keys = {"dL_dQ", "dL_dK", "dL_dV", "all_correct"}
        assert set(results.keys()) == expected_keys

    def test_all_gradients_correct(self, sample_qkv):
        """All manual gradients should match autodiff."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        results = verify_gradients(Q, K, V)

        assert results["dL_dQ"], "dL_dQ should be correct"
        assert results["dL_dK"], "dL_dK should be correct"
        assert results["dL_dV"], "dL_dV should be correct"
        assert results["all_correct"], "All gradients should be correct"

    @given(qkv_tensors())
    @settings(deadline=None)
    def test_always_passes(self, qkv):
        """Verification should pass for all valid inputs."""
        # Use relaxed tolerances for edge cases from hypothesis
        results = verify_gradients(qkv["Q"], qkv["K"], qkv["V"], rtol=1e-3, atol=1e-3)
        assert results["all_correct"], "Verification should pass"


# =============================================================================
# Gradient Flow Analysis Tests
# =============================================================================


class TestGradientFlowAnalysis:
    """Tests for gradient_flow_analysis function."""

    def test_returns_all_tensors(self, sample_qkv):
        """Should return all forward and backward tensors."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        result = gradient_flow_analysis(Q, K, V)

        # Check forward quantities
        assert "S" in result
        assert "A" in result
        assert "O" in result

        # Check gradients
        assert "dL_dO" in result
        assert "dL_dV" in result
        assert "dL_dA" in result
        assert "dL_dS" in result
        assert "dL_dQ" in result
        assert "dL_dK" in result

        # Check statistics
        assert "grad_norms" in result
        assert "dL_dQ" in result["grad_norms"]
        assert "dL_dK" in result["grad_norms"]
        assert "dL_dV" in result["grad_norms"]

    def test_gradient_norms_positive(self, sample_qkv):
        """Gradient norms should be non-negative."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        result = gradient_flow_analysis(Q, K, V)

        for name, norm in result["grad_norms"].items():
            assert norm >= 0, f"{name} norm should be non-negative"

    def test_forward_tensors_correct(self, sample_qkv):
        """Forward tensors should match attention computation."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        result = gradient_flow_analysis(Q, K, V)

        # Verify S
        scale = 1.0 / jnp.sqrt(d)
        expected_S = jnp.einsum("ia,ja->ij", Q, K) * scale
        assert_allclose(result["S"], expected_S, err_msg="S mismatch")

        # Verify A
        expected_A = jax.nn.softmax(expected_S, axis=-1)
        assert_allclose(result["A"], expected_A, err_msg="A mismatch")

        # Verify O
        expected_O = jnp.einsum("ij,jb->ib", expected_A, V)
        assert_allclose(result["O"], expected_O, err_msg="O mismatch")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in gradient computation."""

    def test_single_element(self):
        """Should handle single element tensors."""
        Q = jnp.array([[1.0, 2.0]])
        K = jnp.array([[3.0, 4.0]])
        V = jnp.array([[5.0, 6.0]])

        results = verify_gradients(Q, K, V)
        assert results["all_correct"]

    def test_large_values(self, rng_key):
        """Should handle large values (numerical stability)."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (3, 8)) * 10.0
        K = random.normal(keys[1], (5, 8)) * 10.0
        V = random.normal(keys[2], (5, 8)) * 10.0

        results = verify_gradients(Q, K, V, rtol=1e-3, atol=1e-3)
        assert results["all_correct"], "Should handle large values"

    def test_zero_values(self):
        """Should handle zero-valued tensors."""
        Q = jnp.zeros((3, 8))
        K = jnp.zeros((5, 8))
        V = jnp.ones((5, 8))  # Non-zero to avoid division issues

        result = gradient_flow_analysis(Q, K, V)
        assert_finite(result["dL_dQ"], "dL_dQ with zero Q/K")
        assert_finite(result["dL_dK"], "dL_dK with zero Q/K")
        assert_finite(result["dL_dV"], "dL_dV with zero Q/K")

    def test_gradient_symmetry(self, rng_key):
        """When Q = K, gradients should have specific relationships."""
        X = random.normal(rng_key, (5, 8))
        V = random.normal(random.split(rng_key, 1)[0], (5, 8))

        result = gradient_flow_analysis(X, X, V)

        # dL_dQ and dL_dK should both be computed correctly
        # (they won't be equal due to the asymmetric role in gradient formulas)
        assert_finite(result["dL_dQ"], "dL_dQ when Q=K")
        assert_finite(result["dL_dK"], "dL_dK when Q=K")


# =============================================================================
# Gradient Properties Tests
# =============================================================================


# =============================================================================
# Numerical Gradient Check Tests
# =============================================================================


class TestGradientNumericalCheck:
    """Tests for gradient_numerical_check function."""

    @pytest.mark.slow
    def test_returns_expected_keys(self, rng_key):
        """Should return dictionary with all expected keys."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (2, 4))
        K = random.normal(keys[1], (3, 4))
        V = random.normal(keys[2], (3, 4))

        result = gradient_numerical_check(Q, K, V)

        expected_keys = {"numerical", "analytic", "max_error", "rel_error"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.slow
    def test_numerical_matches_analytic(self, rng_key):
        """Numerical gradient should closely match analytic gradient."""
        keys = random.split(rng_key, 3)
        # Use small tensors for speed
        Q = random.normal(keys[0], (2, 3))
        K = random.normal(keys[1], (2, 3))
        V = random.normal(keys[2], (2, 3))

        result = gradient_numerical_check(Q, K, V)

        # Numerical gradients should be close to analytic
        # Max absolute error is more reliable than relative error for small values
        assert result["max_error"] < 0.05, f"Max error too large: {result['max_error']}"

        # Check that the gradients are correlated (same direction)
        numerical = result["numerical"].flatten()
        analytic = result["analytic"].flatten()
        correlation = jnp.corrcoef(numerical, analytic)[0, 1]
        assert correlation > 0.99, f"Gradients not well correlated: {correlation}"

    @pytest.mark.slow
    def test_shapes_match(self, rng_key):
        """Numerical and analytic gradients should have same shape."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 2, 3, 4
        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        V = random.normal(keys[2], (n_k, d))

        result = gradient_numerical_check(Q, K, V)

        assert result["numerical"].shape == Q.shape
        assert result["analytic"].shape == Q.shape

    @pytest.mark.slow
    def test_custom_epsilon(self, rng_key):
        """Different epsilon values should still give reasonable results."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (2, 3))
        K = random.normal(keys[1], (2, 3))
        V = random.normal(keys[2], (2, 3))

        result_small = gradient_numerical_check(Q, K, V, eps=1e-6)
        result_large = gradient_numerical_check(Q, K, V, eps=1e-4)

        # Both should give reasonable errors (relaxed for float32)
        assert result_small["max_error"] < 0.5, f"Small eps error: {result_small['max_error']}"
        assert result_large["max_error"] < 0.1, f"Large eps error: {result_large['max_error']}"

    @pytest.mark.slow
    def test_finite_gradients(self, rng_key):
        """All gradient values should be finite."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (2, 3))
        K = random.normal(keys[1], (2, 3))
        V = random.normal(keys[2], (2, 3))

        result = gradient_numerical_check(Q, K, V)

        assert_finite(result["numerical"], "numerical gradient")
        assert_finite(result["analytic"], "analytic gradient")


class TestGradientProperties:
    """Tests for mathematical properties of gradients."""

    def test_chain_rule_composition(self, sample_qkv):
        """Verify the chain rule composition through layers."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        scale = 1.0 / jnp.sqrt(d)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale
        A = jax.nn.softmax(S, axis=-1)
        O = jnp.einsum("ij,jb->ib", A, V)

        # Upstream gradient
        dL_dO = jnp.ones_like(O)

        # Step by step through backward pass
        dL_dV = grad_output_wrt_V(dL_dO, A)
        dL_dA = grad_output_wrt_A(dL_dO, V)
        dL_dS = grad_softmax(dL_dA, A)
        dL_dQ = grad_scores_wrt_Q(dL_dS, K, scale)
        dL_dK = grad_scores_wrt_K(dL_dS, Q, scale)

        # Verify against combined backward pass
        dL_dQ_full, dL_dK_full, dL_dV_full = attention_backward(dL_dO, Q, K, V, A, S)

        assert_allclose(dL_dQ, dL_dQ_full, err_msg="Chain rule Q")
        assert_allclose(dL_dK, dL_dK_full, err_msg="Chain rule K")
        assert_allclose(dL_dV, dL_dV_full, err_msg="Chain rule V")

    def test_linearity_in_upstream_gradient(self, sample_qkv):
        """Gradients should scale linearly with upstream gradient."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        scale = 1.0 / jnp.sqrt(d)
        S = jnp.einsum("ia,ja->ij", Q, K) * scale
        A = jax.nn.softmax(S, axis=-1)
        O = jnp.einsum("ij,jb->ib", A, V)

        # Two different upstream gradients
        dL_dO_1 = jnp.ones_like(O)
        dL_dO_2 = dL_dO_1 * 2.0

        dL_dQ_1, _, _ = attention_backward(dL_dO_1, Q, K, V, A, S)
        dL_dQ_2, _, _ = attention_backward(dL_dO_2, Q, K, V, A, S)

        # Should scale linearly
        assert_allclose(dL_dQ_2, dL_dQ_1 * 2.0, err_msg="Linearity in upstream gradient")
