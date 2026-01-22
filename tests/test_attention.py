"""Tests for attention module."""

import jax
import jax.numpy as jnp
import jax.random as random
from hypothesis import given, settings

from attn_tensors.attention import (
    attention_output,
    attention_scores,
    attention_scores_with_metric,
    attention_weights,
    batched_attention,
    bilinear_attention,
    decompose_attention,
    scaled_dot_product_attention,
)
from attn_tensors.bilinear import scaled_euclidean_metric

from .helpers import (
    assert_allclose,
    assert_finite,
    assert_probability_distribution,
    assert_shape,
    qkv_tensors,
)

# =============================================================================
# Attention Scores Tests
# =============================================================================


class TestAttentionScores:
    """Tests for attention_scores function."""

    def test_basic_shape(self, sample_qkv):
        """Test output shape is (n_q, n_k)."""
        Q, K = sample_qkv["Q"], sample_qkv["K"]
        n_q, n_k = sample_qkv["n_q"], sample_qkv["n_k"]

        S = attention_scores(Q, K)
        assert_shape(S, (n_q, n_k), "attention scores")

    def test_unscaled_is_dot_product(self, rng_key):
        """Unscaled scores should be Q @ K^T."""
        keys = random.split(rng_key, 2)
        Q = random.normal(keys[0], (3, 4))
        K = random.normal(keys[1], (5, 4))

        S = attention_scores(Q, K, scale=False)
        expected = Q @ K.T

        assert_allclose(S, expected, err_msg="Unscaled scores should be Q @ K^T")

    def test_scaling_factor(self, rng_key):
        """Scaled scores should be Q @ K^T / sqrt(d_k)."""
        keys = random.split(rng_key, 2)
        d_k = 16
        Q = random.normal(keys[0], (3, d_k))
        K = random.normal(keys[1], (5, d_k))

        S_scaled = attention_scores(Q, K, scale=True)
        S_unscaled = attention_scores(Q, K, scale=False)

        assert_allclose(
            S_scaled,
            S_unscaled / jnp.sqrt(d_k),
            err_msg="Scaling should divide by sqrt(d_k)",
        )

    def test_zero_scores_for_orthogonal(self, rng_key):
        """Orthogonal Q and K should give zero scores."""
        # Create orthogonal vectors
        Q = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        K = jnp.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        S = attention_scores(Q, K, scale=False)
        assert_allclose(S, jnp.zeros((1, 2)), err_msg="Orthogonal vectors should give zero scores")

    def test_identical_qk_gives_positive_diagonal(self, rng_key):
        """When Q == K, diagonal scores should be positive (for non-zero vectors)."""
        Q = random.normal(rng_key, (4, 8))
        K = Q  # Same as queries

        S = attention_scores(Q, K, scale=False)
        diagonal = jnp.diag(S)

        # Diagonal should be ||q_i||^2 which is positive
        assert jnp.all(diagonal > 0), "Self-similarity should be positive"

    @given(qkv_tensors())
    @settings(deadline=None)
    def test_finite_outputs(self, qkv):
        """Scores should always be finite for bounded inputs."""
        S = attention_scores(qkv["Q"], qkv["K"])
        assert_finite(S, "attention scores")


class TestAttentionScoresWithMetric:
    """Tests for attention_scores_with_metric function."""

    def test_basic_shape(self, sample_qkv, rng_key):
        """Test output shape is (n_q, n_k)."""
        Q, K = sample_qkv["Q"], sample_qkv["K"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        g = scaled_euclidean_metric(d)
        S = attention_scores_with_metric(Q, K, g)
        assert_shape(S, (n_q, n_k), "attention scores with metric")

    def test_identity_metric_matches_unscaled(self, rng_key):
        """Identity metric should give same result as unscaled dot product."""
        keys = random.split(rng_key, 2)
        d = 8
        Q = random.normal(keys[0], (3, d))
        K = random.normal(keys[1], (5, d))

        g = jnp.eye(d)
        S_metric = attention_scores_with_metric(Q, K, g)
        S_unscaled = attention_scores(Q, K, scale=False)

        assert_allclose(S_metric, S_unscaled, err_msg="Identity metric should match unscaled")

    def test_scaled_metric_matches_scaled(self, rng_key):
        """Scaled Euclidean metric should match scaled attention scores."""
        keys = random.split(rng_key, 2)
        d = 16
        Q = random.normal(keys[0], (3, d))
        K = random.normal(keys[1], (5, d))

        g = scaled_euclidean_metric(d)
        S_metric = attention_scores_with_metric(Q, K, g)
        S_scaled = attention_scores(Q, K, scale=True)

        assert_allclose(S_metric, S_scaled, err_msg="Scaled metric should match scaled scores")


# =============================================================================
# Attention Weights Tests
# =============================================================================


class TestAttentionWeights:
    """Tests for attention_weights function."""

    def test_weights_are_probabilities(self, sample_qkv):
        """Weights should form valid probability distributions."""
        Q, K = sample_qkv["Q"], sample_qkv["K"]
        S = attention_scores(Q, K)
        A = attention_weights(S)

        assert_probability_distribution(A, axis=-1, name="attention weights")

    def test_temperature_effect(self, rng_key):
        """Higher temperature should give more uniform weights."""
        S = random.normal(rng_key, (4, 6))

        A_low_temp = attention_weights(S, temperature=0.1)
        A_high_temp = attention_weights(S, temperature=10.0)

        # Entropy of high temp should be higher (more uniform)
        entropy_low = -jnp.sum(A_low_temp * jnp.log(A_low_temp + 1e-10), axis=-1).mean()
        entropy_high = -jnp.sum(A_high_temp * jnp.log(A_high_temp + 1e-10), axis=-1).mean()

        assert entropy_high > entropy_low, "Higher temperature should increase entropy"

    def test_mask_zeros_out(self, rng_key):
        """Masked positions should get zero weight."""
        S = random.normal(rng_key, (3, 5))
        mask = jnp.array(
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )

        A = attention_weights(S, mask=mask)

        # Masked positions should be ~0
        masked_values = A[~mask]
        assert jnp.allclose(masked_values, 0.0, atol=1e-6), "Masked positions should be zero"

        # Each row should still sum to 1
        assert_allclose(jnp.sum(A, axis=-1), jnp.ones(3), err_msg="Rows should sum to 1")


# =============================================================================
# Attention Output Tests
# =============================================================================


class TestAttentionOutput:
    """Tests for attention_output function."""

    def test_basic_shape(self, sample_qkv):
        """Output shape should be (n_q, d_v)."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        n_q, d = sample_qkv["n_q"], sample_qkv["d"]

        S = attention_scores(Q, K)
        A = attention_weights(S)
        O = attention_output(A, V)

        assert_shape(O, (n_q, d), "attention output")

    def test_uniform_weights_give_mean(self, rng_key):
        """Uniform attention weights should give mean of values."""
        V = random.normal(rng_key, (5, 8))
        n_q, n_k = 3, 5

        # Uniform weights
        A = jnp.ones((n_q, n_k)) / n_k

        O = attention_output(A, V)

        expected = jnp.mean(V, axis=0, keepdims=True)
        expected = jnp.repeat(expected, n_q, axis=0)

        assert_allclose(O, expected, err_msg="Uniform weights should give mean value")

    def test_one_hot_weights_select_value(self, rng_key):
        """One-hot attention weights should select corresponding value."""
        V = random.normal(rng_key, (5, 8))

        # One-hot weights
        A = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        O = attention_output(A, V)

        assert_allclose(O[0], V[0], err_msg="One-hot should select value 0")
        assert_allclose(O[1], V[2], err_msg="One-hot should select value 2")
        assert_allclose(O[2], V[4], err_msg="One-hot should select value 4")


# =============================================================================
# Full Attention Tests
# =============================================================================


class TestScaledDotProductAttention:
    """Tests for scaled_dot_product_attention function."""

    def test_basic_shape(self, sample_qkv):
        """Output should have shape (n_q, d_v)."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        n_q, d = sample_qkv["n_q"], sample_qkv["d"]

        O = scaled_dot_product_attention(Q, K, V)
        assert_shape(O, (n_q, d), "attention output")

    def test_return_weights(self, sample_qkv):
        """Should optionally return attention weights."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        O, A = scaled_dot_product_attention(Q, K, V, return_weights=True)

        assert isinstance(O, jax.Array), "Output should be array"
        assert isinstance(A, jax.Array), "Weights should be array"
        assert_probability_distribution(A, axis=-1, name="returned weights")

    def test_consistent_with_decomposed(self, sample_qkv):
        """Full attention should match decomposed version."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        O_full = scaled_dot_product_attention(Q, K, V)
        decomposed = decompose_attention(Q, K, V)

        assert_allclose(O_full, decomposed["output"], err_msg="Full should match decomposed")

    def test_mask_effect(self, rng_key):
        """Masking should prevent attending to masked positions."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (2, 4))
        K = random.normal(keys[1], (4, 4))
        V = random.normal(keys[2], (4, 4))

        # Mask: first query can only see first 2 keys
        mask = jnp.array(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )

        O_masked, A_masked = scaled_dot_product_attention(Q, K, V, mask=mask, return_weights=True)

        # First row weights should only be on first two keys
        assert jnp.allclose(A_masked[0, 2:], 0.0, atol=1e-6), "Masked positions should be zero"

    @given(qkv_tensors())
    @settings(deadline=None)
    def test_always_finite(self, qkv):
        """Output should always be finite for bounded inputs."""
        O = scaled_dot_product_attention(qkv["Q"], qkv["K"], qkv["V"])
        assert_finite(O, "attention output")


class TestBilinearAttention:
    """Tests for bilinear_attention function."""

    def test_identity_metric_matches_unscaled(self, rng_key):
        """Identity metric should behave like unscaled attention."""
        keys = random.split(rng_key, 3)
        d = 8
        Q = random.normal(keys[0], (3, d))
        K = random.normal(keys[1], (5, d))
        V = random.normal(keys[2], (5, d))

        g = jnp.eye(d)
        O_bilinear = bilinear_attention(Q, K, V, g)

        # Manual computation with unscaled scores
        S = Q @ K.T
        A = jax.nn.softmax(S, axis=-1)
        O_manual = A @ V

        assert_allclose(O_bilinear, O_manual, err_msg="Identity metric should match manual")

    def test_scaled_metric_matches_standard(self, sample_qkv):
        """Scaled Euclidean metric should match standard scaled attention."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        g = scaled_euclidean_metric(d)
        O_bilinear = bilinear_attention(Q, K, V, g)
        O_standard = scaled_dot_product_attention(Q, K, V)

        assert_allclose(O_bilinear, O_standard, err_msg="Scaled metric should match standard")


# =============================================================================
# Batched Attention Tests
# =============================================================================


class TestBatchedAttention:
    """Tests for batched_attention function."""

    def test_basic_shape(self, rng_key):
        """Output should have shape (batch, n_q, d_v)."""
        keys = random.split(rng_key, 3)
        batch, n_q, n_k, d = 4, 3, 5, 8

        Q = random.normal(keys[0], (batch, n_q, d))
        K = random.normal(keys[1], (batch, n_k, d))
        V = random.normal(keys[2], (batch, n_k, d))

        O = batched_attention(Q, K, V)
        assert_shape(O, (batch, n_q, d), "batched attention output")

    def test_matches_unbatched(self, rng_key):
        """Each batch element should match unbatched attention."""
        keys = random.split(rng_key, 3)
        batch, n_q, n_k, d = 3, 4, 6, 8

        Q = random.normal(keys[0], (batch, n_q, d))
        K = random.normal(keys[1], (batch, n_k, d))
        V = random.normal(keys[2], (batch, n_k, d))

        O_batched = batched_attention(Q, K, V)

        for b in range(batch):
            O_single = scaled_dot_product_attention(Q[b], K[b], V[b])
            assert_allclose(
                O_batched[b], O_single, err_msg=f"Batch element {b} should match unbatched"
            )

    def test_mask_broadcast(self, rng_key):
        """Non-batched mask should broadcast across batch."""
        keys = random.split(rng_key, 3)
        batch, n_q, n_k, d = 2, 3, 4, 8

        Q = random.normal(keys[0], (batch, n_q, d))
        K = random.normal(keys[1], (batch, n_k, d))
        V = random.normal(keys[2], (batch, n_k, d))

        # Shared causal mask
        mask = jnp.tril(jnp.ones((n_q, n_k), dtype=bool))

        O = batched_attention(Q, K, V, mask=mask)
        assert_finite(O, "batched output with mask")


# =============================================================================
# Decompose Attention Tests
# =============================================================================


class TestDecomposeAttention:
    """Tests for decompose_attention function."""

    def test_all_keys_present(self, sample_qkv):
        """Decomposed result should have all expected keys."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        result = decompose_attention(Q, K, V)

        expected_keys = {"scores_raw", "scores", "weights", "output", "partition", "scale"}
        assert set(result.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_shapes_correct(self, sample_qkv):
        """All decomposed tensors should have correct shapes."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        n_q, n_k, d = sample_qkv["n_q"], sample_qkv["n_k"], sample_qkv["d"]

        result = decompose_attention(Q, K, V)

        assert_shape(result["scores_raw"], (n_q, n_k), "scores_raw")
        assert_shape(result["scores"], (n_q, n_k), "scores")
        assert_shape(result["weights"], (n_q, n_k), "weights")
        assert_shape(result["output"], (n_q, d), "output")
        assert_shape(result["partition"], (n_q,), "partition")

    def test_scaling_relationship(self, sample_qkv):
        """Scaled scores should be raw_scores * scale."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]
        d = sample_qkv["d"]

        result = decompose_attention(Q, K, V)

        expected_scale = 1.0 / jnp.sqrt(d)
        assert_allclose(result["scale"], expected_scale, err_msg="Scale should be 1/sqrt(d)")

        expected_scores = result["scores_raw"] * result["scale"]
        assert_allclose(result["scores"], expected_scores, err_msg="Scores should be raw * scale")

    def test_weights_from_scores(self, sample_qkv):
        """Weights should be softmax of scores."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        result = decompose_attention(Q, K, V)

        expected_weights = jax.nn.softmax(result["scores"], axis=-1)
        assert_allclose(
            result["weights"], expected_weights, err_msg="Weights should be softmax(scores)"
        )

    def test_partition_function(self, sample_qkv):
        """Partition function should be sum of exp(scores)."""
        Q, K, V = sample_qkv["Q"], sample_qkv["K"], sample_qkv["V"]

        result = decompose_attention(Q, K, V)

        expected_Z = jnp.sum(jnp.exp(result["scores"]), axis=-1)
        assert_allclose(
            result["partition"], expected_Z, err_msg="Partition should be sum(exp(scores))"
        )


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_single_key(self, rng_key):
        """Should work with single key."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (4, 8))
        K = random.normal(keys[1], (1, 8))
        V = random.normal(keys[2], (1, 8))

        O = scaled_dot_product_attention(Q, K, V)
        assert_shape(O, (4, 8), "output with single key")

        # All queries attend to the same (only) key, so output should be V
        assert_allclose(O, jnp.repeat(V, 4, axis=0), err_msg="Single key should broadcast")

    def test_single_query(self, rng_key):
        """Should work with single query."""
        keys = random.split(rng_key, 3)
        Q = random.normal(keys[0], (1, 8))
        K = random.normal(keys[1], (4, 8))
        V = random.normal(keys[2], (4, 8))

        O = scaled_dot_product_attention(Q, K, V)
        assert_shape(O, (1, 8), "output with single query")
        assert_finite(O, "single query output")

    def test_large_dimension(self, rng_key):
        """Should be stable with large dimensions."""
        keys = random.split(rng_key, 3)
        d = 512

        Q = random.normal(keys[0], (2, d))
        K = random.normal(keys[1], (4, d))
        V = random.normal(keys[2], (4, d))

        O = scaled_dot_product_attention(Q, K, V)
        assert_finite(O, "output with large dimension")

    def test_equal_qkv(self, rng_key):
        """Should work when Q = K = V (self-attention)."""
        X = random.normal(rng_key, (5, 16))

        O = scaled_dot_product_attention(X, X, X)
        assert_shape(O, (5, 16), "self-attention output")
        assert_finite(O, "self-attention output")

    def test_identical_keys(self, rng_key):
        """Should handle identical keys (uniform attention)."""
        keys = random.split(rng_key, 2)
        Q = random.normal(keys[0], (3, 8))
        single_key = random.normal(keys[1], (1, 8))
        K = jnp.repeat(single_key, 4, axis=0)  # 4 identical keys
        V = random.normal(random.split(rng_key, 1)[0], (4, 8))

        O, A = scaled_dot_product_attention(Q, K, V, return_weights=True)

        # Attention should be uniform since all keys are identical
        expected_A = jnp.ones((3, 4)) / 4
        assert_allclose(A, expected_A, err_msg="Identical keys should give uniform attention")

    def test_zero_inputs(self):
        """Should handle zero inputs."""
        Q = jnp.zeros((2, 4))
        K = jnp.zeros((3, 4))
        V = jnp.ones((3, 4))  # Non-zero values

        O = scaled_dot_product_attention(Q, K, V)
        assert_finite(O, "zero input output")

        # With zero scores, softmax gives uniform weights
        expected = jnp.mean(V, axis=0, keepdims=True)
        expected = jnp.repeat(expected, 2, axis=0)
        assert_allclose(O, expected, err_msg="Zero Q/K should give mean of V")
