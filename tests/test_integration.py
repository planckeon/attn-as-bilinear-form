"""Integration tests - end-to-end tests for the full attention pipeline."""

import jax
import jax.numpy as jnp
import jax.random as random

from attn_tensors.attention import (
    batched_attention,
    bilinear_attention,
    decompose_attention,
    scaled_dot_product_attention,
)
from attn_tensors.bilinear import scaled_euclidean_metric
from attn_tensors.gradients import gradient_flow_analysis, verify_gradients
from attn_tensors.hopfield import (
    attention_as_hopfield,
    hopfield_retrieve,
)
from attn_tensors.masking import (
    causal_mask,
    causal_padding_mask,
    padding_mask,
)
from attn_tensors.multihead import (
    init_multihead_weights,
    multihead_attention,
    multihead_attention_batched,
)
from attn_tensors.softmax import entropy

from .helpers import (
    assert_allclose,
    assert_finite,
    assert_shape,
)

# =============================================================================
# Full Attention Pipeline Tests
# =============================================================================


class TestFullAttentionPipeline:
    """End-to-end tests for attention computation."""

    def test_self_attention_pipeline(self, rng_key):
        """Test complete self-attention: input -> output."""
        n, d = 8, 16
        X = random.normal(rng_key, (n, d))

        # Self-attention
        O = scaled_dot_product_attention(X, X, X)

        assert_shape(O, (n, d), "self-attention output")
        assert_finite(O, "self-attention output")

        # Output should be different from input (non-trivial transformation)
        assert not jnp.allclose(O, X), "Output should differ from input"

    def test_cross_attention_pipeline(self, rng_key):
        """Test complete cross-attention: queries attend to key-value memory."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 8, 16

        Q = random.normal(keys[0], (n_q, d))  # Query sequence
        K = random.normal(keys[1], (n_k, d))  # Memory keys
        V = random.normal(keys[2], (n_k, d))  # Memory values

        O = scaled_dot_product_attention(Q, K, V)

        assert_shape(O, (n_q, d), "cross-attention output")
        assert_finite(O, "cross-attention output")

    def test_causal_attention_pipeline(self, rng_key):
        """Test causal self-attention for autoregressive models."""
        n, d = 8, 16
        X = random.normal(rng_key, (n, d))
        mask = causal_mask(n)

        O, A = scaled_dot_product_attention(X, X, X, mask=mask, return_weights=True)

        # Check causality: position i should not attend to j > i
        for i in range(n):
            for j in range(i + 1, n):
                assert jnp.isclose(A[i, j], 0.0, atol=1e-6), f"Position ({i}, {j}) should be masked"

    def test_attention_with_temperature(self, rng_key):
        """Test attention with varying temperatures."""
        n, d = 6, 8
        X = random.normal(rng_key, (n, d))

        # Low temperature (sharp attention)
        _, A_sharp = scaled_dot_product_attention(X, X, X, temperature=0.1, return_weights=True)

        # High temperature (uniform attention)
        _, A_smooth = scaled_dot_product_attention(X, X, X, temperature=10.0, return_weights=True)

        # Sharp attention should have lower entropy
        H_sharp = jnp.mean(entropy(A_sharp))
        H_smooth = jnp.mean(entropy(A_smooth))

        assert H_sharp < H_smooth, "Lower temperature should give lower entropy"


class TestMultiheadPipeline:
    """End-to-end tests for multi-head attention."""

    def test_transformer_self_attention(self, rng_key):
        """Test multi-head self-attention as used in transformers."""
        keys = random.split(rng_key, 2)
        n, d_model = 8, 32
        num_heads = 4

        # Initialize
        weights = init_multihead_weights(keys[0], d_model, num_heads)
        X = random.normal(keys[1], (n, d_model))

        # Forward pass
        O = multihead_attention(
            X, X, X, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
        )

        assert_shape(O, (n, d_model), "multihead self-attention output")
        assert_finite(O, "multihead self-attention output")

    def test_transformer_causal_attention(self, rng_key):
        """Test causal multi-head attention for decoder."""
        keys = random.split(rng_key, 2)
        n, d_model = 8, 32
        num_heads = 4

        weights = init_multihead_weights(keys[0], d_model, num_heads)
        X = random.normal(keys[1], (n, d_model))
        mask = causal_mask(n)

        O, A = multihead_attention(
            X,
            X,
            X,
            weights["W_Q"],
            weights["W_K"],
            weights["W_V"],
            weights["W_O"],
            mask=mask,
            return_weights=True,
        )

        # Check causality for each head
        for h in range(num_heads):
            for i in range(n):
                for j in range(i + 1, n):
                    assert jnp.isclose(A[h, i, j], 0.0, atol=1e-6)

    def test_batched_inference(self, rng_key):
        """Test batched multi-head attention for efficient inference."""
        keys = random.split(rng_key, 2)
        batch, n, d_model = 4, 8, 32
        num_heads = 4

        weights = init_multihead_weights(keys[0], d_model, num_heads)
        X = random.normal(keys[1], (batch, n, d_model))

        O = multihead_attention_batched(
            X, X, X, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
        )

        assert_shape(O, (batch, n, d_model), "batched multihead output")
        assert_finite(O, "batched multihead output")

    def test_encoder_decoder_cross_attention(self, rng_key):
        """Test cross-attention between encoder and decoder."""
        keys = random.split(rng_key, 3)
        n_dec, n_enc, d_model = 6, 10, 32
        num_heads = 4

        weights = init_multihead_weights(keys[0], d_model, num_heads)
        decoder_states = random.normal(keys[1], (n_dec, d_model))  # Queries from decoder
        encoder_states = random.normal(keys[2], (n_enc, d_model))  # Keys/values from encoder

        O = multihead_attention(
            decoder_states,
            encoder_states,
            encoder_states,
            weights["W_Q"],
            weights["W_K"],
            weights["W_V"],
            weights["W_O"],
        )

        assert_shape(O, (n_dec, d_model), "cross-attention output")


class TestGradientPipeline:
    """End-to-end tests for gradient computation."""

    def test_gradient_verification_passes(self, rng_key):
        """Manual gradients should match autodiff."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 6, 8

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        V = random.normal(keys[2], (n_k, d))

        results = verify_gradients(Q, K, V)
        assert results["all_correct"], "All gradients should match"

    def test_gradient_flow_analysis_complete(self, rng_key):
        """Gradient flow analysis should return all expected tensors."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 6, 8

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        V = random.normal(keys[2], (n_k, d))

        result = gradient_flow_analysis(Q, K, V)

        # Check all forward tensors
        assert "S" in result and result["S"].shape == (n_q, n_k)
        assert "A" in result and result["A"].shape == (n_q, n_k)
        assert "O" in result and result["O"].shape == (n_q, d)

        # Check all gradients
        assert "dL_dQ" in result and result["dL_dQ"].shape == (n_q, d)
        assert "dL_dK" in result and result["dL_dK"].shape == (n_k, d)
        assert "dL_dV" in result and result["dL_dV"].shape == (n_k, d)

    def test_gradient_through_multihead(self, rng_key):
        """Test that gradients flow through multi-head attention."""
        keys = random.split(rng_key, 2)
        n, d_model = 4, 16
        num_heads = 2

        weights = init_multihead_weights(keys[0], d_model, num_heads)
        X = random.normal(keys[1], (n, d_model))

        def loss_fn(W_Q, W_K, W_V, W_O, X):
            O = multihead_attention(X, X, X, W_Q, W_K, W_V, W_O)
            return jnp.sum(O**2)

        # Should be able to compute gradients without error
        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(
            weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"], X
        )

        for i, g in enumerate(grads):
            assert_finite(g, f"gradient {i}")


class TestHopfieldIntegration:
    """Integration tests connecting Hopfield networks and attention."""

    def test_hopfield_attention_equivalence(self, rng_key):
        """Hopfield retrieval and attention should give same results."""
        keys = random.split(rng_key, 3)
        n_q, n_patterns, d = 4, 8, 16

        queries = random.normal(keys[0], (n_q, d))
        patterns = random.normal(keys[1], (n_patterns, d))
        values = random.normal(keys[2], (n_patterns, d))

        # Attention as Hopfield (uses default beta = 1/sqrt(d))
        out_hopfield = attention_as_hopfield(queries, patterns, values)

        # Standard attention
        out_attention = scaled_dot_product_attention(queries, patterns, values)

        assert_allclose(out_hopfield, out_attention, err_msg="Hopfield and attention should match")

    def test_memory_retrieval_cycle(self, rng_key):
        """Store patterns and retrieve them via Hopfield dynamics."""
        keys = random.split(rng_key, 2)
        n_patterns, d = 5, 32

        # Create well-separated patterns
        patterns = random.normal(keys[0], (n_patterns, d))
        patterns = patterns / jnp.linalg.norm(patterns, axis=-1, keepdims=True)  # Normalize

        # Query with small noise
        noise = random.normal(keys[1], (d,)) * 0.1
        query = patterns[0] + noise

        # Retrieve
        retrieved, iterations = hopfield_retrieve(query, patterns, beta=5.0)

        # Should be closest to pattern 0
        distances = jnp.sum((retrieved - patterns) ** 2, axis=-1)
        closest = jnp.argmin(distances)

        assert closest == 0, f"Should retrieve pattern 0, got {closest}"


class TestBilinearIntegration:
    """Integration tests for bilinear form view of attention."""

    def test_metric_attention_equivalence(self, rng_key):
        """Scaled Euclidean metric should give same result as standard attention."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 6, 16

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        V = random.normal(keys[2], (n_k, d))

        # Standard attention
        out_standard = scaled_dot_product_attention(Q, K, V)

        # Bilinear attention with scaled Euclidean metric
        g = scaled_euclidean_metric(d)
        out_bilinear = bilinear_attention(Q, K, V, g)

        assert_allclose(out_standard, out_bilinear, err_msg="Metric should match standard")

    def test_decomposition_consistency(self, rng_key):
        """Decomposed attention should be consistent with full computation."""
        keys = random.split(rng_key, 3)
        n_q, n_k, d = 4, 6, 8

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        V = random.normal(keys[2], (n_k, d))

        # Full computation
        out_full = scaled_dot_product_attention(Q, K, V)

        # Decomposed
        decomposed = decompose_attention(Q, K, V)

        assert_allclose(out_full, decomposed["output"], err_msg="Decomposed should match full")

        # Check intermediate values are consistent
        expected_weights = jax.nn.softmax(decomposed["scores"], axis=-1)
        assert_allclose(
            decomposed["weights"], expected_weights, err_msg="Weights should be softmax of scores"
        )


class TestVariableLengthSequences:
    """Integration tests for variable-length sequence handling."""

    def test_padding_mask_attention(self, rng_key):
        """Attention with padding masks should work correctly."""
        keys = random.split(rng_key, 2)
        batch, max_len, d = 2, 8, 16

        # Different lengths for each sequence
        lengths = jnp.array([6, 4])

        X = random.normal(keys[0], (batch, max_len, d))

        # Create padding mask
        pad_mask = padding_mask(lengths, max_len)

        # For batched attention, we need 2D attention mask
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :]

        # Apply batched attention with mask
        O = batched_attention(X, X, X, mask=attn_mask)

        assert_shape(O, (batch, max_len, d), "padded attention output")
        assert_finite(O, "padded attention output")

    def test_causal_with_padding(self, rng_key):
        """Combined causal and padding masks should work."""
        keys = random.split(rng_key, 2)
        batch, max_len, d = 2, 6, 8

        lengths = jnp.array([4, 6])
        X = random.normal(keys[0], (batch, max_len, d))

        # Combined mask
        mask = causal_padding_mask(lengths, max_len)

        O = batched_attention(X, X, X, mask=mask)

        assert_shape(O, (batch, max_len, d), "causal+padding attention output")
        assert_finite(O, "causal+padding attention output")


class TestNumericalStability:
    """Integration tests for numerical stability."""

    def test_large_sequences(self, rng_key):
        """Should handle large sequences without numerical issues."""
        keys = random.split(rng_key, 3)
        n, d = 128, 64

        Q = random.normal(keys[0], (n, d))
        K = random.normal(keys[1], (n, d))
        V = random.normal(keys[2], (n, d))

        O = scaled_dot_product_attention(Q, K, V)

        assert_finite(O, "large sequence output")

    def test_large_values(self, rng_key):
        """Should handle large input values."""
        keys = random.split(rng_key, 3)
        n, d = 8, 16

        # Large values
        Q = random.normal(keys[0], (n, d)) * 10.0
        K = random.normal(keys[1], (n, d)) * 10.0
        V = random.normal(keys[2], (n, d)) * 10.0

        O = scaled_dot_product_attention(Q, K, V)

        assert_finite(O, "large value output")

    def test_small_values(self, rng_key):
        """Should handle small input values."""
        keys = random.split(rng_key, 3)
        n, d = 8, 16

        # Small values
        Q = random.normal(keys[0], (n, d)) * 0.01
        K = random.normal(keys[1], (n, d)) * 0.01
        V = random.normal(keys[2], (n, d)) * 0.01

        O = scaled_dot_product_attention(Q, K, V)

        assert_finite(O, "small value output")

    def test_high_dimensional(self, rng_key):
        """Should handle high-dimensional features."""
        keys = random.split(rng_key, 3)
        n, d = 4, 512

        Q = random.normal(keys[0], (n, d))
        K = random.normal(keys[1], (n, d))
        V = random.normal(keys[2], (n, d))

        O = scaled_dot_product_attention(Q, K, V)

        assert_finite(O, "high-dimensional output")


class TestJITCompilation:
    """Integration tests for JAX JIT compilation."""

    def test_attention_jit(self, rng_key):
        """Attention should compile with JIT."""
        keys = random.split(rng_key, 3)
        n, d = 8, 16

        Q = random.normal(keys[0], (n, d))
        K = random.normal(keys[1], (n, d))
        V = random.normal(keys[2], (n, d))

        @jax.jit
        def jitted_attention(Q, K, V):
            return scaled_dot_product_attention(Q, K, V)

        O = jitted_attention(Q, K, V)
        assert_finite(O, "JIT attention output")

    def test_multihead_jit(self, rng_key):
        """Multi-head attention should compile with JIT."""
        keys = random.split(rng_key, 2)
        n, d_model = 8, 32
        num_heads = 4

        weights = init_multihead_weights(keys[0], d_model, num_heads)
        X = random.normal(keys[1], (n, d_model))

        @jax.jit
        def jitted_multihead(X):
            return multihead_attention(
                X, X, X, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
            )

        O = jitted_multihead(X)
        assert_finite(O, "JIT multihead output")

    def test_gradient_jit(self, rng_key):
        """Gradient computation should work with JIT."""
        keys = random.split(rng_key, 3)
        n, d = 4, 8

        Q = random.normal(keys[0], (n, d))
        K = random.normal(keys[1], (n, d))
        V = random.normal(keys[2], (n, d))

        @jax.jit
        def loss_and_grad(Q, K, V):
            def loss_fn(Q, K, V):
                O = scaled_dot_product_attention(Q, K, V)
                return jnp.sum(O**2)

            loss = loss_fn(Q, K, V)
            grads = jax.grad(loss_fn, argnums=(0, 1, 2))(Q, K, V)
            return loss, grads

        loss, grads = loss_and_grad(Q, K, V)
        assert jnp.isfinite(loss)
        for g in grads:
            assert_finite(g, "JIT gradient")
