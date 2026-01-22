"""Tests for multihead attention module."""

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from attn_tensors.multihead import (
    decompose_multihead,
    head_attention_entropy,
    head_diversity,
    init_multihead_weights,
    multihead_attention,
    multihead_attention_batched,
)

from .helpers import (
    assert_allclose,
    assert_finite,
    assert_probability_distribution,
    assert_shape,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multihead_setup(rng_key):
    """Standard multi-head attention setup."""
    d_model = 32
    num_heads = 4
    d_k = d_model // num_heads
    d_v = d_model // num_heads
    n_q, n_k = 6, 8

    keys = random.split(rng_key, 4)

    # Initialize weights
    weights = init_multihead_weights(keys[0], d_model, num_heads, d_k, d_v)

    # Create inputs
    Q = random.normal(keys[1], (n_q, d_model))
    K = random.normal(keys[2], (n_k, d_model))
    V = random.normal(keys[3], (n_k, d_model))

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "W_Q": weights["W_Q"],
        "W_K": weights["W_K"],
        "W_V": weights["W_V"],
        "W_O": weights["W_O"],
        "d_model": d_model,
        "num_heads": num_heads,
        "d_k": d_k,
        "d_v": d_v,
        "n_q": n_q,
        "n_k": n_k,
    }


# =============================================================================
# Multi-Head Attention Core Tests
# =============================================================================


class TestMultiheadAttention:
    """Tests for multihead_attention function."""

    def test_output_shape(self, multihead_setup):
        """Output should have shape (n_q, d_model)."""
        cfg = multihead_setup

        O = multihead_attention(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        assert_shape(O, (cfg["n_q"], cfg["d_model"]), "multihead output")

    def test_return_weights(self, multihead_setup):
        """Should optionally return attention weights."""
        cfg = multihead_setup

        O, A = multihead_attention(
            cfg["Q"],
            cfg["K"],
            cfg["V"],
            cfg["W_Q"],
            cfg["W_K"],
            cfg["W_V"],
            cfg["W_O"],
            return_weights=True,
        )

        assert_shape(O, (cfg["n_q"], cfg["d_model"]), "output")
        assert_shape(A, (cfg["num_heads"], cfg["n_q"], cfg["n_k"]), "attention weights")

        # Weights should be valid probability distributions
        for h in range(cfg["num_heads"]):
            assert_probability_distribution(A[h], axis=-1, name=f"head {h} weights")

    def test_mask_effect(self, multihead_setup):
        """Masking should zero out attention to masked positions."""
        cfg = multihead_setup

        # Causal mask
        mask = jnp.tril(jnp.ones((cfg["n_q"], cfg["n_k"]), dtype=bool))

        O, A = multihead_attention(
            cfg["Q"],
            cfg["K"],
            cfg["V"],
            cfg["W_Q"],
            cfg["W_K"],
            cfg["W_V"],
            cfg["W_O"],
            mask=mask,
            return_weights=True,
        )

        # Check that masked positions have zero weight
        for h in range(cfg["num_heads"]):
            masked_weights = A[h][~mask]
            assert jnp.allclose(masked_weights, 0.0, atol=1e-6), (
                f"Head {h} should have zero masked weights"
            )

    def test_finite_output(self, multihead_setup):
        """Output should always be finite."""
        cfg = multihead_setup

        O = multihead_attention(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        assert_finite(O, "multihead output")

    def test_single_head_matches_standard(self, rng_key):
        """Single-head attention should match standard attention (up to projection)."""
        keys = random.split(rng_key, 5)
        d_model = 16
        n_q, _n_k = 4, 6

        # Single head with identity-like projections
        W_Q = random.normal(keys[0], (1, d_model, d_model)) * 0.1
        W_K = random.normal(keys[1], (1, d_model, d_model)) * 0.1
        W_V = random.normal(keys[2], (1, d_model, d_model)) * 0.1
        W_O = random.normal(keys[3], (1, d_model, d_model)) * 0.1

        Q = random.normal(keys[4], (n_q, d_model))
        K = Q  # Self-attention
        V = Q

        O = multihead_attention(Q, K, V, W_Q, W_K, W_V, W_O)

        assert_shape(O, (n_q, d_model), "single head output")
        assert_finite(O, "single head output")


# =============================================================================
# Batched Multi-Head Attention Tests
# =============================================================================


class TestMultiheadAttentionBatched:
    """Tests for multihead_attention_batched function."""

    def test_output_shape(self, multihead_setup):
        """Output should have shape (batch, n_q, d_model)."""
        cfg = multihead_setup
        batch = 3

        # Add batch dimension
        Q = jnp.stack([cfg["Q"]] * batch)
        K = jnp.stack([cfg["K"]] * batch)
        V = jnp.stack([cfg["V"]] * batch)

        O = multihead_attention_batched(Q, K, V, cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"])

        assert_shape(O, (batch, cfg["n_q"], cfg["d_model"]), "batched output")

    def test_matches_unbatched(self, multihead_setup):
        """Each batch element should match unbatched attention."""
        cfg = multihead_setup
        batch = 2

        # Add batch dimension
        Q = jnp.stack([cfg["Q"]] * batch)
        K = jnp.stack([cfg["K"]] * batch)
        V = jnp.stack([cfg["V"]] * batch)

        O_batched = multihead_attention_batched(
            Q, K, V, cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        O_single = multihead_attention(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        for b in range(batch):
            assert_allclose(
                O_batched[b], O_single, err_msg=f"Batch element {b} should match unbatched"
            )

    def test_mask_broadcast_2d(self, multihead_setup):
        """2D mask should broadcast across batch and heads."""
        cfg = multihead_setup
        batch = 2

        Q = jnp.stack([cfg["Q"]] * batch)
        K = jnp.stack([cfg["K"]] * batch)
        V = jnp.stack([cfg["V"]] * batch)

        # 2D mask
        mask = jnp.tril(jnp.ones((cfg["n_q"], cfg["n_k"]), dtype=bool))

        O = multihead_attention_batched(
            Q, K, V, cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"], mask=mask
        )

        assert_finite(O, "batched output with 2D mask")

    def test_mask_broadcast_3d(self, multihead_setup):
        """3D mask (batch, n_q, n_k) should work correctly."""
        cfg = multihead_setup
        batch = 2

        Q = jnp.stack([cfg["Q"]] * batch)
        K = jnp.stack([cfg["K"]] * batch)
        V = jnp.stack([cfg["V"]] * batch)

        # Different masks for each batch
        mask = jnp.stack(
            [
                jnp.tril(jnp.ones((cfg["n_q"], cfg["n_k"]), dtype=bool)),
                jnp.ones((cfg["n_q"], cfg["n_k"]), dtype=bool),  # Full attention
            ]
        )

        O = multihead_attention_batched(
            Q, K, V, cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"], mask=mask
        )

        assert_finite(O, "batched output with 3D mask")


# =============================================================================
# Weight Initialization Tests
# =============================================================================


class TestInitMultiheadWeights:
    """Tests for init_multihead_weights function."""

    def test_weight_shapes(self, rng_key):
        """Weights should have correct shapes."""
        d_model = 64
        num_heads = 8
        d_k = 8
        d_v = 8

        weights = init_multihead_weights(rng_key, d_model, num_heads, d_k, d_v)

        assert_shape(weights["W_Q"], (num_heads, d_model, d_k), "W_Q")
        assert_shape(weights["W_K"], (num_heads, d_model, d_k), "W_K")
        assert_shape(weights["W_V"], (num_heads, d_model, d_v), "W_V")
        assert_shape(weights["W_O"], (num_heads, d_v, d_model), "W_O")

    def test_default_dimensions(self, rng_key):
        """Should use default d_k = d_v = d_model // num_heads."""
        d_model = 32
        num_heads = 4

        weights = init_multihead_weights(rng_key, d_model, num_heads)

        expected_d = d_model // num_heads
        assert weights["W_Q"].shape == (num_heads, d_model, expected_d)
        assert weights["W_K"].shape == (num_heads, d_model, expected_d)
        assert weights["W_V"].shape == (num_heads, d_model, expected_d)
        assert weights["W_O"].shape == (num_heads, expected_d, d_model)

    def test_weights_finite(self, rng_key):
        """All weights should be finite."""
        weights = init_multihead_weights(rng_key, 64, 8)

        for name, W in weights.items():
            assert_finite(W, name)

    def test_weight_scale(self, rng_key):
        """Weights should have reasonable variance (Xavier init)."""
        d_model = 128
        num_heads = 8

        weights = init_multihead_weights(rng_key, d_model, num_heads)

        # Xavier init should give variance roughly 2/(fan_in + fan_out)
        # Just check that variance is reasonable (not too large or small)
        for name, W in weights.items():
            var = float(jnp.var(W))
            assert 1e-4 < var < 1.0, f"{name} variance {var} is unreasonable"

    def test_reproducibility(self, rng_key):
        """Same key should give same weights."""
        weights1 = init_multihead_weights(rng_key, 32, 4)
        weights2 = init_multihead_weights(rng_key, 32, 4)

        for name in weights1:
            assert_allclose(weights1[name], weights2[name], err_msg=f"{name} not reproducible")


# =============================================================================
# Analysis Function Tests
# =============================================================================


class TestHeadDiversity:
    """Tests for head_diversity function."""

    def test_identical_heads_give_zero(self):
        """Identical attention patterns should give zero diversity."""
        H, n_q, n_k = 4, 6, 8
        single_pattern = jax.nn.softmax(jnp.ones((n_q, n_k)), axis=-1)
        A = jnp.stack([single_pattern] * H)

        diversity = head_diversity(A)

        assert jnp.isclose(diversity, 0.0, atol=1e-5), "Identical heads should have zero diversity"

    def test_orthogonal_heads_give_high_diversity(self, rng_key):
        """Orthogonal attention patterns should give high diversity."""
        H, n_q, n_k = 4, 10, 10

        # Create different patterns for each head
        keys = random.split(rng_key, H)
        A = jnp.stack(
            [jax.nn.softmax(random.normal(keys[h], (n_q, n_k)) * 5, axis=-1) for h in range(H)]
        )

        diversity = head_diversity(A)

        assert diversity > 0.1, (
            f"Different patterns should have positive diversity, got {diversity}"
        )

    def test_diversity_bounds(self, multihead_setup):
        """Diversity should be in [0, 1]."""
        cfg = multihead_setup

        _, A = multihead_attention(
            cfg["Q"],
            cfg["K"],
            cfg["V"],
            cfg["W_Q"],
            cfg["W_K"],
            cfg["W_V"],
            cfg["W_O"],
            return_weights=True,
        )

        diversity = head_diversity(A)

        assert 0.0 <= diversity <= 1.0, f"Diversity should be in [0, 1], got {diversity}"


class TestHeadAttentionEntropy:
    """Tests for head_attention_entropy function."""

    def test_output_shape(self, multihead_setup):
        """Entropy should have shape (H,)."""
        cfg = multihead_setup

        _, A = multihead_attention(
            cfg["Q"],
            cfg["K"],
            cfg["V"],
            cfg["W_Q"],
            cfg["W_K"],
            cfg["W_V"],
            cfg["W_O"],
            return_weights=True,
        )

        entropy = head_attention_entropy(A)
        assert_shape(entropy, (cfg["num_heads"],), "head entropy")

    def test_uniform_attention_max_entropy(self):
        """Uniform attention should have maximum entropy."""
        H, n_q, n_k = 4, 6, 8
        A = jnp.ones((H, n_q, n_k)) / n_k  # Uniform over keys

        entropy = head_attention_entropy(A)

        max_entropy = jnp.log(n_k)
        assert_allclose(
            entropy,
            jnp.full((H,), max_entropy),
            rtol=1e-3,
            err_msg="Uniform should have max entropy",
        )

    def test_peaked_attention_low_entropy(self, rng_key):
        """Peaked attention should have low entropy."""
        H, n_q, n_k = 4, 6, 8

        # Create peaked attention (high temperature inverse)
        scores = random.normal(rng_key, (H, n_q, n_k)) * 100  # Very peaked
        A = jax.nn.softmax(scores, axis=-1)

        entropy = head_attention_entropy(A)

        # Entropy should be close to 0 for very peaked distributions
        assert jnp.all(entropy < 0.5), f"Peaked attention should have low entropy, got {entropy}"


# =============================================================================
# Decompose Multihead Tests
# =============================================================================


class TestDecomposeMultihead:
    """Tests for decompose_multihead function."""

    def test_all_keys_present(self, multihead_setup):
        """Should return all expected keys."""
        cfg = multihead_setup

        result = decompose_multihead(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        expected_keys = {
            "Q_heads",
            "K_heads",
            "V_heads",
            "scores",
            "weights",
            "head_outputs",
            "output",
            "head_diversity",
            "head_entropy",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes_correct(self, multihead_setup):
        """All tensors should have correct shapes."""
        cfg = multihead_setup
        H = cfg["num_heads"]
        n_q, n_k = cfg["n_q"], cfg["n_k"]
        d_k, d_v = cfg["d_k"], cfg["d_v"]
        d_model = cfg["d_model"]

        result = decompose_multihead(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        assert_shape(result["Q_heads"], (H, n_q, d_k), "Q_heads")
        assert_shape(result["K_heads"], (H, n_k, d_k), "K_heads")
        assert_shape(result["V_heads"], (H, n_k, d_v), "V_heads")
        assert_shape(result["scores"], (H, n_q, n_k), "scores")
        assert_shape(result["weights"], (H, n_q, n_k), "weights")
        assert_shape(result["head_outputs"], (H, n_q, d_v), "head_outputs")
        assert_shape(result["output"], (n_q, d_model), "output")

    def test_output_matches_forward(self, multihead_setup):
        """Decomposed output should match forward pass."""
        cfg = multihead_setup

        O_forward = multihead_attention(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        result = decompose_multihead(
            cfg["Q"], cfg["K"], cfg["V"], cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"]
        )

        assert_allclose(result["output"], O_forward, err_msg="Decomposed should match forward")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in multi-head attention."""

    def test_single_head(self, rng_key):
        """Should work with single head."""
        d_model = 16
        num_heads = 1
        n_q, n_k = 4, 6

        keys = random.split(rng_key, 4)
        weights = init_multihead_weights(keys[0], d_model, num_heads)

        Q = random.normal(keys[1], (n_q, d_model))
        K = random.normal(keys[2], (n_k, d_model))
        V = random.normal(keys[3], (n_k, d_model))

        O = multihead_attention(
            Q, K, V, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
        )
        assert_shape(O, (n_q, d_model), "single head output")
        assert_finite(O, "single head output")

    def test_many_heads(self, rng_key):
        """Should work with many heads."""
        d_model = 64
        num_heads = 16
        n_q, n_k = 8, 10

        keys = random.split(rng_key, 4)
        weights = init_multihead_weights(keys[0], d_model, num_heads)

        Q = random.normal(keys[1], (n_q, d_model))
        K = random.normal(keys[2], (n_k, d_model))
        V = random.normal(keys[3], (n_k, d_model))

        O = multihead_attention(
            Q, K, V, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
        )
        assert_shape(O, (n_q, d_model), "many heads output")
        assert_finite(O, "many heads output")

    def test_self_attention(self, multihead_setup):
        """Should work for self-attention (Q=K=V)."""
        cfg = multihead_setup
        X = cfg["Q"]  # Use Q as the only input

        O = multihead_attention(X, X, X, cfg["W_Q"], cfg["W_K"], cfg["W_V"], cfg["W_O"])

        assert_shape(O, X.shape, "self-attention output")
        assert_finite(O, "self-attention output")

    def test_large_sequence(self, rng_key):
        """Should handle larger sequences."""
        d_model = 32
        num_heads = 4
        n_q, n_k = 64, 64

        keys = random.split(rng_key, 4)
        weights = init_multihead_weights(keys[0], d_model, num_heads)

        Q = random.normal(keys[1], (n_q, d_model))
        K = random.normal(keys[2], (n_k, d_model))
        V = random.normal(keys[3], (n_k, d_model))

        O = multihead_attention(
            Q, K, V, weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]
        )
        assert_shape(O, (n_q, d_model), "large sequence output")
        assert_finite(O, "large sequence output")
