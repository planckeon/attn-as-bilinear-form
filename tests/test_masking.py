"""Tests for masking module."""

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from attn_tensors.masking import (
    apply_mask,
    attention_mask_from_padding,
    block_sparse_mask,
    causal_mask,
    causal_mask_with_window,
    causal_padding_mask,
    global_local_mask,
    local_attention_mask,
    mask_to_additive,
    padding_mask,
    strided_attention_mask,
    visualize_mask,
)

from .helpers import (
    assert_allclose,
    assert_shape,
)

# =============================================================================
# Causal Mask Tests
# =============================================================================


class TestCausalMask:
    """Tests for causal_mask function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        for n in [1, 4, 8, 16]:
            mask = causal_mask(n)
            assert_shape(mask, (n, n), f"causal mask n={n}")

    def test_is_lower_triangular(self):
        """Mask should be lower triangular."""
        n = 6
        mask = causal_mask(n)

        # Upper triangle (excluding diagonal) should be False
        for i in range(n):
            for j in range(i + 1, n):
                assert not mask[i, j], f"Position ({i}, {j}) should be False"

    def test_diagonal_true(self):
        """Diagonal should be True (can attend to self)."""
        n = 5
        mask = causal_mask(n)

        for i in range(n):
            assert mask[i, i], f"Diagonal ({i}, {i}) should be True"

    def test_lower_triangle_true(self):
        """Lower triangle should be True (can attend to past)."""
        n = 5
        mask = causal_mask(n)

        for i in range(n):
            for j in range(i + 1):
                assert mask[i, j], f"Position ({i}, {j}) should be True"

    def test_dtype(self):
        """Mask should be boolean."""
        mask = causal_mask(4)
        assert mask.dtype == jnp.bool_

    def test_size_1(self):
        """Size 1 should give [[True]]."""
        mask = causal_mask(1)
        expected = jnp.array([[True]])
        assert jnp.array_equal(mask, expected)


class TestCausalMaskWithWindow:
    """Tests for causal_mask_with_window function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        mask = causal_mask_with_window(8, 3)
        assert_shape(mask, (8, 8), "windowed causal mask")

    def test_large_window_matches_causal(self):
        """Window >= n should match standard causal mask."""
        n = 6
        mask_window = causal_mask_with_window(n, n + 10)
        mask_causal = causal_mask(n)

        assert jnp.array_equal(mask_window, mask_causal)

    def test_window_limit(self):
        """Should only attend to positions within window."""
        n, window = 8, 3
        mask = causal_mask_with_window(n, window)

        for i in range(n):
            for j in range(n):
                if j > i:  # Future - should be False
                    assert not mask[i, j]
                elif i - j >= window:  # Too far in past - should be False
                    assert not mask[i, j]
                else:  # Within window and not future - should be True
                    assert mask[i, j]

    def test_window_1(self):
        """Window 1 should only attend to self."""
        n = 5
        mask = causal_mask_with_window(n, 1)
        expected = jnp.eye(n, dtype=bool)

        assert jnp.array_equal(mask, expected)


# =============================================================================
# Padding Mask Tests
# =============================================================================


class TestPaddingMask:
    """Tests for padding_mask function."""

    def test_shape(self):
        """Mask should have shape (batch, max_len)."""
        lengths = jnp.array([3, 5, 7])
        mask = padding_mask(lengths, max_len=10)
        assert_shape(mask, (3, 10), "padding mask")

    def test_masks_padding(self):
        """Positions >= length should be False."""
        lengths = jnp.array([3, 5, 7])
        max_len = 10
        mask = padding_mask(lengths, max_len)

        for b in range(3):
            for j in range(max_len):
                if j < lengths[b]:
                    assert mask[b, j], f"Position ({b}, {j}) should be True"
                else:
                    assert not mask[b, j], f"Position ({b}, {j}) should be False"

    def test_all_valid(self):
        """Full-length sequences should have all True."""
        lengths = jnp.array([5, 5])
        mask = padding_mask(lengths, max_len=5)

        assert jnp.all(mask)

    def test_all_padding(self):
        """Zero-length sequences should have all False."""
        lengths = jnp.array([0, 0])
        mask = padding_mask(lengths, max_len=5)

        assert not jnp.any(mask)


class TestAttentionMaskFromPadding:
    """Tests for attention_mask_from_padding function."""

    def test_shape(self):
        """Mask should have shape (batch, max_q, max_k)."""
        q_lengths = jnp.array([3, 5])
        k_lengths = jnp.array([4, 6])
        mask = attention_mask_from_padding(q_lengths, k_lengths, max_q=6, max_k=8)
        assert_shape(mask, (2, 6, 8), "attention padding mask")

    def test_masks_both(self):
        """Position (i, j) should be True only if both i and j are valid."""
        q_lengths = jnp.array([3])
        k_lengths = jnp.array([4])
        mask = attention_mask_from_padding(q_lengths, k_lengths, max_q=5, max_k=6)

        # Valid positions: query < 3 AND key < 4
        for i in range(5):
            for j in range(6):
                expected = (i < 3) and (j < 4)
                assert mask[0, i, j] == expected, f"Position ({i}, {j})"


class TestCausalPaddingMask:
    """Tests for causal_padding_mask function."""

    def test_shape(self):
        """Mask should have shape (batch, max_len, max_len)."""
        lengths = jnp.array([3, 5])
        mask = causal_padding_mask(lengths, max_len=6)
        assert_shape(mask, (2, 6, 6), "causal padding mask")

    def test_combines_causal_and_padding(self):
        """Should be causal AND respect padding."""
        lengths = jnp.array([4])
        max_len = 6
        mask = causal_padding_mask(lengths, max_len)

        for i in range(max_len):
            for j in range(max_len):
                causal_ok = j <= i
                padding_ok = (i < 4) and (j < 4)
                expected = causal_ok and padding_ok
                assert mask[0, i, j] == expected, f"Position ({i}, {j})"


# =============================================================================
# Sparse Pattern Tests
# =============================================================================


class TestLocalAttentionMask:
    """Tests for local_attention_mask function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        mask = local_attention_mask(8, window=3)
        assert_shape(mask, (8, 8), "local mask")

    def test_window_symmetry(self):
        """Local attention should be symmetric."""
        n, window = 8, 5
        mask = local_attention_mask(n, window)

        assert jnp.array_equal(mask, mask.T), "Local mask should be symmetric"

    def test_window_3_pattern(self):
        """Window 3 should allow distance <= 1."""
        n = 5
        mask = local_attention_mask(n, window=3)

        for i in range(n):
            for j in range(n):
                expected = abs(i - j) <= 1
                assert mask[i, j] == expected, f"Position ({i}, {j})"

    def test_window_1(self):
        """Window 1 should only allow self-attention."""
        n = 5
        mask = local_attention_mask(n, window=1)
        expected = jnp.eye(n, dtype=bool)

        assert jnp.array_equal(mask, expected)


class TestStridedAttentionMask:
    """Tests for strided_attention_mask function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        mask = strided_attention_mask(8, stride=2)
        assert_shape(mask, (8, 8), "strided mask")

    def test_stride_2_pattern(self):
        """Stride 2 should connect even with even, odd with odd."""
        n = 6
        mask = strided_attention_mask(n, stride=2)

        for i in range(n):
            for j in range(n):
                expected = (i % 2) == (j % 2)
                assert mask[i, j] == expected, f"Position ({i}, {j})"

    def test_stride_1(self):
        """Stride 1 should allow full attention."""
        n = 5
        mask = strided_attention_mask(n, stride=1)

        assert jnp.all(mask)


class TestBlockSparseMask:
    """Tests for block_sparse_mask function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        mask = block_sparse_mask(8, block_size=2)
        assert_shape(mask, (8, 8), "block sparse mask")

    def test_block_pattern(self):
        """Should only attend within blocks."""
        n, block_size = 6, 2
        mask = block_sparse_mask(n, block_size)

        for i in range(n):
            for j in range(n):
                same_block = (i // block_size) == (j // block_size)
                assert mask[i, j] == same_block, f"Position ({i}, {j})"

    def test_block_1(self):
        """Block size 1 should only allow self-attention."""
        n = 5
        mask = block_sparse_mask(n, block_size=1)
        expected = jnp.eye(n, dtype=bool)

        assert jnp.array_equal(mask, expected)


class TestGlobalLocalMask:
    """Tests for global_local_mask function."""

    def test_shape(self):
        """Mask should have shape (n, n)."""
        mask = global_local_mask(8, window=3, global_tokens=2)
        assert_shape(mask, (8, 8), "global-local mask")

    def test_global_tokens_attend_everywhere(self):
        """Global tokens should attend to all positions."""
        n, global_tokens = 8, 2
        mask = global_local_mask(n, window=3, global_tokens=global_tokens)

        # First global_tokens rows should be all True
        assert jnp.all(mask[:global_tokens, :])

    def test_all_attend_to_global(self):
        """All tokens should attend to global tokens."""
        n, global_tokens = 8, 2
        mask = global_local_mask(n, window=3, global_tokens=global_tokens)

        # First global_tokens columns should be all True
        assert jnp.all(mask[:, :global_tokens])

    def test_local_for_non_global(self):
        """Non-global tokens should use local attention for non-global positions."""
        n, window, global_tokens = 10, 3, 2
        mask = global_local_mask(n, window, global_tokens)
        local = local_attention_mask(n, window)

        # For non-global tokens attending to non-global positions
        for i in range(global_tokens, n):
            for j in range(global_tokens, n):
                # Should have local attention
                assert mask[i, j] == local[i, j], f"Position ({i}, {j})"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestApplyMask:
    """Tests for apply_mask function."""

    def test_keeps_unmasked(self):
        """Unmasked positions should keep original values."""
        scores = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[True, True], [True, True]])

        result = apply_mask(scores, mask)
        assert_allclose(result, scores)

    def test_masks_with_fill_value(self):
        """Masked positions should get fill_value."""
        scores = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[True, False], [False, True]])
        fill = -1e9

        result = apply_mask(scores, mask, fill_value=fill)

        expected = jnp.array([[1.0, fill], [fill, 4.0]])
        assert_allclose(result, expected)

    def test_broadcasts_correctly(self):
        """Should handle broadcasting for batched scores."""
        scores = jnp.ones((2, 3, 4))  # Batched
        mask = jnp.tril(jnp.ones((3, 4), dtype=bool))  # Shared mask

        result = apply_mask(scores, mask)

        # Upper triangle should be fill value
        assert jnp.all(result[:, 0, 1:] == -1e9)


class TestMaskToAdditive:
    """Tests for mask_to_additive function."""

    def test_true_gives_zero(self):
        """True positions should become 0."""
        mask = jnp.array([[True, True], [True, True]])
        additive = mask_to_additive(mask)

        assert_allclose(additive, jnp.zeros((2, 2)))

    def test_false_gives_fill(self):
        """False positions should become fill_value."""
        mask = jnp.array([[False, False], [False, False]])
        fill = -1e9
        additive = mask_to_additive(mask, fill_value=fill)

        expected = jnp.full((2, 2), fill)
        assert_allclose(additive, expected)

    def test_can_add_to_scores(self):
        """Should work when added to scores."""
        scores = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[True, False], [True, True]])

        additive = mask_to_additive(mask)
        result = scores + additive

        # Masked position should be very negative
        assert result[0, 1] < -1e8


class TestVisualizeMask:
    """Tests for visualize_mask function."""

    def test_returns_string(self):
        """Should return a string."""
        mask = causal_mask(4)
        viz = visualize_mask(mask)

        assert isinstance(viz, str)

    def test_correct_symbols(self):
        """Should use # for True and . for False."""
        mask = jnp.array([[True, False], [True, True]])
        viz = visualize_mask(mask)

        lines = viz.split("\n")
        assert lines[0] == "#."
        assert lines[1] == "##"

    def test_identity_visualization(self):
        """Identity mask should give diagonal pattern."""
        mask = jnp.eye(3, dtype=bool)
        viz = visualize_mask(mask)

        expected = "#..\n.#.\n..#"
        assert viz == expected


# =============================================================================
# Property Tests
# =============================================================================


class TestMaskProperties:
    """Property-based tests for masks."""

    @given(st.integers(min_value=1, max_value=20))
    @settings(deadline=None)
    def test_causal_mask_count(self, n):
        """Causal mask should have n(n+1)/2 True values."""
        mask = causal_mask(n)
        expected_count = n * (n + 1) // 2

        assert int(jnp.sum(mask)) == expected_count

    @given(st.integers(min_value=1, max_value=20))
    @settings(deadline=None)
    def test_local_mask_symmetric(self, n):
        """Local mask should always be symmetric."""
        window = max(1, n // 2)
        mask = local_attention_mask(n, window)

        assert jnp.array_equal(mask, mask.T)

    @given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=5))
    @settings(deadline=None)
    def test_block_sparse_count(self, n_blocks, block_size):
        """Block sparse mask should have exactly block_size^2 * n_blocks True values."""
        n = n_blocks * block_size
        mask = block_sparse_mask(n, block_size)

        expected_count = n_blocks * (block_size**2)
        assert int(jnp.sum(mask)) == expected_count
