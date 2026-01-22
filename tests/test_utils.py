"""
Tests for utils.py visualization functions.

These tests verify that visualization functions:
1. Execute without errors
2. Return proper matplotlib figure objects
3. Handle edge cases (single head, empty inputs, etc.)

Note: We use matplotlib's Agg backend to avoid display issues in CI.
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib.figure import Figure, SubFigure

from attn_tensors.bilinear import euclidean_metric, scaled_euclidean_metric
from attn_tensors.utils import (
    compare_metrics,
    plot_attention_weights,
    plot_entropy_distribution,
    plot_gradient_flow,
    plot_hopfield_energy,
    plot_mask,
    plot_multihead_attention,
    plot_temperature_sweep,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng_key():
    return random.PRNGKey(42)


@pytest.fixture
def attention_weights_2d(rng_key):
    """Random attention weights (n_q=4, n_k=5)."""
    logits = random.normal(rng_key, (4, 5))
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)


@pytest.fixture
def attention_weights_3d(rng_key):
    """Random multi-head attention weights (H=3, n_q=4, n_k=5)."""
    logits = random.normal(rng_key, (3, 4, 5))
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)


@pytest.fixture
def scores_1d(rng_key):
    """Random scores for temperature sweep (n=5)."""
    return random.normal(rng_key, (5,))


@pytest.fixture
def gradient_flow_data(rng_key):
    """Sample gradient flow data."""
    keys = random.split(rng_key, 3)
    return {
        "dL_dQ": random.normal(keys[0], (4, 8)),
        "dL_dK": random.normal(keys[1], (5, 8)),
        "dL_dV": random.normal(keys[2], (5, 6)),
    }


@pytest.fixture
def boolean_mask():
    """Causal mask (lower triangular)."""
    n = 5
    return jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))


@pytest.fixture
def patterns_2d(rng_key):
    """2D patterns for Hopfield visualization."""
    return random.normal(rng_key, (3, 2))


@pytest.fixture
def qk_for_metrics(rng_key):
    """Q and K for metric comparison."""
    keys = random.split(rng_key, 2)
    Q = random.normal(keys[0], (4, 8))
    K = random.normal(keys[1], (5, 8))
    return Q, K


# =============================================================================
# Helper to close figures after tests
# =============================================================================


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# =============================================================================
# Tests for plot_attention_weights
# =============================================================================


class TestPlotAttentionWeights:
    """Tests for plot_attention_weights function."""

    def test_returns_figure(self, attention_weights_2d):
        """Should return a matplotlib figure."""
        fig = plot_attention_weights(attention_weights_2d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_with_labels(self, attention_weights_2d):
        """Should work with custom labels."""
        query_labels = ["q0", "q1", "q2", "q3"]
        key_labels = ["k0", "k1", "k2", "k3", "k4"]
        fig = plot_attention_weights(
            attention_weights_2d,
            query_labels=query_labels,
            key_labels=key_labels,
        )
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_title(self, attention_weights_2d):
        """Should accept custom title."""
        fig = plot_attention_weights(attention_weights_2d, title="Custom Title")
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_cmap(self, attention_weights_2d):
        """Should accept custom colormap."""
        fig = plot_attention_weights(attention_weights_2d, cmap="Reds")
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, attention_weights_2d):
        """Should accept custom figure size."""
        fig = plot_attention_weights(attention_weights_2d, figsize=(10, 8))
        assert isinstance(fig, (Figure, SubFigure))

    def test_with_existing_axes(self, attention_weights_2d):
        """Should work with provided axes."""
        _, ax = plt.subplots()
        fig = plot_attention_weights(attention_weights_2d, ax=ax)
        assert isinstance(fig, (Figure, SubFigure))

    def test_small_matrix(self):
        """Should work with small 2x2 matrix."""
        A = jnp.array([[0.7, 0.3], [0.4, 0.6]])
        fig = plot_attention_weights(A)
        assert isinstance(fig, (Figure, SubFigure))

    def test_single_row(self):
        """Should work with single query."""
        A = jnp.array([[0.2, 0.3, 0.5]])
        fig = plot_attention_weights(A)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_multihead_attention
# =============================================================================


class TestPlotMultiheadAttention:
    """Tests for plot_multihead_attention function."""

    def test_returns_figure(self, attention_weights_3d):
        """Should return a matplotlib figure."""
        fig = plot_multihead_attention(attention_weights_3d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_with_head_names(self, attention_weights_3d):
        """Should work with custom head names."""
        head_names = ["Query", "Key", "Value"]
        fig = plot_multihead_attention(attention_weights_3d, head_names=head_names)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, attention_weights_3d):
        """Should accept custom figure size."""
        fig = plot_multihead_attention(attention_weights_3d, figsize=(15, 5))
        assert isinstance(fig, (Figure, SubFigure))

    def test_single_head(self, rng_key):
        """Should work with single head."""
        logits = random.normal(rng_key, (1, 4, 5))
        A = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
        fig = plot_multihead_attention(A)
        assert isinstance(fig, (Figure, SubFigure))

    def test_many_heads(self, rng_key):
        """Should work with many heads."""
        logits = random.normal(rng_key, (8, 4, 5))
        A = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
        fig = plot_multihead_attention(A)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_entropy_distribution
# =============================================================================


class TestPlotEntropyDistribution:
    """Tests for plot_entropy_distribution function."""

    def test_returns_figure_2d(self, attention_weights_2d):
        """Should return figure for 2D input."""
        fig = plot_entropy_distribution(attention_weights_2d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_returns_figure_3d(self, attention_weights_3d):
        """Should return figure for 3D multi-head input."""
        fig = plot_entropy_distribution(attention_weights_3d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_title(self, attention_weights_2d):
        """Should accept custom title."""
        fig = plot_entropy_distribution(attention_weights_2d, title="Custom Title")
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, attention_weights_2d):
        """Should accept custom figure size."""
        fig = plot_entropy_distribution(attention_weights_2d, figsize=(10, 6))
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_temperature_sweep
# =============================================================================


class TestPlotTemperatureSweep:
    """Tests for plot_temperature_sweep function."""

    def test_returns_figure(self, scores_1d):
        """Should return a matplotlib figure."""
        fig = plot_temperature_sweep(scores_1d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_temperatures(self, scores_1d):
        """Should work with custom temperatures."""
        temps = jnp.array([0.1, 1.0, 10.0])
        fig = plot_temperature_sweep(scores_1d, temperatures=temps)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, scores_1d):
        """Should accept custom figure size."""
        fig = plot_temperature_sweep(scores_1d, figsize=(12, 5))
        assert isinstance(fig, (Figure, SubFigure))

    def test_short_scores(self):
        """Should work with short score vector."""
        scores = jnp.array([1.0, 2.0])
        fig = plot_temperature_sweep(scores)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_gradient_flow
# =============================================================================


class TestPlotGradientFlow:
    """Tests for plot_gradient_flow function."""

    def test_returns_figure(self, gradient_flow_data):
        """Should return a matplotlib figure."""
        fig = plot_gradient_flow(gradient_flow_data)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, gradient_flow_data):
        """Should accept custom figure size."""
        fig = plot_gradient_flow(gradient_flow_data, figsize=(15, 5))
        assert isinstance(fig, (Figure, SubFigure))

    def test_small_gradients(self):
        """Should work with small gradient tensors."""
        data = {
            "dL_dQ": jnp.array([[1.0, 2.0]]),
            "dL_dK": jnp.array([[3.0, 4.0]]),
            "dL_dV": jnp.array([[5.0, 6.0]]),
        }
        fig = plot_gradient_flow(data)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_mask
# =============================================================================


class TestPlotMask:
    """Tests for plot_mask function."""

    def test_returns_figure(self, boolean_mask):
        """Should return a matplotlib figure."""
        fig = plot_mask(boolean_mask)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_title(self, boolean_mask):
        """Should accept custom title."""
        fig = plot_mask(boolean_mask, title="Causal Mask")
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, boolean_mask):
        """Should accept custom figure size."""
        fig = plot_mask(boolean_mask, figsize=(8, 8))
        assert isinstance(fig, (Figure, SubFigure))

    def test_all_true_mask(self):
        """Should work with all-true mask."""
        mask = jnp.ones((4, 4), dtype=jnp.bool_)
        fig = plot_mask(mask)
        assert isinstance(fig, (Figure, SubFigure))

    def test_all_false_mask(self):
        """Should work with all-false mask."""
        mask = jnp.zeros((4, 4), dtype=jnp.bool_)
        fig = plot_mask(mask)
        assert isinstance(fig, (Figure, SubFigure))

    def test_rectangular_mask(self):
        """Should work with non-square mask."""
        mask = jnp.ones((3, 5), dtype=jnp.bool_)
        fig = plot_mask(mask)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for plot_hopfield_energy
# =============================================================================


class TestPlotHopfieldEnergy:
    """Tests for plot_hopfield_energy function."""

    def test_returns_figure(self, patterns_2d):
        """Should return a matplotlib figure."""
        fig = plot_hopfield_energy(patterns_2d)
        assert isinstance(fig, (Figure, SubFigure))

    def test_with_trajectory(self, patterns_2d, rng_key):
        """Should work with state trajectory."""
        trajectory = [random.normal(k, (2,)) for k in random.split(rng_key, 5)]
        fig = plot_hopfield_energy(patterns_2d, state_trajectory=trajectory)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, patterns_2d):
        """Should accept custom figure size."""
        fig = plot_hopfield_energy(patterns_2d, figsize=(12, 5))
        assert isinstance(fig, (Figure, SubFigure))

    def test_single_pattern(self, rng_key):
        """Should work with single pattern."""
        pattern = random.normal(rng_key, (1, 2))
        fig = plot_hopfield_energy(pattern)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Tests for compare_metrics
# =============================================================================


class TestCompareMetrics:
    """Tests for compare_metrics function."""

    def test_returns_figure(self, qk_for_metrics):
        """Should return a matplotlib figure."""
        Q, K = qk_for_metrics
        d = Q.shape[-1]
        metrics = {
            "Euclidean": euclidean_metric(d),
            "Scaled": scaled_euclidean_metric(d),
        }
        fig = compare_metrics(Q, K, metrics)
        assert isinstance(fig, (Figure, SubFigure))

    def test_single_metric(self, qk_for_metrics):
        """Should work with single metric."""
        Q, K = qk_for_metrics
        d = Q.shape[-1]
        metrics = {"Euclidean": euclidean_metric(d)}
        fig = compare_metrics(Q, K, metrics)
        assert isinstance(fig, (Figure, SubFigure))

    def test_custom_figsize(self, qk_for_metrics):
        """Should accept custom figure size."""
        Q, K = qk_for_metrics
        d = Q.shape[-1]
        metrics = {"Euclidean": euclidean_metric(d)}
        fig = compare_metrics(Q, K, metrics, figsize=(6, 6))
        assert isinstance(fig, (Figure, SubFigure))

    def test_many_metrics(self, qk_for_metrics):
        """Should work with many metrics."""
        Q, K = qk_for_metrics
        d = Q.shape[-1]
        metrics = {
            "Euclidean": euclidean_metric(d),
            "Scaled 1": scaled_euclidean_metric(d, scale=0.5),
            "Scaled 2": scaled_euclidean_metric(d, scale=1.0),
            "Scaled 3": scaled_euclidean_metric(d, scale=2.0),
        }
        fig = compare_metrics(Q, K, metrics)
        assert isinstance(fig, (Figure, SubFigure))


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case tests for utils module."""

    def test_attention_weights_uniform(self):
        """Uniform attention weights should visualize correctly."""
        A = jnp.ones((4, 4)) / 4
        fig = plot_attention_weights(A)
        assert isinstance(fig, (Figure, SubFigure))

    def test_attention_weights_one_hot(self):
        """One-hot attention weights should visualize correctly."""
        A = jnp.eye(4)
        fig = plot_attention_weights(A)
        assert isinstance(fig, (Figure, SubFigure))

    def test_very_small_attention(self):
        """Very small attention values should visualize correctly."""
        A = jnp.array([[1e-10, 1 - 1e-10], [1 - 1e-10, 1e-10]])
        fig = plot_attention_weights(A)
        assert isinstance(fig, (Figure, SubFigure))
