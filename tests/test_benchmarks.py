"""Benchmark tests for attention operations.

These tests require pytest-benchmark to be installed:
    uv pip install pytest-benchmark

Run with: uv run pytest tests/test_benchmarks.py -v --benchmark-only
Or: uv run pytest tests/test_benchmarks.py -v -m benchmark

When pytest-benchmark is not installed, only the non-benchmark tests will run.
"""

import pytest
import jax
import jax.random as random

from attn_tensors.attention import scaled_dot_product_attention
from attn_tensors.bilinear import bilinear_form_batch, scaled_euclidean_metric
from attn_tensors.multihead import multihead_attention
from attn_tensors.softmax import gibbs_distribution


# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

requires_benchmark = pytest.mark.skipif(
    not HAS_BENCHMARK,
    reason="pytest-benchmark not installed",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def benchmark_key():
    """Fixed key for reproducible benchmarks."""
    return random.PRNGKey(42)


def generate_qkv(key, n_q, n_k, d):
    """Generate random Q, K, V tensors."""
    keys = random.split(key, 3)
    Q = random.normal(keys[0], (n_q, d))
    K = random.normal(keys[1], (n_k, d))
    V = random.normal(keys[2], (n_k, d))
    return Q, K, V


# =============================================================================
# Attention Benchmarks
# =============================================================================


@requires_benchmark
@pytest.mark.benchmark
class TestAttentionBenchmarks:
    """Benchmarks for core attention operations."""

    @pytest.mark.parametrize(
        "n,d",
        [
            (32, 64),
            (64, 64),
            (128, 64),
            (256, 64),
            (512, 64),
        ],
    )
    def test_attention_scaling_seq_length(self, benchmark, benchmark_key, n, d):
        """Benchmark attention with varying sequence length."""
        Q, K, V = generate_qkv(benchmark_key, n, n, d)

        # Warmup
        _ = scaled_dot_product_attention(Q, K, V)

        # Benchmark
        result = benchmark(scaled_dot_product_attention, Q, K, V)
        assert result.shape == (n, d)

    @pytest.mark.parametrize(
        "n,d",
        [
            (64, 32),
            (64, 64),
            (64, 128),
            (64, 256),
        ],
    )
    def test_attention_scaling_dim(self, benchmark, benchmark_key, n, d):
        """Benchmark attention with varying feature dimension."""
        Q, K, V = generate_qkv(benchmark_key, n, n, d)

        # Warmup
        _ = scaled_dot_product_attention(Q, K, V)

        # Benchmark
        result = benchmark(scaled_dot_product_attention, Q, K, V)
        assert result.shape == (n, d)


@requires_benchmark
@pytest.mark.benchmark
class TestBilinearBenchmarks:
    """Benchmarks for bilinear form operations."""

    @pytest.mark.parametrize(
        "n,d",
        [
            (64, 64),
            (128, 64),
            (256, 64),
        ],
    )
    def test_bilinear_form_batch(self, benchmark, benchmark_key, n, d):
        """Benchmark batch bilinear form computation."""
        keys = random.split(benchmark_key, 2)
        Q = random.normal(keys[0], (n, d))
        K = random.normal(keys[1], (n, d))
        g = scaled_euclidean_metric(d)

        # Warmup
        _ = bilinear_form_batch(Q, K, g)

        # Benchmark
        result = benchmark(bilinear_form_batch, Q, K, g)
        assert result.shape == (n, n)


@requires_benchmark
@pytest.mark.benchmark
class TestMultiheadBenchmarks:
    """Benchmarks for multi-head attention."""

    @pytest.mark.parametrize(
        "n,d,h",
        [
            (64, 64, 4),
            (64, 64, 8),
            (128, 64, 8),
            (128, 128, 8),
        ],
    )
    def test_multihead_attention(self, benchmark, benchmark_key, n, d, h):
        """Benchmark multi-head attention."""
        Q, K, V = generate_qkv(benchmark_key, n, n, d)

        # Warmup
        _ = multihead_attention(Q, K, V, num_heads=h)

        # Benchmark
        result = benchmark(multihead_attention, Q, K, V, num_heads=h)
        assert result.shape == (n, d)


@requires_benchmark
@pytest.mark.benchmark
class TestSoftmaxBenchmarks:
    """Benchmarks for softmax operations."""

    @pytest.mark.parametrize("n", [64, 128, 256, 512])
    def test_softmax_temperature(self, benchmark, benchmark_key, n):
        """Benchmark softmax with temperature."""
        scores = random.normal(benchmark_key, (n, n))

        # Warmup
        _ = gibbs_distribution(scores, temperature=1.0)

        # Benchmark
        result = benchmark(gibbs_distribution, scores, temperature=1.0)
        assert result.shape == (n, n)


# =============================================================================
# JIT Compilation Benchmarks
# =============================================================================


@requires_benchmark
@pytest.mark.benchmark
class TestJITBenchmarks:
    """Compare JIT vs non-JIT performance."""

    def test_attention_jit_vs_nojit(self, benchmark, benchmark_key):
        """Compare JIT-compiled vs interpreted attention."""
        n, d = 128, 64
        Q, K, V = generate_qkv(benchmark_key, n, n, d)

        # JIT-compiled version
        attention_jit = jax.jit(scaled_dot_product_attention)

        # Warmup JIT
        _ = attention_jit(Q, K, V)

        # Benchmark JIT version
        result = benchmark(attention_jit, Q, K, V)
        assert result.shape == (n, d)


# =============================================================================
# Memory Usage Tests (not benchmarks, always run)
# =============================================================================


class TestMemoryUsage:
    """Tests to verify memory characteristics."""

    def test_attention_output_shape(self, benchmark_key):
        """Verify attention produces expected output shapes."""
        n_q, n_k, d = 10, 20, 64
        Q, K, V = generate_qkv(benchmark_key, n_q, n_k, d)

        output, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)

        assert output.shape == (n_q, d)
        assert weights.shape == (n_q, n_k)

    def test_quadratic_memory_in_weights(self, benchmark_key):
        """Attention weights are O(n_q * n_k) in memory."""
        sizes = [(10, 10), (20, 20), (40, 40)]
        d = 64

        for n_q, n_k in sizes:
            Q, K, V = generate_qkv(benchmark_key, n_q, n_k, d)
            _, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
            assert weights.size == n_q * n_k
