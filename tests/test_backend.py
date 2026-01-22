"""Tests for backend detection and abstraction."""

import pytest

from attn_tensors.backend import (
    Backend,
    get_array_module,
    get_backend,
    is_jax_available,
    is_mlx_available,
)


class TestBackendDetection:
    """Test backend detection functionality."""

    def test_jax_available(self):
        """JAX should always be available (it's a required dependency)."""
        assert is_jax_available() is True

    def test_get_backend_returns_valid_backend(self):
        """get_backend should return a valid Backend enum."""
        backend = get_backend()
        assert isinstance(backend, Backend)
        assert backend in [Backend.JAX, Backend.MLX]

    def test_get_backend_caching(self):
        """Backend detection should be cached."""
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_get_array_module_jax(self):
        """Should be able to get JAX array module."""
        jnp = get_array_module(Backend.JAX)
        assert hasattr(jnp, "array")
        assert hasattr(jnp, "zeros")
        assert hasattr(jnp, "ones")

    def test_get_array_module_auto(self):
        """Auto-detection should return a valid array module."""
        arr_mod = get_array_module()
        assert hasattr(arr_mod, "array")

    def test_backend_enum_values(self):
        """Backend enum should have expected values."""
        assert Backend.JAX.value == "jax"
        assert Backend.MLX.value == "mlx"


class TestMLXBackend:
    """Tests specific to MLX backend (skipped if MLX not available)."""

    @pytest.mark.skipif(not is_mlx_available(), reason="MLX not available")
    def test_mlx_available(self):
        """MLX should be detected on Apple Silicon."""
        assert is_mlx_available() is True

    @pytest.mark.skipif(not is_mlx_available(), reason="MLX not available")
    def test_get_array_module_mlx(self):
        """Should be able to get MLX array module."""
        mx = get_array_module(Backend.MLX)
        assert hasattr(mx, "array")
        assert hasattr(mx, "zeros")
        assert hasattr(mx, "ones")

    @pytest.mark.skipif(not is_mlx_available(), reason="MLX not available")
    def test_mlx_preferred_on_apple_silicon(self):
        """MLX should be preferred when available and prefer_mlx=True."""
        # Reset cache for this test
        import attn_tensors.backend as backend_module

        backend_module._cached_backend = None

        backend = get_backend(prefer_mlx=True)
        # On Apple Silicon with MLX installed, should prefer MLX
        if is_mlx_available():
            assert backend == Backend.MLX

        # Reset cache
        backend_module._cached_backend = None

    @pytest.mark.skipif(not is_mlx_available(), reason="MLX not available")
    def test_jax_when_mlx_not_preferred(self):
        """JAX should be used when prefer_mlx=False."""
        import attn_tensors.backend as backend_module

        backend_module._cached_backend = None

        backend = get_backend(prefer_mlx=False)
        assert backend == Backend.JAX

        # Reset cache
        backend_module._cached_backend = None


class TestArrayOperations:
    """Test that array operations work with detected backend."""

    def test_create_array(self):
        """Should be able to create arrays with detected backend."""
        arr_mod = get_array_module()
        x = arr_mod.array([1.0, 2.0, 3.0])
        assert x.shape == (3,)

    def test_basic_ops(self):
        """Basic operations should work."""
        arr_mod = get_array_module()
        x = arr_mod.ones((3, 3))
        y = arr_mod.zeros((3, 3))
        z = x + y
        assert z.shape == (3, 3)

    def test_matmul(self):
        """Matrix multiplication should work."""
        arr_mod = get_array_module()
        x = arr_mod.ones((2, 3))
        y = arr_mod.ones((3, 4))
        z = x @ y
        assert z.shape == (2, 4)
