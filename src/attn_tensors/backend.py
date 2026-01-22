"""
Backend abstraction for JAX and MLX.

Automatically detects and uses the best available backend:
- MLX on Apple Silicon (if installed)
- JAX otherwise (default)
"""

from enum import Enum
from typing import Any

__all__ = ["Backend", "get_backend", "get_array_module"]


class Backend(Enum):
    """Available compute backends."""

    JAX = "jax"
    MLX = "mlx"


_cached_backend: Backend | None = None


def _detect_mlx() -> bool:
    """Check if MLX is available and functional."""
    try:
        import mlx.core as mx

        # Verify MLX is actually usable (Apple Silicon check)
        _ = mx.array([1.0])
        return True
    except (ImportError, RuntimeError):
        return False


def _detect_jax() -> bool:
    """Check if JAX is available."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend(prefer_mlx: bool = True) -> Backend:
    """
    Get the best available backend.

    Args:
        prefer_mlx: If True, prefer MLX over JAX when both are available.
                   Default True since MLX is faster on Apple Silicon.

    Returns:
        Backend enum indicating which backend to use.

    Raises:
        RuntimeError: If no backend is available.
    """
    global _cached_backend

    if _cached_backend is not None:
        return _cached_backend

    has_mlx = _detect_mlx()
    has_jax = _detect_jax()

    if prefer_mlx and has_mlx:
        _cached_backend = Backend.MLX
    elif has_jax:
        _cached_backend = Backend.JAX
    elif has_mlx:
        _cached_backend = Backend.MLX
    else:
        raise RuntimeError(
            "No backend available. Install JAX or MLX:\n"
            "  pip install jax jaxlib\n"
            "  pip install mlx  # Apple Silicon only"
        )

    return _cached_backend


def get_array_module(backend: Backend | None = None) -> Any:
    """
    Get the array module for the specified backend.

    Args:
        backend: Backend to use. If None, auto-detects.

    Returns:
        The array module (jax.numpy or mlx.core).
    """
    if backend is None:
        backend = get_backend()

    if backend == Backend.MLX:
        import mlx.core as mx

        return mx
    else:
        import jax.numpy as jnp

        return jnp


def is_mlx_available() -> bool:
    """Check if MLX is available."""
    return _detect_mlx()


def is_jax_available() -> bool:
    """Check if JAX is available."""
    return _detect_jax()
