"""
Attention as Bilinear Form - A Physicist's Guide

Core library for tensor calculus operations on attention mechanisms.
"""

__version__ = "0.1.0"

# Core attention operations
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

# Backend detection
from attn_tensors.backend import (
    Backend,
    get_array_module,
    get_backend,
    is_jax_available,
    is_mlx_available,
)

# Bilinear forms and metrics
from attn_tensors.bilinear import (
    bilinear_form,
    bilinear_form_batch,
    euclidean_metric,
    learned_metric,
    scaled_euclidean_metric,
    verify_metric_properties,
)

# Masking utilities
from attn_tensors.masking import (
    apply_mask,
    causal_mask,
    local_attention_mask,
    padding_mask,
)

# Multi-head attention
from attn_tensors.multihead import (
    multihead_attention,
    multihead_attention_batched,
)

# Softmax and statistical mechanics
from attn_tensors.softmax import (
    attention_entropy,
    entropy,
    gibbs_distribution,
    partition_function,
    softmax_rows,
)

__all__ = [
    # Version
    "__version__",
    # Backend
    "Backend",
    "get_backend",
    "get_array_module",
    "is_mlx_available",
    "is_jax_available",
    # Attention
    "scaled_dot_product_attention",
    "attention_scores",
    "attention_scores_with_metric",
    "attention_weights",
    "attention_output",
    "decompose_attention",
    "batched_attention",
    "bilinear_attention",
    # Bilinear
    "bilinear_form",
    "bilinear_form_batch",
    "euclidean_metric",
    "scaled_euclidean_metric",
    "learned_metric",
    "verify_metric_properties",
    # Softmax
    "softmax_rows",
    "gibbs_distribution",
    "partition_function",
    "entropy",
    "attention_entropy",
    # Masking
    "causal_mask",
    "padding_mask",
    "local_attention_mask",
    "apply_mask",
    # Multi-head
    "multihead_attention",
    "multihead_attention_batched",
]
