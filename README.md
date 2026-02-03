# Attention as Bilinear Form

[![Tests](https://github.com/planckeon/attn-as-bilinear-form/actions/workflows/test.yml/badge.svg)](https://github.com/planckeon/attn-as-bilinear-form/actions/workflows/test.yml)
[![Deploy](https://github.com/planckeon/attn-as-bilinear-form/actions/workflows/deploy.yml/badge.svg)](https://github.com/planckeon/attn-as-bilinear-form/actions/workflows/deploy.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://planckeon.github.io/attn-as-bilinear-form/)

A physicist's guide to transformer attention through tensor calculus, bilinear forms, and statistical mechanics.

## Overview

This project recasts the attention mechanism from transformers using the language of tensor calculus and differential geometry. Instead of viewing attention as just matrix operations, we reveal its deeper mathematical structure:

- **Bilinear Forms**: Attention scores are bilinear forms with a metric tensor
- **Statistical Mechanics**: Softmax is a Gibbs distribution with temperature
- **Hopfield Networks**: Attention implements associative memory retrieval
- **Riemannian Geometry**: Gradient descent on parameter manifolds

## Documentation

📚 **[Full Documentation](https://planckeon.github.io/attn-as-bilinear-form/)**

### Theory Deep Dives

| Topic | Description |
|-------|-------------|
| [Bilinear Forms](https://planckeon.github.io/attn-as-bilinear-form/theory/bilinear/) | Metric tensors, index notation, Riemannian structure |
| [Einsum Notation](https://planckeon.github.io/attn-as-bilinear-form/theory/einsum/) | Einstein summation, tensor contractions, attention patterns |
| [Attention Mechanism](https://planckeon.github.io/attn-as-bilinear-form/theory/attention/) | Step-by-step breakdown in index notation |
| [Gradient Derivations](https://planckeon.github.io/attn-as-bilinear-form/theory/gradients/) | Full backprop derivation, softmax Jacobian |
| [Statistical Mechanics](https://planckeon.github.io/attn-as-bilinear-form/theory/statistical/) | Gibbs distribution, entropy, Hopfield networks |
| [Multi-Head Attention](https://planckeon.github.io/attn-as-bilinear-form/theory/multihead/) | Head projections, parameter gradients |
| [Positional Encodings](https://planckeon.github.io/attn-as-bilinear-form/theory/positional/) | RoPE, ALiBi, relative encodings |
| [Efficient Attention](https://planckeon.github.io/attn-as-bilinear-form/theory/efficient/) | Flash Attention, sparse patterns, linear attention |

## Installation

```bash
# Clone the repository
git clone https://github.com/planckeon/attn-as-bilinear-form
cd attn-as-bilinear-form

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# With MLX support (Apple Silicon only)
uv sync --extra mlx
# or: pip install -e ".[mlx]"

# With development dependencies
uv sync --dev
# or: pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from attn_tensors import scaled_dot_product_attention
from attn_tensors.bilinear import bilinear_form_batch, scaled_euclidean_metric

# Create sample data
Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # 2 queries
K = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # 3 keys
V = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # 3 values

# Standard attention
output = scaled_dot_product_attention(Q, K, V)

# With explicit metric tensor (bilinear form view)
g = scaled_euclidean_metric(d=2)
scores = bilinear_form_batch(Q, K, g)  # S^{ij} = Q^{ia} g_{ab} K^{jb}
```

### Statistical Mechanics View

```python
from attn_tensors.softmax import softmax_temperature, attention_entropy

# Temperature-controlled attention
scores = Q @ K.T / jnp.sqrt(2)

# Sharp attention (low temperature)
A_sharp = softmax_temperature(scores, beta=10.0)

# Soft attention (high temperature)
A_soft = softmax_temperature(scores, beta=0.1)

# Measure concentration
entropy = attention_entropy(A_sharp)  # Lower = more focused
```

### Gradient Verification

```python
from attn_tensors.gradients import attention_backward, verify_gradients

# Manual gradients match JAX autodiff
results = verify_gradients(Q, K, V)
print(results)  # {'dL_dQ': True, 'dL_dK': True, 'dL_dV': True, 'all_correct': True}
```

## Modules

| Module | Description |
|--------|-------------|
| `attn_tensors.attention` | Core attention operations (scores, weights, output) |
| `attn_tensors.bilinear` | Metric tensors and bilinear forms |
| `attn_tensors.einsum` | Einstein summation utilities and examples |
| `attn_tensors.gradients` | Manual gradient derivations verified against autodiff |
| `attn_tensors.softmax` | Softmax with temperature, entropy, Gibbs distribution |
| `attn_tensors.multihead` | Multi-head attention with head splitting |
| `attn_tensors.masking` | Causal masks, padding masks, local attention masks |
| `attn_tensors.hopfield` | Modern Hopfield network interpretation |
| `attn_tensors.backend` | JAX/MLX backend detection and selection |

## The Core Insight

### Standard Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### In Index Notation

$$O^{ib} = A^{ij} V^{jb}, \quad A^{ij} = \frac{\exp(S^{ij})}{\sum_k \exp(S^{ik})}, \quad S^{ij} = \frac{1}{\sqrt{d_k}} Q^{ia} K^{ja}$$

### As Bilinear Form

The score computation is a **bilinear form** with metric tensor:

$$S^{ij} = Q^{ia} g_{ab} K^{jb}$$

where $g_{ab} = \frac{1}{\sqrt{d_k}} \delta_{ab}$ is the scaled Euclidean metric.

### As Statistical Mechanics

Attention weights are a **Gibbs distribution**:

$$A^{ij} = \frac{e^{\beta S^{ij}}}{Z^i}, \quad Z^i = \sum_j e^{\beta S^{ij}}$$

where $\beta = 1$ is the inverse temperature.

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=attn_tensors --cov-report=term-missing

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run benchmarks
uv run pytest tests/test_benchmarks.py -v --benchmark-only
```

### Test Markers

```bash
# Skip slow tests
uv run pytest tests/ -m "not slow"

# Run only slow tests
uv run pytest tests/ -m slow

# Run only benchmark tests
uv run pytest tests/ -m benchmark
```

## Backend Support

### JAX (Default)

The library uses JAX for automatic differentiation and JIT compilation:

```python
import jax
print(jax.devices())  # Check available devices
```

### MLX (Apple Silicon)

On Apple Silicon Macs, MLX can be used as an accelerator backend:

```bash
# Install with MLX support
uv sync --extra mlx
```

```python
from attn_tensors.backend import get_backend, Backend

# Auto-detects best available backend
backend = get_backend()  # Returns Backend.MLX on Apple Silicon, Backend.JAX otherwise
```

## Project Structure

```
attn-as-bilinear-form/
├── src/attn_tensors/       # Core library
│   ├── attention.py        # Attention operations
│   ├── bilinear.py         # Metric tensors, bilinear forms
│   ├── einsum.py           # Einstein summation utilities
│   ├── gradients.py        # Manual gradient derivations
│   ├── softmax.py          # Temperature, entropy, Gibbs
│   ├── multihead.py        # Multi-head attention
│   ├── masking.py          # Attention masks
│   ├── hopfield.py         # Hopfield network view
│   └── backend.py          # JAX/MLX backend detection
├── tests/                  # Test suite (465+ tests)
├── site/                   # Documentation (Zola)
│   └── content/theory/     # Theory deep dives
├── post.md                 # Tutorial document
└── post.typ                # Typst paper source
```

## References

1. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Ramsauer et al. (2020). *Hopfield Networks is All You Need*. ICLR.
3. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*. NeurIPS.
4. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
5. Press et al. (2022). *ALiBi: Train Short, Test Long*. ICLR.
6. Amari (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation.
7. Sankalp (2024). [*Shape Rotation 101: An Intro to Einsum and Jax Transformers*](https://sankalp.bearblog.dev/einsum-new/).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{attn-bilinear,
  author = {Kataru, Baalateja},
  title = {Attention as Bilinear Form: A Physicist's Guide to Transformer Attention},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/planckeon/attn-as-bilinear-form},
  note = {Tensor calculus, statistical mechanics, and differential geometry perspectives on attention}
}
```

[Live Site](https://planckeon.github.io/attn-as-bilinear-form/)

## License

MIT License - see [LICENSE](LICENSE) for details.
