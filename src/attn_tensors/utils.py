"""
Visualization utilities for attention analysis.

Provides functions for:
- Attention weight heatmaps
- Entropy visualizations
- Gradient flow analysis
- Comparative plots
"""

from typing import Sequence

import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure, SubFigure

# =============================================================================
# Attention Visualization
# =============================================================================


def plot_attention_weights(
    A: Array,
    query_labels: Sequence[str] | None = None,
    key_labels: Sequence[str] | None = None,
    title: str = "Attention Weights",
    cmap: str = "Blues",
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
) -> Figure | SubFigure:
    """
    Plot attention weights as a heatmap.

    Args:
        A: Attention weights of shape (n_q, n_k)
        query_labels: Labels for query positions
        key_labels: Labels for key positions
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        ax: Existing axes (optional)

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(A, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Attention Weight")

    # Labels
    ax.set_xlabel("Key Position (j)")
    ax.set_ylabel("Query Position (i)")
    ax.set_title(title)

    if query_labels is not None:
        ax.set_yticks(range(len(query_labels)))
        ax.set_yticklabels(query_labels)

    if key_labels is not None:
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(key_labels, rotation=45, ha="right")

    # Add value annotations
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            val = float(A[i, j])
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    return fig


def plot_multihead_attention(
    A: Array,
    head_names: Sequence[str] | None = None,
    figsize: tuple[int, int] | None = None,
) -> Figure | SubFigure:
    """
    Plot attention weights for all heads.

    Args:
        A: Attention weights of shape (H, n_q, n_k)
        head_names: Names for each head
        figsize: Figure size (default based on number of heads)

    Returns:
        Matplotlib figure
    """
    H = A.shape[0]

    if figsize is None:
        figsize = (4 * H, 4)

    fig, axes = plt.subplots(1, H, figsize=figsize)
    if H == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        im = ax.imshow(A[h], cmap="Blues", aspect="auto", vmin=0, vmax=1)

        name = head_names[h] if head_names else f"Head {h}"
        ax.set_title(name)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    plt.colorbar(im, ax=axes, label="Weight", shrink=0.6)
    plt.tight_layout()
    return fig


# =============================================================================
# Entropy Visualization
# =============================================================================


def plot_entropy_distribution(
    A: Array,
    title: str = "Attention Entropy Distribution",
    figsize: tuple[int, int] = (8, 5),
) -> Figure | SubFigure:
    """
    Plot distribution of attention entropy across queries.

    Args:
        A: Attention weights of shape (n_q, n_k) or (H, n_q, n_k)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from .softmax import entropy, max_entropy

    fig, ax = plt.subplots(figsize=figsize)

    if A.ndim == 2:
        H_vals = entropy(A)
        ax.hist(H_vals, bins=20, alpha=0.7, label="Query entropy")
    else:
        # Multi-head
        for h in range(A.shape[0]):
            H_vals = entropy(A[h])
            ax.hist(H_vals, bins=20, alpha=0.5, label=f"Head {h}")

    # Add reference lines
    n_k = A.shape[-1]
    ax.axvline(max_entropy(n_k), color="red", linestyle="--", label=f"Max entropy (log {n_k})")
    ax.axvline(0, color="gray", linestyle="--", label="Min entropy (delta)")

    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Temperature Effects
# =============================================================================


def plot_temperature_sweep(
    scores: Array,
    temperatures: Array | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | SubFigure:
    """
    Visualize how attention weights change with temperature.

    Args:
        scores: Raw attention scores of shape (n,)
        temperatures: Array of temperatures to visualize
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from .softmax import entropy, temperature_sweep

    if temperatures is None:
        temperatures = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])

    probs = temperature_sweep(scores, temperatures)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Probability distributions
    x = jnp.arange(len(scores))
    for i, T in enumerate(temperatures):
        axes[0].plot(x, probs[i], "o-", label=f"T={float(T):.1f}")

    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Softmax at Different Temperatures")
    axes[0].legend()

    # Right: Entropy vs temperature
    temps_fine = jnp.linspace(0.1, 5.0, 50)
    probs_fine = temperature_sweep(scores, temps_fine)
    entropies = jnp.array([entropy(p) for p in probs_fine])

    axes[1].plot(temps_fine, entropies)
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Entropy vs Temperature")
    axes[1].axhline(jnp.log(len(scores)), color="red", linestyle="--", label="Max (uniform)")
    axes[1].legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Gradient Visualization
# =============================================================================


def plot_gradient_flow(
    flow_data: dict[str, Array],
    figsize: tuple[int, int] = (12, 4),
) -> Figure | SubFigure:
    """
    Visualize gradient flow through attention.

    Args:
        flow_data: Output from gradients.gradient_flow_analysis()
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot dL/dQ
    im0 = axes[0].imshow(flow_data["dL_dQ"], cmap="RdBu", aspect="auto")
    axes[0].set_title("dL/dQ")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Query")
    plt.colorbar(im0, ax=axes[0])

    # Plot dL/dK
    im1 = axes[1].imshow(flow_data["dL_dK"], cmap="RdBu", aspect="auto")
    axes[1].set_title("dL/dK")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Key")
    plt.colorbar(im1, ax=axes[1])

    # Plot dL/dV
    im2 = axes[2].imshow(flow_data["dL_dV"], cmap="RdBu", aspect="auto")
    axes[2].set_title("dL/dV")
    axes[2].set_xlabel("Feature")
    axes[2].set_ylabel("Value")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Gradient Flow Through Attention")
    plt.tight_layout()
    return fig


# =============================================================================
# Mask Visualization
# =============================================================================


def plot_mask(
    mask: Array,
    title: str = "Attention Mask",
    figsize: tuple[int, int] = (6, 6),
) -> Figure | SubFigure:
    """
    Visualize an attention mask.

    Args:
        mask: Boolean mask of shape (n_q, n_k)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use custom colormap: white for True (attend), black for False (mask)
    cmap = mcolors.ListedColormap(["black", "white"])

    ax.imshow(mask, cmap=cmap, aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)

    plt.tight_layout()
    return fig


# =============================================================================
# Hopfield Visualization
# =============================================================================


def plot_hopfield_energy(
    patterns: Array,
    state_trajectory: list[Array] | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> Figure | SubFigure:
    """
    Visualize Hopfield network energy landscape (2D projection).

    Args:
        patterns: Stored patterns of shape (M, d)
        state_trajectory: Optional list of states during retrieval
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from .hopfield import modern_hopfield_energy

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Create 2D grid
    x = jnp.linspace(-2, 2, 50)
    y = jnp.linspace(-2, 2, 50)
    X, Y = jnp.meshgrid(x, y)

    # For 2D patterns
    if patterns.shape[1] == 2:
        Z = jnp.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                state = jnp.array([X[i, j], Y[i, j]])
                Z = Z.at[i, j].set(modern_hopfield_energy(state, patterns))

        # Energy surface
        axes[0].contour(X, Y, Z, levels=20)
        axes[0].scatter(
            patterns[:, 0], patterns[:, 1], c="red", s=100, marker="*", label="Patterns"
        )

        if state_trajectory:
            traj = jnp.array(state_trajectory)
            axes[0].plot(traj[:, 0], traj[:, 1], "b.-", label="Retrieval")

        axes[0].set_xlabel("x_1")
        axes[0].set_ylabel("x_2")
        axes[0].set_title("Energy Landscape")
        axes[0].legend()

    # Energy along retrieval
    if state_trajectory:
        energies = [modern_hopfield_energy(s, patterns) for s in state_trajectory]
        axes[1].plot(energies, "o-")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Energy")
        axes[1].set_title("Energy During Retrieval")

    plt.tight_layout()
    return fig


# =============================================================================
# Comparison Plots
# =============================================================================


def compare_metrics(
    Q: Array,
    K: Array,
    metrics: dict[str, Array],
    figsize: tuple[int, int] | None = None,
) -> Figure | SubFigure:
    """
    Compare attention scores under different metrics.

    Args:
        Q: Queries of shape (n_q, d)
        K: Keys of shape (n_k, d)
        metrics: Dictionary of metric tensors
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from .bilinear import bilinear_form_batch
    from .softmax import softmax_rows

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (4 * n_metrics, 4)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, g) in zip(axes, metrics.items()):
        S = bilinear_form_batch(Q, K, g)
        A = softmax_rows(S)

        im = ax.imshow(A, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{name}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    plt.colorbar(im, ax=axes, label="Weight", shrink=0.6)
    plt.suptitle("Attention Under Different Metrics")
    plt.tight_layout()
    return fig
