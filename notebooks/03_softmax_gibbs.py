# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax",
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 3. Softmax as Gibbs Distribution

        The softmax function is the **Gibbs/Boltzmann distribution** from statistical mechanics!

        ## Statistical Mechanics Connection

        In thermodynamics, a system with energy levels $E_j$ at temperature $T$ has probability:

        $$P(j) = \frac{\exp(-E_j / T)}{Z}, \quad Z = \sum_j \exp(-E_j / T)$$

        For attention:
        - **Scores = negative energies**: $S^{ij} = -E^{ij}$
        - **Higher score = lower energy = more probable**
        """
    )
    return


@app.cell
def _():
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    return jax, jnp, plt


@app.cell
def _(jnp):
    def softmax(x, temperature=1.0):
        """
        Softmax = Gibbs distribution

        P(j) = exp(x_j / T) / sum_k exp(x_k / T)
        """
        scaled = x / temperature
        # Subtract max for numerical stability
        scaled = scaled - jnp.max(scaled, axis=-1, keepdims=True)
        exp_x = jnp.exp(scaled)
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def entropy(probs, eps=1e-12):
        """Shannon entropy: H = -sum_j P_j log P_j"""
        p = jnp.clip(probs, eps, 1.0)
        return -jnp.sum(probs * jnp.log(p), axis=-1)

    def partition_function(x, temperature=1.0):
        """Partition function Z = sum_j exp(x_j / T)"""
        scaled = x / temperature
        max_x = jnp.max(scaled)
        return jnp.exp(max_x) * jnp.sum(jnp.exp(scaled - max_x))

    return entropy, partition_function, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Temperature Effects

        The temperature parameter $T$ controls how "peaked" the distribution is:

        - $T \to 0$: **Hard attention** (argmax) - all probability on highest score
        - $T = 1$: **Standard softmax**
        - $T \to \infty$: **Uniform** - equal probability on all states
        """
    )
    return


@app.cell
def _(mo):
    temp_slider = mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0, label="Temperature T")
    temp_slider
    return (temp_slider,)


@app.cell
def _(entropy, jnp, plt, softmax, temp_slider):
    # Example scores
    scores = jnp.array([2.0, 1.0, 0.5, 0.0, -0.5])

    T = temp_slider.value
    probs = softmax(scores, temperature=T)
    H = entropy(probs)
    H_max = jnp.log(len(scores))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Probability distribution
    axes[0].bar(range(len(scores)), probs, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("State j")
    axes[0].set_ylabel("Probability P(j)")
    axes[0].set_title(f"Gibbs Distribution at T = {T:.1f}")
    axes[0].set_ylim(0, 1)

    # Add score labels
    for idx, (s, p) in enumerate(zip(scores, probs)):
        axes[0].text(idx, p + 0.02, f"S={s:.1f}", ha="center", fontsize=8)

    # Right: Entropy gauge
    axes[1].barh([0], [H], color="coral", alpha=0.8, label=f"H = {H:.2f}")
    axes[1].barh([0], [H_max], color="lightgray", alpha=0.3)
    axes[1].set_xlim(0, H_max * 1.1)
    axes[1].set_xlabel("Entropy")
    axes[1].set_title(f"Entropy: {H:.2f} / {H_max:.2f} (max)")
    axes[1].set_yticks([])
    axes[1].axvline(H_max, color="red", linestyle="--", alpha=0.5, label="Max (uniform)")
    axes[1].legend()

    plt.tight_layout()
    fig
    return H, H_max, T, axes, fig, idx, p, probs, s, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Temperature Sweep

        Let's visualize how the distribution changes across temperatures:
        """
    )
    return


@app.cell
def _(entropy, jnp, plt, softmax):
    scores_sweep = jnp.array([2.0, 1.0, 0.0])
    temperatures = jnp.array([0.2, 0.5, 1.0, 2.0, 5.0])

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Distribution at each temperature
    x = jnp.arange(len(scores_sweep))
    width = 0.15
    for i, temp in enumerate(temperatures):
        probs_t = softmax(scores_sweep, temperature=temp)
        axes2[0].bar(x + i * width, probs_t, width, label=f"T={temp:.1f}", alpha=0.8)

    axes2[0].set_xlabel("State")
    axes2[0].set_ylabel("Probability")
    axes2[0].set_title("Distribution vs Temperature")
    axes2[0].set_xticks(x + width * 2)
    axes2[0].set_xticklabels([f"S={s:.0f}" for s in scores_sweep])
    axes2[0].legend()

    # Right: Entropy vs temperature
    temps_fine = jnp.linspace(0.1, 5.0, 50)
    entropies = jnp.array([entropy(softmax(scores_sweep, t)) for t in temps_fine])

    axes2[1].plot(temps_fine, entropies, "b-", linewidth=2)
    axes2[1].axhline(
        jnp.log(len(scores_sweep)),
        color="red",
        linestyle="--",
        label=f"Max = log({len(scores_sweep)}) = {jnp.log(len(scores_sweep)):.2f}",
    )
    axes2[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes2[1].set_xlabel("Temperature T")
    axes2[1].set_ylabel("Entropy H")
    axes2[1].set_title("Entropy vs Temperature")
    axes2[1].legend()

    plt.tight_layout()
    fig2
    return (
        axes2,
        entropies,
        fig2,
        i,
        probs_t,
        scores_sweep,
        temp,
        temperatures,
        temps_fine,
        width,
        x,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Free Energy

        In statistical mechanics, the **free energy** combines energy and entropy:

        $$F = \langle E \rangle - T \cdot H$$

        For attention:
        $$F^i = -T \log Z^i = -T \log \sum_j \exp(S^{ij} / T)$$

        Minimizing free energy balances:
        1. **Low energy**: Attend to high-scoring keys
        2. **High entropy**: Spread attention broadly
        """
    )
    return


@app.cell
def _(entropy, jnp, plt, softmax):
    def free_energy(scores, temperature):
        """Free energy F = -T * log Z"""
        scaled = scores / temperature
        max_s = jnp.max(scaled)
        log_Z = max_s + jnp.log(jnp.sum(jnp.exp(scaled - max_s)))
        return -temperature * log_Z

    def expected_energy(scores, temperature):
        """Expected energy <E> = -sum_j P_j * S_j (since S = -E)"""
        probs = softmax(scores, temperature)
        return -jnp.sum(probs * scores)

    # Verify: F = <E> - T*H
    test_scores = jnp.array([2.0, 1.0, 0.0])
    test_T = 1.5

    F = free_energy(test_scores, test_T)
    E_avg = expected_energy(test_scores, test_T)
    H_val = entropy(softmax(test_scores, test_T))

    print(f"Scores: {test_scores}")
    print(f"Temperature: {test_T}")
    print()
    print(f"Free energy F = -T log Z:        {F:.4f}")
    print(f"<E> - T*H = {E_avg:.4f} - {test_T}*{H_val:.4f} = {E_avg - test_T * H_val:.4f}")
    print()
    print(f"Verification: F = <E> - T*H? {jnp.isclose(F, E_avg - test_T * H_val)}")
    return E_avg, F, H_val, expected_energy, free_energy, test_T, test_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Partition Function

        The **partition function** $Z = \sum_j \exp(S_j / T)$ is fundamental:

        - Normalizes the Gibbs distribution
        - Its log gives free energy: $F = -T \log Z$
        - Derivatives give thermodynamic quantities
        """
    )
    return


@app.cell
def _(jnp, partition_function, plt):
    # Partition function vs temperature
    test_scores_z = jnp.array([2.0, 1.0, 0.0, -1.0])
    temps_z = jnp.linspace(0.1, 5.0, 50)

    Z_values = jnp.array([partition_function(test_scores_z, t) for t in temps_z])
    log_Z_values = jnp.log(Z_values)

    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

    axes3[0].plot(temps_z, Z_values, "b-", linewidth=2)
    axes3[0].set_xlabel("Temperature T")
    axes3[0].set_ylabel("Z")
    axes3[0].set_title("Partition Function Z")
    axes3[0].set_yscale("log")

    axes3[1].plot(temps_z, log_Z_values, "r-", linewidth=2)
    axes3[1].set_xlabel("Temperature T")
    axes3[1].set_ylabel("log Z")
    axes3[1].set_title("Log Partition Function")

    plt.tight_layout()
    fig3
    return Z_values, axes3, fig3, log_Z_values, temps_z, test_scores_z


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Effective Number of States

        The **perplexity** or effective number of states is:

        $$n_{\text{eff}} = \exp(H) = \exp\left(-\sum_j P_j \log P_j\right)$$

        This tells us how many states "effectively" contribute:
        - $n_{\text{eff}} = 1$: Delta distribution (one state)
        - $n_{\text{eff}} = n$: Uniform (all states equal)
        """
    )
    return


@app.cell
def _(entropy, jnp, plt, softmax):
    scores_eff = jnp.array([2.0, 1.0, 0.5, 0.0])
    temps_eff = jnp.linspace(0.1, 5.0, 50)

    n_eff = jnp.array([jnp.exp(entropy(softmax(scores_eff, t))) for t in temps_eff])

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(temps_eff, n_eff, "g-", linewidth=2)
    ax4.axhline(1, color="gray", linestyle="--", alpha=0.5, label="Min (delta)")
    ax4.axhline(
        len(scores_eff),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Max (uniform) = {len(scores_eff)}",
    )
    ax4.set_xlabel("Temperature T")
    ax4.set_ylabel("Effective Number of States")
    ax4.set_title("Perplexity = exp(Entropy)")
    ax4.legend()
    ax4.set_ylim(0, len(scores_eff) + 0.5)

    plt.tight_layout()
    fig4
    return ax4, fig4, n_eff, scores_eff, temps_eff


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        - **Softmax = Gibbs distribution** from statistical mechanics
        - **Temperature** controls sharpness: low T → peaked, high T → uniform
        - **Entropy** measures how diffuse the attention is
        - **Free energy** balances expected score vs entropy
        - **Partition function** normalizes and connects to thermodynamics

        The $1/\sqrt{d_k}$ scaling in attention can be viewed as temperature adjustment!
        """
    )
    return


if __name__ == "__main__":
    app.run()
