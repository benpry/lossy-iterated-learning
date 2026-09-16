"""
Create the component plots for the worked example figure in the paper: each
belief distribution in the Alice-and-Bob story and each communication channel.

For every channel rate regime (zero, medium, high), this script saves a heatmap
of the full channel and a bar chart of the distribution over beliefs Bob might
receive given Alice's posterior Beta(2, 1).
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Rectangle
from pyprojroot import here
from scipy.stats import beta as beta_distribution

from src.experiment import get_distortion_matrix
from src.info_theory import blahut_arimoto
from src.utils import index_to_params, params_to_index

# Every belief that appears in the worked example, as (alpha, beta) pairs:
# Alice's prior (1, 1) and posterior (2, 1), then the belief Bob receives and
# his posterior after observing heads for each channel rate regime:
# rate zero (2, 2) -> (3, 2), medium rate (4, 2) -> (5, 2), high rate (2, 1) -> (3, 1)
WORKED_EXAMPLE_BELIEFS = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (5, 2), (3, 1)]

BELIEF_COLOR = "#4477AA"
TRUE_P_COLOR = "#555555"


def beta_belief_to_index(alpha: int, beta: int, max_val: int) -> int:
    """
    Convert a belief Beta(alpha, beta) to its index on the pseudocount grid.

    Following the convention in the channel visualization scripts, the parameter
    vector stores the beta pseudocount in the first (low-order) position.
    """
    return int(params_to_index(jnp.array([beta, alpha]), max_val=max_val))


def belief_tick_labels(max_val: int) -> list[str]:
    """
    Short "alpha,beta" labels for every belief on the pseudocount grid, in index order
    """
    labels = []
    for index in range(max_val**2):
        beta, alpha = index_to_params(index, dimension=2, max_val=max_val).astype(int)
        labels.append(f"{alpha},{beta}")
    return labels


def plot_belief(alpha: int, beta: int, true_p: float, max_density: float):
    """
    Plot the density of a Beta(alpha, beta) belief with a dashed line at the
    true probability of heads
    """
    fig, ax = plt.subplots(figsize=(2.6, 2.0), layout="constrained")

    grid = np.linspace(0, 1, 500)
    density = beta_distribution.pdf(grid, alpha, beta)
    ax.plot(grid, density, color=BELIEF_COLOR, linewidth=2)
    ax.fill_between(grid, density, color=BELIEF_COLOR, alpha=0.2, linewidth=0)
    ax.axvline(true_p, color=TRUE_P_COLOR, linestyle="--", linewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_density)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("$p$")
    ax.set_ylabel("Density")
    ax.set_title(f"Beta({alpha}, {beta})")
    ax.spines[["top", "right"]].set_visible(False)

    return fig


def plot_channel_heatmap(channel, rate: float, max_val: int, highlight_index: int):
    """
    Plot a channel as a heatmap of transmitted beliefs by received beliefs,
    with a box around the row for the belief Alice transmits
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.0), layout="constrained")

    im = ax.imshow(channel, cmap="magma_r", vmin=0, vmax=1)

    size = max_val**2
    labels = belief_tick_labels(max_val)
    ax.set_xticks(range(size))
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticks(range(size))
    ax.set_yticklabels(labels, fontsize=7)

    ax.add_patch(
        Rectangle(
            (-0.5, highlight_index - 0.5),
            width=size,
            height=1,
            fill=False,
            edgecolor=BELIEF_COLOR,
            linewidth=1.8,
        )
    )

    ax.set_xlabel(r"Received belief ($\alpha,\beta$)")
    ax.set_ylabel(r"Transmitted belief ($\alpha,\beta$)")
    ax.set_title(f"Rate: {rate:.2f} bits")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Channel probability")

    return fig


def plot_received_beliefs(received_probs, rate: float, max_val: int):
    """
    Plot the distribution over beliefs Bob might receive given Alice's belief
    """
    fig, ax = plt.subplots(figsize=(4.5, 2.0), layout="constrained")

    size = max_val**2
    ax.bar(range(size), received_probs, color=BELIEF_COLOR)

    ax.set_xticks(range(size))
    ax.set_xticklabels(belief_tick_labels(max_val), fontsize=7, rotation=90)
    ax.set_xlim(-1, size)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Received belief ($\alpha,\beta$)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Rate: {rate:.2f} bits")
    ax.spines[["top", "right"]].set_visible(False)

    return fig


def save_component(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main(
    max_val: int = 5,
    true_p: float = 0.8,
    beta_rate_zero: float = 0.5,
    beta_rate_medium: float = 5.0,
    beta_rate_high: float = 30.0,
    output_dir: Path = here("figures/worked_example"),
) -> dict:
    """
    Create every component of the worked example figure and return a dictionary
    mapping each rate regime to its channel and rate.

    Args:
        max_val: maximum pseudocount value on the belief grid
        true_p: true probability of heads, marked with a dashed line
        beta_rate_zero: Blahut-Arimoto beta producing a rate of (nearly) zero
        beta_rate_medium: Blahut-Arimoto beta producing a medium rate
        beta_rate_high: Blahut-Arimoto beta producing a high rate
        output_dir: directory where the component PDFs are saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Charter"

    # plot each belief distribution, sharing a y-axis limit for comparability
    grid = np.linspace(0, 1, 500)
    max_density = 1.05 * max(
        beta_distribution.pdf(grid, alpha, beta).max()
        for alpha, beta in WORKED_EXAMPLE_BELIEFS
    )
    for alpha, beta in WORKED_EXAMPLE_BELIEFS:
        fig = plot_belief(alpha, beta, true_p, max_density)
        save_component(fig, output_dir / f"belief_beta_{alpha}_{beta}.pdf")

    # compute the optimal channel and plot its behavior in each rate regime
    distortion_matrix = get_distortion_matrix(
        max_val, dimension=2, lifespan=0, cache=False
    )
    num_encodings = distortion_matrix.shape[0]
    uniform_source = jnp.ones(num_encodings) / num_encodings
    alice_index = beta_belief_to_index(alpha=2, beta=1, max_val=max_val)

    channel_betas = {
        "zero": beta_rate_zero,
        "medium": beta_rate_medium,
        "high": beta_rate_high,
    }
    results = {}
    for regime, channel_beta in channel_betas.items():
        channel, _, rate, _, _, _ = blahut_arimoto(
            uniform_source,
            distortion_matrix,
            channel_beta,
            num_encodings,
            max_iters=1000,
        )
        if jnp.isnan(channel).any():
            raise ValueError(
                f"Channel for regime '{regime}' (beta={channel_beta}) contains NaNs"
            )

        # rates can come out as tiny negative numbers due to floating point error
        rate = float(rate)
        if rate < -1e-6:
            raise ValueError(f"Channel for regime '{regime}' has negative rate {rate}")
        rate = max(rate, 0.0)

        fig = plot_channel_heatmap(channel, rate, max_val, alice_index)
        save_component(fig, output_dir / f"channel_rate_{regime}.pdf")

        fig = plot_received_beliefs(channel[alice_index], rate, max_val)
        save_component(fig, output_dir / f"received_beliefs_rate_{regime}.pdf")

        results[regime] = {"channel": channel, "rate": rate}

    return results


if __name__ == "__main__":
    tyro.cli(main)
