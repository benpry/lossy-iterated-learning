"""
Make a gif showing how the channel evolves as the rate changes
"""

import matplotlib.pyplot as plt
import numpy as np
from pyprojroot import here
from tqdm import tqdm

from src.experiment import Experiment, get_distortion_matrix
from src.utils import index_to_params, uniform_source_density


def format_tick_labels(params):
    a1, a2 = params
    return "{}, {}".format(int(a1), int(a2))


def make_channel_plot(channel, ax=None, vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots(sharex=True, sharey=True)
    else:
        fig = ax.get_figure()
    im = ax.imshow(channel, cmap="magma_r", vmin=vmin, vmax=vmax)

    ax.set_xticks([i for i in range(size)])
    ax.set_xticklabels(
        [
            format_tick_labels(index_to_params(i, max_val=MAX_VAL, dimension=2))
            for i in range(size)
        ],
        fontsize=8,
        rotation=90,
        fontfamily="Charter",
    )
    ax.set_yticks([i for i in range(size)])
    ax.set_yticklabels(
        [
            format_tick_labels(index_to_params(i, max_val=MAX_VAL, dimension=2))
            for i in range(size)
        ],
        fontsize=8,
        fontfamily="Charter",
    )

    return fig, ax, im


BETAS = np.logspace(-1, 4, num=30, base=10)
MAX_VAL = 5
size = MAX_VAL**2

if __name__ == "__main__":
    distortion_matrix = get_distortion_matrix(
        max_val=MAX_VAL, dimension=2, lifespan=0, metric="jeff_divergence", cache=False
    )

    chosen_betas = [0, 10, 15]

    fig, axs = plt.subplots(
        layout="constrained", ncols=len(chosen_betas), nrows=1, figsize=(9, 3.5)
    )

    channels = []
    rates = []
    distortions = []
    for beta_idx in tqdm(chosen_betas):
        beta = BETAS[beta_idx]
        channel, channel_marginal, rate, distortion, iters = Experiment(
            true_probs=[1, 0],
            distortion_metric="jeff_divergence",
            distortion_matrix=distortion_matrix,
            max_param_val=MAX_VAL,
            n_generations=1,
        ).compute_channel(beta=beta, source_distribution_fn=uniform_source_density)
        channel = channel / channel.sum(axis=1)[:, np.newaxis]
        channels.append(channel)
        rates.append(rate)
        distortions.append(distortion)

    # Calculate global min/max for consistent color scaling
    all_channels = np.array(channels)
    vmin = 0
    vmax = 1

    ims = []  # Store image objects for colorbar
    for channel, rate, distortion, ax in zip(channels, rates, distortions, axs):
        fig, ax, im = make_channel_plot(channel, ax=ax, vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(
            f"Rate: {rate:.2f}, Distortion: {distortion:.2f}", fontfamily="Charter"
        )

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(ims[0], ax=axs, shrink=0.8, aspect=20)
    cbar.set_label("Channel probability", fontfamily="Charter", fontsize=12)

    fig.supylabel("Transmitted parameters", fontfamily="Charter", x=-0.017)
    fig.supxlabel("Received parameters", fontfamily="Charter", y=0.034)

    # save the figure
    fig.savefig(
        here("figures/three_channels.pdf"), bbox_inches="tight", transparent=True
    )
