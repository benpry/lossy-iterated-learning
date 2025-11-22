"""
This file computes small distortion matrices and channels for visualization purposes.
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyprojroot import here
from visualize_distortion_matrices import make_channel_plot

from src.experiment import get_distortion_matrix
from src.info_theory import blahut_arimoto
from src.utils import (
    decreasing_source_density,
    index_to_params,
    peaked_source_density,
    uniform_source_density,
)

SOURCE_DENSITIES = {
    "uniform": uniform_source_density,
    "peaked": peaked_source_density,
    "decreasing": decreasing_source_density,
}


def format_tick_labels(params):
    beta, alpha = params.astype(int)
    return "$\\alpha={},\\beta={}$".format(alpha, beta)


def plot_matrix(matrix, title, x_lab, y_lab, dimension, max_val, normalized=False):
    size = matrix.shape[0]
    if normalized:
        matrix = matrix / jnp.sum(matrix, axis=1, keepdims=True)

    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="viridis")
    ax.set_xticks([i for i in range(size)])
    ax.set_xticklabels(
        [
            format_tick_labels(index_to_params(i, dimension=dimension, max_val=max_val))
            for i in range(size)
        ],
        fontsize=26,
        rotation=75,
        ha="right",
    )
    ax.set_yticks([i for i in range(size)])
    ax.set_yticklabels(
        [
            format_tick_labels(index_to_params(i, dimension=dimension, max_val=max_val))
            for i in range(size)
        ],
        fontsize=26,
    )

    ax.set_xlabel(x_lab, fontsize=34)
    ax.set_ylabel(y_lab, fontsize=34)
    ax.set_title(title, fontsize=34)

    return fig, ax


def main(args):
    distortion = get_distortion_matrix(
        args["max_val"], args["dimension"], lifespan=0, cache=False
    )

    num_encodings = distortion.shape[0]
    source_p = SOURCE_DENSITIES[args["source_distribution"]]
    source_p = jax.vmap(
        partial(source_p, dimension=args["dimension"], max_val=args["max_val"])
    )(jnp.arange(num_encodings))
    source_p = source_p / source_p.sum()

    channel, channel_marginal, R, D, iters = blahut_arimoto(
        source_p, distortion, args["beta"], num_encodings, max_iters=args["max_iters"]
    )

    fig, ax = plt.subplots()

    fig, ax, im = make_channel_plot(channel, ax=ax, vmin=0, vmax=1)

    ax.set_xlabel("Transmitted parameters", fontfamily="Charter")
    ax.set_ylabel("Received parameters", fontfamily="Charter")
    ax.set_title(f"Rate: {R:.2f}, Distortion: {D:.2f}", fontfamily="Charter")

    fig.savefig(
        here(f"figures/{args['source_distribution']}_beta-{args['beta']}.pdf"),
        bbox_inches="tight",
        transparent=True,
    )


if __name__ == "__main__":
    betas = [5, 6, 7, 8, 9]
    args = {
        "max_val": 5,
        "dimension": 2,
        "max_iters": 1000,
        "source_distribution": "peaked",
    }

    for beta in betas:
        args["beta"] = beta
        main(args)
