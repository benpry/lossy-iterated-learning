"""
This file contains code for analyzing which outputs a rate-limited channel actually uses.

Each channel maps a set of Dirichlet pseudocounts onto a (smaller) set of encodings. The marginal
probability of an encoding under the source distribution tells us how often the channel uses it, so
the highest-marginal-probability outputs are the "words" the channel spends its rate budget on.
"""

import os
import pickle
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.info_theory import CONVERGENCE_WINDOW, remaining_movement
from src.utils import (
    decreasing_source_density,
    index_to_params,
    peaked_source_density,
    uniform_source_density,
)

SOURCE_DENSITIES = {
    "uniform": uniform_source_density,
    "decreasing": decreasing_source_density,
    "peaked": peaked_source_density,
}

CHANNEL_FILENAME_PATTERN = re.compile(
    r"^channel"
    r"_dim-(?P<dimension>\d+)"
    r"_max_val-(?P<max_val>\d+)"
    r"_beta-(?P<beta>[0-9.eE+-]+)"
    r"_source-(?P<source_distribution>[a-zA-Z]+)"
    r"_distortion-(?P<distortion_metric>.+)"
    r"\.npy$"
)

# The largest difference we tolerate between a channel's cached marginal and the one we recompute
# from the channel itself. These now agree by construction, but channels cached before that was true
# carry the marginal from one iteration earlier, so for them a small gap is how far the marginal
# moved in the last iteration. A large gap means something is wrong either way, like a cache file
# whose name does not describe its contents.
MARGINAL_TOLERANCE = 1e-3

# marginal probabilities below this count as unused, rather than as numerical noise
MIN_USED_PROB = 1e-10

# How much of a channel's probability mass its vocabulary has to account for. Matching the 90 percent
# that summarize_marginal already reports, so the two ways of counting words agree.
VOCABULARY_COVERAGE = 0.9


@dataclass(frozen=True)
class ChannelSpec:
    """
    The settings that uniquely identify a channel
    """

    dimension: int
    max_val: int
    beta: float
    source_distribution: str
    distortion_metric: str


def parse_channel_filename(filename) -> ChannelSpec:
    """
    Recover the settings a channel was computed with from the name of its cache file
    """
    match = CHANNEL_FILENAME_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"{filename} does not look like a cached channel")

    return ChannelSpec(
        dimension=int(match["dimension"]),
        max_val=int(match["max_val"]),
        beta=float(match["beta"]),
        source_distribution=match["source_distribution"],
        distortion_metric=match["distortion_metric"],
    )


def channel_cache_path(spec: ChannelSpec, cache_dir) -> Path:
    """
    Rebuild the path Experiment.compute_channel caches a channel at.

    The rounding of beta has to match the one in src/experiment.py exactly, so this uses the same
    jnp.round call rather than Python's round.
    """
    return Path(cache_dir) / (
        f"channel_dim-{spec.dimension}"
        f"_max_val-{spec.max_val}"
        f"_beta-{jnp.round(spec.beta, 3)}"
        f"_source-{spec.source_distribution}"
        f"_distortion-{spec.distortion_metric}.npy"
    )


def resolve_channel_dir(channel_dir=None) -> Path:
    """
    Work out where the cached channels are.

    They live on whichever node computed them, under the local disk SCR_ROOT_DIR points at. That
    variable only exists on the cluster, and the analysis scripts are also run on a laptop to redraw
    figures from a saved table, where no channel needs to be read at all. So resolving the directory
    has to happen when one is actually wanted rather than while arguments are being parsed, and
    failing has to say what to do instead of raising a KeyError from inside argument parsing.
    """
    if channel_dir is not None:
        return Path(channel_dir)

    scr_root = os.environ.get("SCR_ROOT_DIR")
    if scr_root is None:
        raise ValueError(
            "No --channel-dir was given and SCR_ROOT_DIR is not set, so there is nowhere to read "
            "cached channels from. Pass --channel-dir to point at a copy of the cache, or, if the "
            "script offers it, --reuse-table to redraw from a saved table without reading channels."
        )

    return Path(scr_root) / "cache" / "channels"


def exact_beta_from_grid(rounded_beta: float, betas) -> float:
    """
    Recover the full precision beta that a cached channel's filename was rounded from.

    Channels are cached under a beta rounded to three decimals, which is far too coarse to rebuild
    the exp2(-beta * distortion) kernel the channel was computed with, so anything that re-applies
    the update rule has to get the exact value back from the grid the sweep swept over.
    """
    matches = [
        beta for beta in np.asarray(betas) if float(jnp.round(beta, 3)) == rounded_beta
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{rounded_beta} does not match any beta on the grid exactly once "
            f"(found {len(matches)} matches)"
        )

    return float(matches[0])


def infer_encoding_max_val(n_encodings: int, dimension: int) -> int:
    """
    Recover the largest pseudocount a channel can encode from its number of outputs.

    The channel's outputs are all pseudocount vectors with entries in [1, encoding_max_val], so there
    are encoding_max_val ** dimension of them. This is smaller than the source's max_val because
    agents observe more data over their lifespan than they can transmit.
    """
    encoding_max_val = round(n_encodings ** (1 / dimension))
    if encoding_max_val**dimension != n_encodings:
        raise ValueError(
            f"{n_encodings} channel outputs is not a perfect power of dimension {dimension}"
        )

    return encoding_max_val


def decode_pseudocounts(output_index: int, dimension: int, encoding_max_val: int):
    """
    Turn the index of a channel output into the vector of Dirichlet pseudocounts it stands for
    """
    params = index_to_params(output_index, dimension, max_val=encoding_max_val)
    return np.asarray(params).astype(int)


def encode_pseudocounts(pseudocounts, encoding_max_val: int) -> int:
    """
    Turn a vector of Dirichlet pseudocounts into the index of the output that stands for it.

    The inverse of `decode_pseudocounts`.
    """
    return int(
        sum(
            (int(count) - 1) * encoding_max_val**i
            for i, count in enumerate(pseudocounts)
        )
    )


def self_transmission_probabilities(
    channel: np.ndarray, dimension: int, max_val: int, encoding_max_val: int
):
    """
    How often each belief comes back as itself, for the beliefs that can come back at all.

    Inputs range over pseudocounts up to max_val while outputs only reach encoding_max_val, because
    an agent observes more over a lifetime than it can say. So this is not the diagonal of the
    channel matrix: a belief has to be looked up by its pseudocounts in both index spaces. Beliefs
    with a pseudocount larger than the outputs can express have no cell to land in and are left out.
    Returns the indices of the beliefs that remain alongside their probabilities.
    """
    indices = np.arange(max_val**dimension)
    # digit i of an index counts in units of max_val ** i and stands for that pseudocount plus one
    digits = np.stack([(indices // max_val**i) % max_val + 1 for i in range(dimension)])

    representable = (digits <= encoding_max_val).all(axis=0)
    rows = indices[representable]
    cols = sum(
        (digits[i][representable] - 1) * encoding_max_val**i for i in range(dimension)
    )

    return rows, np.asarray(channel)[rows, cols]


def format_vector(values, format_spec="") -> str:
    """
    Format a vector of numbers so it stays readable inside a csv cell
    """
    return "(" + ", ".join(f"{value:{format_spec}}" for value in values) + ")"


def compute_source_distribution(spec: ChannelSpec) -> np.ndarray:
    """
    Recompute the source distribution a channel was optimized for
    """
    if spec.source_distribution not in SOURCE_DENSITIES:
        raise ValueError(f"Unknown source distribution {spec.source_distribution}")

    density_fn = SOURCE_DENSITIES[spec.source_distribution]
    densities = jax.vmap(
        partial(density_fn, dimension=spec.dimension, max_val=spec.max_val)
    )(jnp.arange(spec.max_val**spec.dimension))

    return np.asarray(densities / densities.sum())


def marginal_consistency_gap(
    channel: np.ndarray, marginal: np.ndarray, spec: ChannelSpec
) -> float:
    """
    Measure how far the cached marginal is from the one the cached channel induces on its outputs.

    This says whether the two things in a cache file describe the same channel. It is not a measure
    of convergence: a channel cached by src/info_theory.py records that separately.
    """
    recomputed_marginal = compute_source_distribution(spec) @ channel

    return float(np.abs(recomputed_marginal - marginal).max())


def blahut_arimoto_kernel(distortion: np.ndarray, beta: float) -> np.ndarray:
    """
    The part of the Blahut-Arimoto update that is the same on every iteration.

    Subtracting each row's smallest scaled distortion cancels in the normalization and keeps the
    exponentials from underflowing to zero when beta is large.
    """
    scaled = beta * distortion

    return np.exp2(-(scaled - scaled.min(axis=1, keepdims=True)))


def blahut_arimoto_step(
    channel: np.ndarray, source_p: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """
    One Blahut-Arimoto update, in numpy so that checking a channel shares no code with computing one
    """
    updated = (source_p @ channel) * kernel

    return updated / updated.sum(axis=1, keepdims=True)


def max_row_total_variation(channel: np.ndarray, other: np.ndarray) -> float:
    """
    The largest total variation distance between corresponding rows of two channels
    """
    return float(np.abs(channel - other).sum(axis=1).max() / 2)


def fixed_point_residual(
    channel: np.ndarray, source_p: np.ndarray, distortion: np.ndarray, beta: float
) -> float:
    """
    Measure how far a channel is from the fixed point of the Blahut-Arimoto update.

    Applies one update and reports the largest total variation distance any row moves. This checks
    the channel against the equations it is supposed to satisfy, independently of the convergence
    estimate the algorithm recorded when it stopped.

    Note that this is how far the channel moves in one step, which near the betas where the codebook
    changes shape is hundreds of thousands of times smaller than how far it has left to go. Use
    `measure_remaining_movement` for the latter.
    """
    updated = blahut_arimoto_step(
        channel, source_p, blahut_arimoto_kernel(distortion, beta)
    )

    return max_row_total_variation(updated, channel)


def measure_remaining_movement(
    channel: np.ndarray,
    source_p: np.ndarray,
    distortion: np.ndarray,
    beta: float,
    window: int = CONVERGENCE_WINDOW,
) -> dict:
    """
    Measure how much further a cached channel would move if it kept iterating.

    Iterates it across two windows and extrapolates the geometric decay from how far it travelled in
    each. This repeats, from the cached channel and in numpy, the estimate the algorithm made for
    itself, so a channel that stopped early has nowhere to hide.

    It measures across windows for the same reason the algorithm does: near the betas where the
    codebook changes shape the steps shrink by around one part in a million each, which is smaller
    than the error in measuring a step, so the ratio of two consecutive steps carries no signal at
    all. Across a window the decay grows with the number of steps in it while the measurement error
    does not.
    """
    kernel = blahut_arimoto_kernel(distortion, beta)

    displacements = []
    for _ in range(2):
        window_start = channel
        for _ in range(window):
            channel = blahut_arimoto_step(channel, source_p, kernel)
        displacements.append(max_row_total_variation(channel, window_start))

    previous_displacement, displacement = displacements
    # how far a window's travel shrinks from one window to the next, as a per-iteration rate; this is
    # what decides how much of the distance left is hidden behind a step too small to see
    window_ratio = (
        displacement / previous_displacement if previous_displacement > 0 else 0.0
    )

    return {
        "remaining_movement": float(
            remaining_movement(displacement, previous_displacement)
        ),
        "window_displacement": displacement,
        "contraction_ratio": window_ratio ** (1.0 / window) if window_ratio > 0 else 0.0,
    }


def top_marginal_outputs(
    marginal: np.ndarray, dimension: int, encoding_max_val: int, top_k: int
) -> pd.DataFrame:
    """
    Find the outputs a channel sends most often, ranked by their marginal probability
    """
    ranked_indices = np.argsort(-marginal, kind="stable")[:top_k]
    cumulative_probs = np.cumsum(marginal[ranked_indices])

    rows = []
    for rank, (output_index, cumulative_prob) in enumerate(
        zip(ranked_indices, cumulative_probs), start=1
    ):
        pseudocounts = decode_pseudocounts(
            int(output_index), dimension, encoding_max_val
        )
        rows.append(
            {
                "rank": rank,
                "output_index": int(output_index),
                "pseudocounts": format_vector(pseudocounts),
                "concentration": int(pseudocounts.sum()),
                "mean_probs": format_vector(
                    pseudocounts / pseudocounts.sum(), format_spec=".3f"
                ),
                "marginal_prob": float(marginal[output_index]),
                "cumulative_prob": float(cumulative_prob),
            }
        )

    return pd.DataFrame(rows)


def summarize_marginal(marginal: np.ndarray) -> dict:
    """
    Summarize how concentrated a channel's marginal distribution over outputs is
    """
    used_probs = marginal[marginal > MIN_USED_PROB]
    entropy_bits = float(-np.sum(used_probs * np.log2(used_probs)))
    sorted_probs = np.sort(marginal)[::-1]

    return {
        "marginal_entropy_bits": entropy_bits,
        "marginal_perplexity": 2**entropy_bits,
        "n_outputs_90_pct_mass": int(np.searchsorted(np.cumsum(sorted_probs), 0.9) + 1),
        "n_outputs_used": int(len(used_probs)),
    }


def output_concentrations(dimension: int, encoding_max_val: int) -> np.ndarray:
    """
    The concentration -- the sum of the pseudocounts -- of every output a channel can send.

    A Dirichlet's concentration is how much evidence the belief it describes claims to rest on, so
    this is the precision of each word in the channel's vocabulary. Decoding indices one at a time
    is too slow to do for every output of every channel in a sweep, so this reproduces
    `decode_pseudocounts` for all of them at once: digit i of an index counts in units of
    encoding_max_val ** i and stands for a pseudocount of that digit plus one.
    """
    indices = np.arange(encoding_max_val**dimension)
    concentration = np.zeros_like(indices)
    for i in range(dimension):
        concentration += (indices // encoding_max_val**i) % encoding_max_val + 1

    return concentration


def summarize_precision(
    marginal: np.ndarray,
    dimension: int,
    encoding_max_val: int,
    coverage: float = VOCABULARY_COVERAGE,
) -> dict:
    """
    Summarize how confident the beliefs a channel sends are.

    A channel with no rate at all sends a single vague belief. As the rate rises it can afford a
    larger vocabulary, and the optimal way to spend that room is on more confident beliefs, so the
    precision of what it sends rises with the rate.

    The vocabulary is the smallest set of outputs carrying `coverage` of the channel's probability
    mass, which is the same convention `summarize_marginal` uses for n_outputs_90_pct_mass. It has
    to be defined by mass rather than by a probability floor because no one floor works across a
    sweep: at rate zero a single output carries everything, while at the highest rates the mass is
    spread so thinly over eight thousand outputs that every one of them falls below any floor worth
    quoting. A floor also lets a word sent once in ten thousand transmissions count as part of the
    vocabulary, which badly misreports how vague the channel is willing to be.
    """
    marginal = np.asarray(marginal)
    total = float(marginal.sum())
    if total <= 0.0:
        raise ValueError(
            "this marginal carries no probability at all, so it has no vocabulary to describe"
        )

    concentrations = output_concentrations(dimension, encoding_max_val)
    expected = float(np.dot(marginal, concentrations)) / total
    variance = float(np.dot(marginal, (concentrations - expected) ** 2)) / total

    # the smallest set of the most likely outputs that between them carry `coverage` of the mass
    ranked = np.argsort(-marginal, kind="stable")
    reached = np.searchsorted(np.cumsum(marginal[ranked]), coverage * total) + 1
    spoken = concentrations[ranked[: min(int(reached), len(ranked))]]

    return {
        "expected_precision": expected,
        "precision_sd": float(np.sqrt(max(variance, 0.0))),
        "min_precision": int(spoken.min()),
        "max_precision": int(spoken.max()),
        "vocabulary_size": int(len(spoken)),
    }


def analyze_channel_file(filepath, top_k: int) -> pd.DataFrame:
    """
    Analyze the most likely outputs of a single cached channel
    """
    spec = parse_channel_filename(filepath)

    # channels cached before they recorded their convergence have one fewer element
    with open(filepath, "rb") as f:
        channel, marginal, rate, distortion = pickle.load(f)[:4]

    channel = np.asarray(channel)
    marginal = np.asarray(marginal)

    consistency_gap = marginal_consistency_gap(channel, marginal, spec)
    if consistency_gap > MARGINAL_TOLERANCE:
        raise ValueError(
            f"The cached marginal for {spec} differs from the marginal its channel induces "
            f"by up to {consistency_gap:.3g}, which is more than the tolerance of "
            f"{MARGINAL_TOLERANCE:.3g}"
        )

    encoding_max_val = infer_encoding_max_val(len(marginal), spec.dimension)
    df_channel = top_marginal_outputs(
        marginal, spec.dimension, encoding_max_val, top_k
    )

    # every row describes the same channel, so label them all with its settings and summary
    df_channel.insert(0, "source_distribution", spec.source_distribution)
    df_channel.insert(1, "dimension", spec.dimension)
    df_channel.insert(2, "max_val", spec.max_val)
    df_channel.insert(3, "encoding_max_val", encoding_max_val)
    df_channel.insert(4, "beta", spec.beta)
    df_channel.insert(5, "distortion_metric", spec.distortion_metric)
    df_channel.insert(6, "rate", float(rate))
    df_channel.insert(7, "distortion", float(distortion))
    df_channel.insert(8, "marginal_consistency_gap", consistency_gap)
    for position, (column, value) in enumerate(
        summarize_marginal(marginal).items(), start=9
    ):
        df_channel.insert(position, column, value)

    return df_channel


def analyze_channel_directory(directory, top_k: int) -> pd.DataFrame:
    """
    Analyze the most likely outputs of every channel cached in a directory
    """
    channel_files = sorted(Path(directory).glob("*.npy"))
    if len(channel_files) == 0:
        raise ValueError(f"No cached channels found in {directory}")

    df_all = pd.concat(
        [analyze_channel_file(filepath, top_k) for filepath in channel_files]
    )
    df_all = df_all.sort_values(
        ["source_distribution", "dimension", "max_val", "beta", "rank"]
    )

    return df_all.reset_index(drop=True)
