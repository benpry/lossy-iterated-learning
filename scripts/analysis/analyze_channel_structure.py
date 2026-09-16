"""
This file measures what a rate-limited channel does to the beliefs passed through it.

`analyze_channel_marginals.py` reports which beliefs come out of a channel. This reports the mapping
itself, which needs the whole channel matrix rather than just its marginal:

  - how often a belief survives, meaning it is received as exactly the belief that was sent
  - how many beliefs are never received at all, whatever is sent -- the ones that are "hard to talk
    about" given the rate limit
  - how many distinct prototypes the beliefs get rounded onto, and how large those basins are

Reading a channel costs far more than measuring one, so everything here is computed on a single pass
over each file and written to a table the paper's numbers can be rebuilt from without the cache,
which lives on a node's local disk and is not backed up.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

from src.channel_analysis import (
    VOCABULARY_COVERAGE,
    decode_pseudocounts,
    format_vector,
    infer_encoding_max_val,
    parse_channel_filename,
    resolve_channel_dir,
    self_transmission_probabilities,
    summarize_precision,
)

# how faithfully a belief has to come back before we call it accurately transmitted
FIDELITY_THRESHOLDS = (0.5, 0.25, 0.1)

# how many of the largest prototype basins to record for each channel
TOP_BASINS = 3


@dataclass
class Args:
    # The directory holding the cached channels. Left unset it is the cache on the cluster
    # node that computed them, which only resolves where SCR_ROOT_DIR exists.
    channel_dir: Optional[Path] = None
    # where to save the table of structure measurements
    output_file: Path = here("data/channel_structure.csv")
    # what share of the channel's output mass counts as its vocabulary
    coverage: float = VOCABULARY_COVERAGE


def describe_pseudocounts(output_index, dimension, encoding_max_val) -> str:
    """
    Render an output index as the pseudocount vector it stands for, for reading in a table
    """
    return format_vector(decode_pseudocounts(int(output_index), dimension, encoding_max_val))


def analyze_one_channel(filepath: Path, coverage: float) -> dict:
    """
    Measure the input-output structure of a single cached channel
    """
    spec = parse_channel_filename(filepath)
    with open(filepath, "rb") as f:
        channel, marginal, rate, distortion = pickle.load(f)[:4]
    channel = np.asarray(channel)
    marginal = np.asarray(marginal)

    n_inputs, n_outputs = channel.shape
    encoding_max_val = infer_encoding_max_val(n_outputs, spec.dimension)

    row = {
        "source_distribution": spec.source_distribution,
        "dimension": spec.dimension,
        "max_val": spec.max_val,
        "encoding_max_val": encoding_max_val,
        "beta": spec.beta,
        "rate": float(rate),
        "distortion": float(distortion),
        "n_inputs": int(n_inputs),
        "n_outputs": int(n_outputs),
    }

    # how much of the output space is reachable at all
    vocabulary = summarize_precision(
        marginal, spec.dimension, encoding_max_val, coverage=coverage
    )
    row["coverage"] = coverage
    row["vocabulary_size"] = vocabulary["vocabulary_size"]
    row["n_outputs_never_received"] = int(n_outputs - vocabulary["vocabulary_size"])

    # how often a belief survives the trip
    representable, fidelity = self_transmission_probabilities(
        channel, spec.dimension, spec.max_val, encoding_max_val
    )
    row["n_inputs_representable"] = int(len(representable))
    row["self_transmission_mean"] = float(fidelity.mean())
    row["self_transmission_median"] = float(np.median(fidelity))
    row["self_transmission_max"] = float(fidelity.max())
    for threshold in FIDELITY_THRESHOLDS:
        row[f"n_survive_above_{threshold:g}"] = int((fidelity > threshold).sum())

    most_faithful = representable[int(np.argmax(fidelity))]
    row["most_faithful_belief"] = describe_pseudocounts(
        most_faithful, spec.dimension, spec.max_val
    )

    # what every belief gets rounded to, whether or not it survives
    rounded_to = np.argmax(channel, axis=1)
    prototypes, basin_sizes = np.unique(rounded_to, return_counts=True)
    row["n_prototypes"] = int(len(prototypes))
    largest = np.argsort(-basin_sizes)[:TOP_BASINS]
    for rank, position in enumerate(largest, start=1):
        row[f"basin_{rank}_belief"] = describe_pseudocounts(
            prototypes[position], spec.dimension, encoding_max_val
        )
        row[f"basin_{rank}_share"] = float(basin_sizes[position] / n_inputs)

    return row


def main(args: Args):
    channel_dir = resolve_channel_dir(args.channel_dir)
    channel_files = sorted(channel_dir.glob("*.npy"))
    if not channel_files:
        raise ValueError(f"No cached channels found in {channel_dir}")

    rows = []
    for n, filepath in enumerate(channel_files, start=1):
        row = analyze_one_channel(filepath, args.coverage)
        rows.append(row)
        print(
            f"[{n}/{len(channel_files)}] {row['source_distribution']}, "
            f"max_val={row['max_val']} beta={row['beta']:<9.3f} rate={row['rate']:>7.3f}  "
            f"never received={row['n_outputs_never_received']:>5,}  "
            f"survive>0.5={row['n_survive_above_0.5']:>5,}  "
            f"prototypes={row['n_prototypes']:>5,}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(
        ["source_distribution", "max_val", "rate"]
    )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"\nSaved structure of {len(df)} channels to {args.output_file}")


if __name__ == "__main__":
    main(tyro.cli(Args))
