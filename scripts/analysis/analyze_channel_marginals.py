"""
This file analyzes the highest marginal probability outputs of every channel this code has computed.

For each cached channel, it reports the outputs the channel is most likely to send, decoded into the
Dirichlet pseudocounts they represent, along with how concentrated the channel's output distribution
is overall.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import tyro
from pyprojroot import here

from src.channel_analysis import (
    MARGINAL_TOLERANCE,
    analyze_channel_directory,
    resolve_channel_dir,
)

# the columns that identify a channel, as opposed to one of its outputs
CHANNEL_COLUMNS = [
    "source_distribution",
    "dimension",
    "max_val",
    "encoding_max_val",
    "distortion_metric",
    "beta",
    "rate",
    "distortion",
    "marginal_consistency_gap",
    "marginal_entropy_bits",
    "marginal_perplexity",
    "n_outputs_90_pct_mass",
    "n_outputs_used",
]


@dataclass
class Args:
    # The directory holding the cached channels. Left unset it is the cache on the cluster
    # node that computed them, which only resolves where SCR_ROOT_DIR exists.
    channel_dir: Optional[Path] = None
    # how many of the most likely outputs to report for each channel
    top_k: int = 10
    # where to save the table of top outputs
    output_file: Path = here("data/channel_top_outputs.csv")


def print_channel_report(channel: pd.Series, df_outputs: pd.DataFrame):
    """
    Print the most likely outputs of one channel in a human-readable form
    """
    n_outputs = channel["encoding_max_val"] ** channel["dimension"]
    print(
        f"\nsource={channel['source_distribution']}, dimension={channel['dimension']}, "
        f"max_val={channel['max_val']}, beta={channel['beta']:g}"
    )
    print(
        f"  rate={channel['rate']:.3f} bits, distortion={channel['distortion']:.3f}, "
        f"marginal entropy={channel['marginal_entropy_bits']:.3f} bits "
        f"(perplexity {channel['marginal_perplexity']:.1f})"
    )
    print(
        f"  {channel['n_outputs_used']} of {n_outputs} outputs used, "
        f"{channel['n_outputs_90_pct_mass']} of them cover 90% of the probability mass"
    )
    print(
        f"  {'rank':>4}  {'index':>6}  {'pseudocounts':<18}  {'mean probs':<24}"
        f"  {'p(output)':>10}  {'cumulative':>10}"
    )
    for _, row in df_outputs.iterrows():
        print(
            f"  {row['rank']:>4}  {row['output_index']:>6}  {row['pseudocounts']:<18}"
            f"  {row['mean_probs']:<24}  {row['marginal_prob']:>10.4f}"
            f"  {row['cumulative_prob']:>10.4f}"
        )


def main(args: Args):
    df_top_outputs = analyze_channel_directory(
        resolve_channel_dir(args.channel_dir), top_k=args.top_k
    )

    for _, df_channel in df_top_outputs.groupby(CHANNEL_COLUMNS, sort=False):
        print_channel_report(df_channel.iloc[0], df_channel)

    df_top_outputs.to_csv(args.output_file, index=False)

    # report the worst disagreement between a cached marginal and the channel it came with, which is
    # nonzero only for channels cached before the two were made consistent
    largest_gap = df_top_outputs["marginal_consistency_gap"].max()
    n_channels = len(df_top_outputs.groupby(CHANNEL_COLUMNS))
    print(
        f"\nLargest marginal consistency gap across channels: {largest_gap:.3g} "
        f"(tolerance {MARGINAL_TOLERANCE:.3g})"
    )
    print(f"Saved the top outputs of {n_channels} channels to {args.output_file}")


if __name__ == "__main__":
    main(tyro.cli(Args))
