"""
This file plots how the vocabulary of a rate-limited channel changes as the rate limit rises.

The channels in this project round beliefs off to a codebook of "prototype" beliefs. Three things
about that codebook are worth seeing as a function of rate:

  - expected precision: how much evidence the average belief the channel sends claims to rest on,
    which is the sum of its Dirichlet pseudocounts. A channel with no rate sends one vague belief,
    and the optimal way to spend extra rate turns out to be on more confident ones.
  - minimum precision: the vaguest belief the channel actually uses. This says whether it keeps a
    hedge available or whether every word it reaches for is a confident one.
  - vocabulary size: how many distinct beliefs account for most of what it sends.

The last two depend on where the line between "used" and "never used" is drawn, and no fixed
probability floor works across a sweep: at rate zero one output carries all the mass, while at the
highest rates it is spread so thinly over eight thousand outputs that none of them clears a floor
worth quoting. So the vocabulary is the smallest set of outputs covering a fixed share of the mass,
and the figures say which share they used.

Each channel is a few hundred megabytes and has to be read off disk, so this takes a few minutes
over a whole sweep. It writes the table it plots from, so the numbers can be reused without paying
that cost again.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotnine as p9
import tyro
from pyprojroot import here

from src.channel_analysis import (
    compute_source_distribution,
    infer_encoding_max_val,
    output_concentrations,
    parse_channel_filename,
    resolve_channel_dir,
    summarize_precision,
)


@dataclass
class Args:
    # The directory holding the cached channels. Left unset it is the cache on the cluster
    # node that computed them, which only resolves where SCR_ROOT_DIR exists.
    channel_dir: Optional[Path] = None
    # where to save the table of diagnostics, so the plots can be redrawn without rereading channels
    output_file: Path = here("data/channel_diagnostics.csv")
    # where to save the figures, one per diagnostic
    figure_dir: Path = here("figures")
    # what share of the channel's output mass its vocabulary has to account for
    coverage: float = 0.99
    # draw the matching precision of the beliefs going in, for comparison with the ones coming
    # out: their average on the expected panel, the vaguest possible one on the minimum panel
    reference_line: bool = True
    # skip reading the channels and plot an existing table instead
    reuse_table: bool = False
    # the image format the figures are saved in: pdf for the paper, png for anywhere else
    figure_format: Literal["pdf", "png"] = "pdf"
    # The size of each figure in inches. The paper lays the three panels out side by side, so
    # each one is taller than it is wide.
    figure_width: float = 4.0
    figure_height: float = 5.0
    # the base size of the text, in points
    font_size: int = 18


# Coverages the vocabulary is measured at alongside the one being plotted. Reading a channel costs
# far more than summarizing it, so measure them all on the one pass and let the reader see how much
# the answer moves with the definition.
COMPARISON_COVERAGES = (0.5, 0.9, 0.99, 0.999)

# The variant of the channel the main experiments transmit beliefs through. The sweep also caches
# channels for the other conditions, and their diagnostics still land in the table, but the figures
# only show this one.
STANDARD_SOURCE_DISTRIBUTION = "uniform"
STANDARD_MAX_VAL = 21

# what each diagnostic is called on the y axis, and whether it wants a log scale
DIAGNOSTICS = {
    "expected_precision": ("Expected confidence of received beliefs", False),
    "min_precision": ("Minimum confidence of beliefs in vocabulary", False),
    "vocabulary_size": ("Vocabulary size", True),
}

# the diagnostics that only count the words inside the vocabulary, so the figure has to name it
COVERAGE_DEPENDENT = ("min_precision", "vocabulary_size")

# The input-side counterpart each diagnostic's reference line shows. The averages the channel
# sends are compared against the average it receives, and the vaguest word it uses against the
# vaguest belief it can receive; the count of words has no input-side counterpart, so no line.
REFERENCE_COLUMNS = {
    "expected_precision": "source_precision",
    "min_precision": "source_min_precision",
}


def diagnostic_column(df: pd.DataFrame, diagnostic: str, coverage: float) -> str:
    """
    Which column of a saved table holds a diagnostic at the coverage being plotted.

    The table carries every coverage it measured, so a figure can be redrawn at any of them without
    reading a channel, which is the only way to do it off the cluster. Selecting the column by name
    rather than trusting the one the table was built with is what keeps the axis label honest: the
    failure it prevents is silent, an axis reading 0.99 above a line showing 0.9.
    """
    if diagnostic not in COVERAGE_DEPENDENT:
        return diagnostic

    # a table measured at the coverage being plotted already holds it in the main column, which is
    # the ordinary case: main() only reuses a table whose coverage matches
    if "coverage" in df.columns and (df["coverage"] == coverage).all():
        return diagnostic

    column = f"{diagnostic}_at_{coverage:g}"
    if column not in df.columns:
        available = sorted(
            name.split("_at_")[1]
            for name in df.columns
            if name.startswith(f"{diagnostic}_at_")
        )
        raise ValueError(
            f"the saved table does not hold {diagnostic} at a coverage of {coverage:g}; it "
            f"measured {', '.join(available)}. Pass --coverage with one of those, or regenerate "
            f"the table from the channels to measure another."
        )

    return column


def channel_label(spec) -> str:
    """
    A short name for the family of channels a spec belongs to, recorded in the table
    """
    return f"{spec.source_distribution}, max_val={spec.max_val}"


def select_standard_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the diagnostics of the channel variant the main experiments use
    """
    standard = df[
        (df["source_distribution"] == STANDARD_SOURCE_DISTRIBUTION)
        & (df["max_val"] == STANDARD_MAX_VAL)
    ]
    if standard.empty:
        raise ValueError(
            f"No channels with source_distribution={STANDARD_SOURCE_DISTRIBUTION!r} and "
            f"max_val={STANDARD_MAX_VAL} to plot; the table only has "
            f"{sorted(df['channel'].unique())}"
        )
    return standard


def diagnose_one_channel(filepath: Path, coverage: float) -> dict:
    """
    Measure the vocabulary of a single cached channel
    """
    spec = parse_channel_filename(filepath)
    with open(filepath, "rb") as f:
        _, marginal, rate, distortion = pickle.load(f)[:4]
    marginal = np.asarray(marginal)

    encoding_max_val = infer_encoding_max_val(len(marginal), spec.dimension)
    diagnostics = summarize_precision(
        marginal, spec.dimension, encoding_max_val, coverage=coverage
    )

    # the same measures at other coverages, so how much the answer depends on where the line is
    # drawn is visible in the table rather than having to be taken on trust
    for other in COMPARISON_COVERAGES:
        at_coverage = summarize_precision(
            marginal, spec.dimension, encoding_max_val, coverage=other
        )
        diagnostics[f"min_precision_at_{other:g}"] = at_coverage["min_precision"]
        diagnostics[f"vocabulary_size_at_{other:g}"] = at_coverage["vocabulary_size"]

    # the same measures applied to the beliefs going in, so the plots can say whether the channel
    # sends beliefs more or less confident than the ones it is given. The source runs over max_val
    # values per parameter where the outputs only run over encoding_max_val of them.
    source_p = np.asarray(compute_source_distribution(spec))
    input_concentrations = np.asarray(
        output_concentrations(spec.dimension, spec.max_val)
    )
    source_precision = float(np.dot(source_p, input_concentrations))
    source_min_precision = float(np.min(input_concentrations[source_p > 0]))

    return {
        "source_distribution": spec.source_distribution,
        "dimension": spec.dimension,
        "max_val": spec.max_val,
        "encoding_max_val": encoding_max_val,
        "beta": spec.beta,
        "rate": float(rate),
        "distortion": float(distortion),
        "coverage": coverage,
        "source_precision": source_precision,
        "source_min_precision": source_min_precision,
        "channel": channel_label(spec),
        **diagnostics,
    }


def collect_diagnostics(args: Args) -> pd.DataFrame:
    """
    Measure every cached channel
    """
    channel_dir = resolve_channel_dir(args.channel_dir)
    channel_files = sorted(channel_dir.glob("*.npy"))
    if not channel_files:
        raise ValueError(f"No cached channels found in {channel_dir}")

    rows = []
    for n, filepath in enumerate(channel_files, start=1):
        row = diagnose_one_channel(filepath, args.coverage)
        rows.append(row)
        print(
            f"[{n}/{len(channel_files)}] {row['channel']} beta={row['beta']:<9.3f} "
            f"rate={row['rate']:>7.3f}  vocabulary={row['vocabulary_size']:>5,}  "
            f"expected precision={row['expected_precision']:>6.2f}  "
            f"vaguest word={row['min_precision']:>3}",
            flush=True,
        )

    return pd.DataFrame(rows).sort_values(["channel", "rate"])


def reference_value(df: pd.DataFrame, diagnostic: str) -> Optional[float]:
    """
    The input-side value a diagnostic's reference line sits at, or None if it has no line
    """
    column = REFERENCE_COLUMNS.get(diagnostic)
    if column is None:
        return None
    if column not in df.columns:
        raise ValueError(
            f"the saved table has no {column} column, so it predates this reference line; "
            f"regenerate the table from the channels to draw it, or turn the line off"
        )
    return float(df[column].iloc[0])


def make_diagnostic_plot(df: pd.DataFrame, diagnostic: str, args: Args):
    """
    Plot one diagnostic against the channel's rate
    """
    label, log_scale = DIAGNOSTICS[diagnostic]

    column = diagnostic_column(df, diagnostic, args.coverage)
    p = (
        p9.ggplot(df, mapping=p9.aes(x="rate", y=column))
        + p9.geom_line()
        + p9.geom_point(size=1.5)
        + p9.labs(x="Channel rate", y=label)
        + p9.theme_tufte(base_size=args.font_size)
        + p9.theme(
            axis_title_x=p9.element_text(family="Charter"),
            # the y axis titles are the longest labels, so they get two points less than the rest
            axis_title_y=p9.element_text(family="Charter", size=args.font_size - 2),
        )
    )

    if args.reference_line:
        reference = reference_value(df, diagnostic)
        if reference is not None:
            p = p + p9.geom_hline(yintercept=reference, linetype="dashed")

    if log_scale:
        p = p + p9.scale_y_log10()
    else:
        # The precision panels have a meaningful zero, and starting the axis there keeps the size
        # of the changes across rates honest. A log axis has no zero to start from.
        p = p + p9.expand_limits(y=0)

    return p


def main(args: Args):
    if args.reuse_table:
        df = pd.read_csv(args.output_file)
        # the main diagnostic columns were measured at the coverage the table was built with,
        # so plotting them under a different coverage would mislabel the figures
        if not (df["coverage"] == args.coverage).all():
            table_coverages = ", ".join(
                f"{c:g}" for c in sorted(df["coverage"].unique())
            )
            raise ValueError(
                f"The saved table measured vocabularies at coverage {table_coverages}, not "
                f"{args.coverage:g}; rerun without --reuse_table to remeasure the channels "
                f"at the new coverage"
            )
    else:
        df = collect_diagnostics(args)
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        df.to_csv(args.output_file, index=False)
        print(f"\nSaved diagnostics for {len(df)} channels to {args.output_file}")

    df = select_standard_channels(df)

    os.makedirs(args.figure_dir, exist_ok=True)
    for diagnostic in DIAGNOSTICS:
        figure_path = (
            Path(args.figure_dir) / f"channel-{diagnostic}_by_rate.{args.figure_format}"
        )
        make_diagnostic_plot(df, diagnostic, args).save(
            figure_path,
            width=args.figure_width,
            height=args.figure_height,
            verbose=False,
        )
        print(f"Saved {figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
