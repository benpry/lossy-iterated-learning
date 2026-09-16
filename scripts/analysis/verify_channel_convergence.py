"""
This file checks that every cached channel really is at the Blahut-Arimoto fixed point.

The algorithm records an estimate of how much movement was left when it stopped, but that estimate
extrapolates from the last two iterations, so it is not independent evidence. This script instead
applies the update rule to each cached channel and measures what actually happens: how far one step
moves it, how fast those steps are still shrinking, and what that implies is left to come.
"""

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

from src.channel_analysis import (
    compute_source_distribution,
    exact_beta_from_grid,
    fixed_point_residual,
    infer_encoding_max_val,
    measure_remaining_movement,
    parse_channel_filename,
)
from src.experiment import get_distortion_matrix

# How many iterations to measure the channel's travel across, twice over.
#
# This does not have to match the window the algorithm stops on. That one is amortized over millions
# of iterations on a GPU and costs nothing; this one runs in numpy on whole cached channels, where a
# single update of a sweep-sized channel takes about a second, so the window sets how long a check
# takes: 200 costs around six minutes a channel where 1000 costs half an hour.
#
# Two hundred is ample for what it has to resolve. The slowest betas contract by about one part in
# 400,000 per iteration, so across this window the two travels differ by about one part in 2,000 --
# comfortably above the error in measuring them, which is at worst the few parts per million that a
# single step carries and in practice far less, because measurement error stays fixed as the travel
# being measured grows with the window.
VERIFICATION_WINDOW = 200


@dataclass
class Args:
    channel_dir: Path = Path("/scr/benpry/cache/channels")
    # the tolerance the channels were supposed to meet
    tolerance: float = 1e-6
    # the beta grid the sweep used, needed because a cache filename only keeps three decimals of beta
    n_betas: int = 30
    max_beta: float = 10.0
    output_file: Path = here("data/channel_convergence_check.csv")
    # how many iterations to measure the channel's travel across, twice over
    window: int = VERIFICATION_WINDOW


def check_one_channel(filepath, betas, args):
    """
    Check a single cached channel against the fixed point equations
    """
    spec = parse_channel_filename(filepath)
    with open(filepath, "rb") as f:
        cached = pickle.load(f)
    channel = np.asarray(cached[0])
    recorded_remaining = float(cached[5]) if len(cached) >= 6 else float("nan")
    recorded_iters = int(cached[4])

    # the filename rounds beta to three decimals, which is nowhere near precise enough to rebuild
    # the distortion kernel this channel was computed with
    beta = exact_beta_from_grid(spec.beta, betas)

    encoding_max_val = infer_encoding_max_val(channel.shape[1], spec.dimension)
    lifespan = spec.max_val - encoding_max_val
    distortion = np.asarray(
        get_distortion_matrix(
            spec.max_val, spec.dimension, spec.distortion_metric, lifespan
        )
    )
    source_p = compute_source_distribution(spec)

    residual = fixed_point_residual(channel, source_p, distortion, beta)
    measured = measure_remaining_movement(
        channel, source_p, distortion, beta, window=args.window
    )
    measured_remaining = measured["remaining_movement"]
    ratio = measured["contraction_ratio"]

    # rows must stay normalized, and no entry may have gone negative or NaN
    row_sums = channel.sum(axis=1)
    is_valid = bool(
        np.isfinite(channel).all()
        and (channel >= 0).all()
        and np.allclose(row_sums, 1.0, atol=1e-12)
    )

    return {
        "beta": beta,
        "source_distribution": spec.source_distribution,
        "max_val": spec.max_val,
        "recorded_iters": recorded_iters,
        "recorded_remaining": recorded_remaining,
        "fixed_point_residual": residual,
        "measured_remaining": measured_remaining,
        "contraction_ratio": ratio,
        "rows_valid": is_valid,
        "meets_tolerance": bool(measured_remaining <= args.tolerance and is_valid),
    }


def main(args: Args):
    channel_files = sorted(Path(args.channel_dir).glob("*.npy"))
    if not channel_files:
        raise ValueError(f"No cached channels found in {args.channel_dir}")

    betas = np.logspace(-4, args.max_beta, num=args.n_betas, base=2.0)

    rows = []
    for filepath in channel_files:
        rows.append(check_one_channel(filepath, betas, args))
        row = rows[-1]
        print(
            f"beta={row['beta']:<9.3f} iters={row['recorded_iters']:>10,}  "
            f"residual={row['fixed_point_residual']:.3g}  rho={row['contraction_ratio']:.6f}  "
            f"measured_remaining={row['measured_remaining']:.3g}  "
            f"recorded={row['recorded_remaining']:.3g}  "
            f"{'OK' if row['meets_tolerance'] else 'FAILS TOLERANCE'}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["source_distribution", "max_val", "beta"])
    df.to_csv(args.output_file, index=False)

    n_bad = int((~df["meets_tolerance"]).sum())
    print(f"\n{len(df) - n_bad}/{len(df)} channels verified at tolerance {args.tolerance:.3g}")
    if n_bad:
        print("channels that do not meet it:")
        print(df[~df["meets_tolerance"]].to_string(index=False))
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main(tyro.cli(Args))
