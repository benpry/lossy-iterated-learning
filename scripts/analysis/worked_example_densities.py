"""
Compute the numbers for the paper's worked example of one-dimensional knowledge
transmission. Alice observes one heads and holds the posterior Beta(2, 1); depending on
the channel rate, Bob receives a different belief, observes one more heads, and ends up
with a different Beta posterior. This script reports the density (and base-2 log density) of
the true probability (0.9 in the paper) under Bob's posterior at each channel rate.

Note that these are densities, not probabilities: under a continuous Beta posterior, any
exact value of p has probability zero.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tyro
from scipy.stats import beta

# Bob's posterior after his own coin flip at each channel rate in the worked example:
# rate zero -> received Beta(2, 2), intermediate -> received Beta(4, 2) (overconfident),
# high -> received Alice's true posterior Beta(2, 1). Each then adds one observed heads.
BOB_POSTERIORS = [
    ("zero", 3, 2),
    ("intermediate", 5, 2),
    ("high", 3, 1),
]


@dataclass
class Args:
    true_p: float = 0.9
    """The ground-truth probability that the biased coin lands heads."""


def bob_posterior_densities(true_p: float) -> pd.DataFrame:
    """The density of the true probability under Bob's posterior at each channel rate."""
    return pd.DataFrame(
        [
            {
                "channel_rate": channel_rate,
                "posterior": f"Beta({a}, {b})",
                "density": beta.pdf(true_p, a, b),
                "log2_density": beta.logpdf(true_p, a, b) / np.log(2),
            }
            for channel_rate, a, b in BOB_POSTERIORS
        ]
    )


def main(args: Args) -> None:
    densities = bob_posterior_densities(args.true_p)
    print(f"Density of true p = {args.true_p} under Bob's posterior at each channel rate:")
    print(densities.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main(tyro.cli(Args))
