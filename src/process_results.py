"""
This file takes the results from the experiments and processes them into a format that can be used for plotting.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from box import Box
from pyprojroot import here

from src.utils import expected_logp_true_probs_from_string, index_to_params


def compute_proportion_max_hit(pseudocount_array_str, dimension, max_val):
    """
    Compute the proportion of the maximum hit
    """
    pseudocount_array = jnp.fromstring(pseudocount_array_str[1:-1], sep=" ")
    param_array = jax.vmap(
        partial(index_to_params, dimension=dimension, max_val=max_val)
    )(jnp.arange(len(pseudocount_array)))

    max_hit = jax.vmap(lambda x: jnp.max(x) == max_val)(param_array)

    return jnp.dot(pseudocount_array, max_hit)


def main(config: Box):
    df = pd.read_csv(here(f"data/raw/{config.save_file_name}.csv"))
    df["expected_true_probs_surprisal"] = df["pseudocount_probs"].apply(
        partial(
            expected_logp_true_probs_from_string,
            true_probs=config.true_probs,
            dimension=config.dimension,
            max_val=config.max_val,
        )
    )
    df["prop_max_hit"] = df["pseudocount_probs"].apply(
        partial(
            compute_proportion_max_hit,
            dimension=config.dimension,
            max_val=config.max_val,
        )
    )

    df = df.drop(columns=["pseudocount_probs"])
    df.to_csv(here(f"data/{config.save_file_name}-processed.csv"), index=False)


if __name__ == "__main__":
    # config = Box.from_yaml(filename=here("configs/beta_bernoulli_replication.yaml"))
    config = Box.from_yaml(filename=here("configs/beta_bernoulli_expanded.yaml"))
    main(config)
