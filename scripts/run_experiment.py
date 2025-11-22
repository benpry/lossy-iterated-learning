"""
This file runs iterated learning simulations with rate-limited channels.
"""

import sys

import jax.numpy as jnp
import numpy as np
from box import Box
from pyprojroot import here

from src.experiment import Experiment, get_distortion_matrix
from src.process_results import main as process_results
from src.utils import (
    accuracy_reweighting_fn,
    decreasing_source_density,
    uniform_source_density,
)


def main(cfg: Box):
    # set the print options to save full arrays to the raw data file
    jnp.set_printoptions(threshold=sys.maxsize, precision=16, suppress=True)

    # get a set of betas to sweep over
    betas = np.logspace(-4, cfg.max_beta, num=cfg.n_betas, base=2.0)

    # set up the source distribution function and reweighting function
    source_distribution_fn = (
        decreasing_source_density
        if cfg.source_distribution == "decreasing"
        else uniform_source_density
    )
    reweighting_fn = (
        accuracy_reweighting_fn
        if "reweighting_fn" in cfg and cfg.reweighting_fn == "accuracy"
        else None
    )

    # if we're doing reweighting (i.e. selective social learning), set up a set of reweighting temperatures to sweep over
    if reweighting_fn is not None:
        reweighting_temps = np.linspace(0.1, 10.0, num=cfg.n_reweighting_temps)
    else:
        reweighting_temps = [None]

    # set up a dictionary of parameters to sweep over
    param_dict = dict(
        channel_beta=betas,
        initial_prior_vec=[
            jnp.zeros(cfg.max_val**cfg.dimension, dtype=jnp.float64).at[0].set(1.0)
        ],
        source_distribution_fn=[source_distribution_fn],
        source_distribution_str=[cfg.source_distribution],
        reweighting_temp=reweighting_temps,
    )

    # compute the distortion matrix
    distortion_matrix = get_distortion_matrix(
        cfg.max_val, cfg.dimension, cfg.distortion_metric, cfg.n_observations
    )

    # check for NaNs in the distortion matrix
    if jnp.isnan(distortion_matrix).any():
        raise ValueError("NaNs in distortion matrix")

    # set up the experiment
    exp = Experiment(
        true_probs=cfg.true_probs,
        distortion_metric=cfg.distortion_metric,
        distortion_matrix=distortion_matrix,
        n_generations=cfg.n_generations,
        n_observations=cfg.n_observations,
        max_param_val=cfg.max_val,
        reweighting_fn=reweighting_fn,
    )

    # run the parameter sweep
    df_sweep = exp.sweep(all_params=param_dict)
    df_sweep.to_csv(here(f"data/raw/{cfg.save_file_name}.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Name of config file")
    args = parser.parse_args()

    # load the config file
    config = Box.from_yaml(filename=here(f"configs/{args.config}.yaml"))
    # run the simulations
    main(config)
    # post-process the simulations
    process_results(config)
