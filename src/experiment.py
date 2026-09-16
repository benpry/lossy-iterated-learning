import os
import pickle
from functools import partial
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.info_theory import blahut_arimoto
from src.utils import (
    get_observation_transition_matrix,
    jeff_divergence_dirichlet_from_indices,
    pad_with_zeros,
)


def get_distortion_matrix(
    max_val, dimension, metric="jeff_divergence", lifespan=1, cache=True
):
    if cache:
        cache_path = f"{os.environ['SCR_ROOT_DIR']}/cache/distortion_matrices/{metric}_distortion_matrix_max_val-{max_val}_dimension-{dimension}_lifespan-{lifespan}.npy"
    else:
        cache_path = None

    # check if there's a cached distortion matrix
    if cache and os.path.exists(cache_path):
        print("Loading cached distortion matrix...")
        with open(cache_path, "rb") as f:
            distortion_matrix = np.load(f)
    else:
        print("Computing distortion matrix...")
        # get all sets of parameters
        indices_a, indices_b = np.meshgrid(
            np.arange((max_val - lifespan) ** dimension), np.arange(max_val**dimension)
        )
        # compute the distortion matrix
        if metric == "jeff_divergence":
            vectorized_distortion_fn = jnp.vectorize(
                partial(
                    jeff_divergence_dirichlet_from_indices,
                    dimension=dimension,
                    p_max_val=max_val - lifespan,
                    q_max_val=max_val,
                )
            )
        else:
            raise ValueError(f"Metric {metric} not supported")

        distortion_matrix = vectorized_distortion_fn(indices_a, indices_b)

        # Cache the distortion matrix. Several workers starting at once will each build this, so it
        # goes to a file of its own first and is then moved into place, which is atomic. That way
        # nobody can load a matrix that is still being written.
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            partial_path = f"{cache_path}.{os.getpid()}.tmp"
            with open(partial_path, "wb") as f:
                np.save(f, distortion_matrix)
            os.replace(partial_path, cache_path)

    return distortion_matrix


class Experiment:
    """
    An iterated learning experiment
    """

    def __init__(
        self,
        true_probs,
        distortion_metric,
        distortion_matrix,
        n_generations,
        reweighting_fn=None,
        n_observations=10,
        max_param_val=100,
        convergence_tolerance=1e-8,
        max_channel_iters=100_000,
        channel_cache_dir=None,
    ):
        if not isinstance(true_probs, jnp.ndarray):
            self.true_probs = jnp.asarray(true_probs)
        else:
            self.true_probs = true_probs

        self.dimension = len(self.true_probs)
        self.max_param_val = max_param_val
        self.distortion_metric = distortion_metric
        self.distortion_matrix = distortion_matrix
        self._observation_matrix = None
        self.n_generations = n_generations
        self.n_observations = n_observations
        self.reweighting_fn = reweighting_fn
        self.convergence_tolerance = convergence_tolerance
        self.max_channel_iters = max_channel_iters
        # workers filling the cache in parallel need to say where it lives
        self.channel_cache_dir = (
            Path(channel_cache_dir)
            if channel_cache_dir is not None
            else Path(os.environ["SCR_ROOT_DIR"]) / "cache" / "channels"
        )

    @property
    def observation_matrix(self):
        """
        The matrix that updates beliefs after one observation, built the first time it is needed.

        It has one row and column per parameter setting, so it runs to gigabytes at the sizes we
        sweep over. Filling the channel cache never transmits anything, so a worker doing only that
        should not pay for it.
        """
        if self._observation_matrix is None:
            self._observation_matrix = get_observation_transition_matrix(
                self.true_probs, self.max_param_val
            )

        return self._observation_matrix

    def load_cached_channel(self, channel_filepath):
        """
        Load a cached channel, treating one that does not meet our tolerance as a cache miss.

        A cached channel is only as good as the convergence it was computed to, and the tolerance is
        not part of its filename, so the channel has to say for itself how converged it is.
        """
        if not os.path.exists(channel_filepath):
            return None

        with open(channel_filepath, "rb") as f:
            channel = pickle.load(f)

        if len(channel) < 6:
            print(
                f"Recomputing {channel_filepath}: it was cached before channels recorded how far "
                f"they were from converging"
            )
            return None

        remaining = float(channel[5])
        if remaining > self.convergence_tolerance:
            print(
                f"Recomputing {channel_filepath}: it had an estimated {remaining:.3g} of channel "
                f"movement still to come, above the tolerance of {self.convergence_tolerance:.3g}"
            )
            return None

        return channel

    def compute_channel(
        self, beta, source_distribution_fn, source_distribution_str=None
    ):
        """
        Compute a channel using the Blahut-Arimoto algorithm, given a beta and a source distribution function
        """

        # compute a source distribution
        possible_params = jnp.arange(self.max_param_val**self.dimension)
        source_distribution = jax.vmap(
            partial(
                source_distribution_fn,
                max_val=self.max_param_val,
                dimension=self.dimension,
            )
        )(possible_params)
        source_distribution = source_distribution / source_distribution.sum()

        # check if the channel is cached
        if source_distribution_str is not None:
            channel_filepath = self.channel_cache_dir / (
                f"channel_dim-{self.dimension}"
                f"_max_val-{self.max_param_val}"
                f"_beta-{jnp.round(beta, 3)}"
                f"_source-{source_distribution_str}"
                f"_distortion-{self.distortion_metric}.npy"
            )
            cached_channel = self.load_cached_channel(channel_filepath)
            if cached_channel is not None:
                return cached_channel

        # compute the channel
        channel = blahut_arimoto(
            source_distribution,
            self.distortion_matrix,
            beta,
            self.distortion_matrix.shape[1],
            max_iters=self.max_channel_iters,
            tolerance=self.convergence_tolerance,
        )

        # check the channel for NaNs
        if jnp.isnan(channel[0]).any():
            raise ValueError("Channel contains NaNs")

        # refuse a channel that ran out of iterations before it settled
        iters, remaining = int(channel[4]), float(channel[5])
        if remaining > self.convergence_tolerance:
            raise ValueError(
                f"Blahut-Arimoto stopped at beta={beta} after {iters} iterations with an estimated "
                f"{remaining:.3g} of channel movement still to come, above the tolerance of "
                f"{self.convergence_tolerance:.3g}. Raise max_channel_iters or loosen "
                f"convergence_tolerance."
            )

        # cache the channel
        if source_distribution_str is not None:
            os.makedirs(channel_filepath.parent, exist_ok=True)
            with open(channel_filepath, "wb") as f:
                pickle.dump(channel, f)

        return channel

    def reweight_pseudocounts(self, pseudocount_probs, temp):
        """
        Reweight the probabilities of pseudocounts to do selective social learning
        """
        if self.reweighting_fn is None:
            return pseudocount_probs
        else:
            reweighted_probs = pseudocount_probs * self.reweighting_fn(
                pseudocount_probs,
                self.dimension,
                self.max_param_val,
                self.true_probs,
                temp=temp,
            )
            return reweighted_probs / reweighted_probs.sum()

    def transmit_knowledge(self, channel, pseudocount_probs):
        """
        Pass the vector of belief prevalences through the channel
        """
        # multiply the channel by the belief prevalences
        new_probs = jnp.dot(channel.transpose(), pseudocount_probs)
        # add zeros corresponding to the pseudocounts we can't receive
        new_probs = pad_with_zeros(
            new_probs,
            self.max_param_val - self.n_observations,
            self.max_param_val,
            self.dimension,
        )
        return new_probs

    def run_iterated_learning(
        self,
        channel_beta,
        initial_prior_vec,
        source_distribution_fn,
        source_distribution_str,
        reweighting_temp=None,
    ):
        """
        Run an iterated learning experiment with a given set of parameters
        """
        # compute a channel using the prior
        channel, channel_marginal, rate, distortion, iters, remaining = self.compute_channel(
            beta=channel_beta,
            source_distribution_fn=source_distribution_fn,
            source_distribution_str=source_distribution_str,
        )

        # initialize the pseudocount probabilities
        pseudocount_probs = initial_prior_vec / initial_prior_vec.sum()

        rows = []
        for gen in range(self.n_generations):
            # log generation 0's pseudocount probabilities
            rows.append(
                {
                    "generation": gen,
                    "timestep": 0,
                    "pseudocount_probs": jnp.array_str(pseudocount_probs),
                }
            )

            # make observations
            for timestep in range(self.n_observations):
                pseudocount_probs = jnp.dot(self.observation_matrix, pseudocount_probs)
                rows.append(
                    {
                        "generation": gen,
                        "timestep": timestep + 1,
                        "pseudocount_probs": jnp.array_str(pseudocount_probs),
                    }
                )

            # apply selective social learning, if we're using it
            pseudocount_probs = self.reweight_pseudocounts(
                pseudocount_probs, temp=reweighting_temp
            )
            # pass the posterior through the channel
            pseudocount_probs = self.transmit_knowledge(channel, pseudocount_probs)

        df_run = pd.DataFrame(rows)
        df_run["channel_beta"] = channel_beta
        df_run["rate"] = rate
        df_run["distortion"] = distortion
        if reweighting_temp is not None:
            df_run["reweighting_temp"] = reweighting_temp

        return df_run

    def sweep(self, all_params: dict):
        """
        Run a sweep over parameter settings
        """
        all_value_combinations = list(product(*list(all_params.values())))

        all_run_dfs = []
        for values in tqdm(all_value_combinations):
            params = dict(zip(all_params.keys(), values))
            # run the experiment
            df_run = self.run_iterated_learning(**params)
            all_run_dfs.append(df_run)

        df_all = pd.concat(all_run_dfs)

        return df_all
