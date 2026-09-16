"""
Tests for computing and caching the channels an experiment uses
"""

import os
import pickle

import jax.numpy as jnp
import pytest

from src.channel_analysis import ChannelSpec, channel_cache_path
from src.experiment import Experiment, get_distortion_matrix
from src.utils import uniform_source_density

TEST_MAX_VAL = 4
TEST_DIMENSION = 2
TEST_LIFESPAN = 1
TEST_BETA = 2.0


def make_experiment(tmp_path, monkeypatch, **kwargs):
    """
    Build a small experiment whose channel cache lives in a temporary directory
    """
    monkeypatch.setenv("SCR_ROOT_DIR", str(tmp_path))
    distortion_matrix = get_distortion_matrix(
        TEST_MAX_VAL, TEST_DIMENSION, "jeff_divergence", TEST_LIFESPAN, cache=False
    )
    return Experiment(
        true_probs=[0.7, 0.3],
        distortion_metric="jeff_divergence",
        distortion_matrix=distortion_matrix,
        n_generations=1,
        n_observations=TEST_LIFESPAN,
        max_param_val=TEST_MAX_VAL,
        **kwargs,
    )


def write_cached_channel(tmp_path, channel, remaining):
    """
    Cache a made-up channel, so a test can tell whether it was used or recomputed.

    Passing no remaining movement writes a channel from before the convergence estimate existed.
    """
    cache_path = channel_cache_path(
        ChannelSpec(
            dimension=TEST_DIMENSION,
            max_val=TEST_MAX_VAL,
            beta=TEST_BETA,
            source_distribution="uniform",
            distortion_metric="jeff_divergence",
        ),
        tmp_path / "cache" / "channels",
    )
    os.makedirs(cache_path.parent, exist_ok=True)

    cached = (channel, jnp.zeros(channel.shape[1]), 0.0, 0.0, 1000)
    if remaining is not None:
        cached = cached + (remaining,)

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)

    return cache_path


def recognizable_channel(experiment):
    """
    A channel that Blahut-Arimoto would never return, so it can be spotted in a result
    """
    return jnp.zeros_like(experiment.distortion_matrix).at[:, 0].set(1.0)


def test_compute_channel_rejects_a_channel_that_did_not_converge(tmp_path, monkeypatch):
    """
    A channel that ran out of iterations is an error rather than a result
    """
    experiment = make_experiment(
        tmp_path, monkeypatch, max_channel_iters=2, convergence_tolerance=1e-10
    )

    with pytest.raises(ValueError, match="movement still to come"):
        experiment.compute_channel(
            beta=TEST_BETA, source_distribution_fn=uniform_source_density
        )


def test_compute_channel_caches_a_converged_channel(tmp_path, monkeypatch):
    """
    A channel that converged is written to the cache along with its remaining movement
    """
    experiment = make_experiment(tmp_path, monkeypatch)

    channel, _, _, _, iters, remaining = experiment.compute_channel(
        beta=TEST_BETA,
        source_distribution_fn=uniform_source_density,
        source_distribution_str="uniform",
    )

    assert remaining < experiment.convergence_tolerance
    cache_path = channel_cache_path(
        ChannelSpec(
            TEST_DIMENSION, TEST_MAX_VAL, TEST_BETA, "uniform", "jeff_divergence"
        ),
        tmp_path / "cache" / "channels",
    )
    assert cache_path.exists()
    with open(cache_path, "rb") as f:
        assert len(pickle.load(f)) == 6


def test_compute_channel_uses_a_converged_cached_channel(tmp_path, monkeypatch):
    """
    A cached channel that meets the tolerance is used instead of being recomputed
    """
    experiment = make_experiment(tmp_path, monkeypatch)
    cached_channel = recognizable_channel(experiment)
    write_cached_channel(tmp_path, cached_channel, remaining=0.0)

    channel, *_ = experiment.compute_channel(
        beta=TEST_BETA,
        source_distribution_fn=uniform_source_density,
        source_distribution_str="uniform",
    )

    assert jnp.array_equal(channel, cached_channel)


def test_compute_channel_ignores_a_cached_channel_that_did_not_converge(
    tmp_path, monkeypatch
):
    """
    A cached channel that never converged is recomputed rather than reused
    """
    experiment = make_experiment(tmp_path, monkeypatch)
    cached_channel = recognizable_channel(experiment)
    write_cached_channel(tmp_path, cached_channel, remaining=1.0)

    channel, *_ = experiment.compute_channel(
        beta=TEST_BETA,
        source_distribution_fn=uniform_source_density,
        source_distribution_str="uniform",
    )

    assert not jnp.array_equal(channel, cached_channel)


def test_compute_channel_ignores_a_cached_channel_from_before_the_tolerance_existed(
    tmp_path, monkeypatch
):
    """
    Channels cached before convergence was recorded are recomputed, not trusted
    """
    experiment = make_experiment(tmp_path, monkeypatch)
    cached_channel = recognizable_channel(experiment)
    write_cached_channel(tmp_path, cached_channel, remaining=None)

    channel, *_ = experiment.compute_channel(
        beta=TEST_BETA,
        source_distribution_fn=uniform_source_density,
        source_distribution_str="uniform",
    )

    assert not jnp.array_equal(channel, cached_channel)


def test_computing_a_channel_does_not_build_the_observation_matrix(tmp_path, monkeypatch):
    """
    Computing a channel must not pay for the observation matrix.

    That matrix is several gigabytes at the sizes we run at, and a worker that only fills the channel
    cache never transmits anything, so it should never be built.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the observation matrix should not be built to compute a channel")

    # patched before the experiment exists, so construction is covered too
    monkeypatch.setattr("src.experiment.get_observation_transition_matrix", explode)
    experiment = make_experiment(tmp_path, monkeypatch)

    experiment.compute_channel(
        beta=TEST_BETA, source_distribution_fn=uniform_source_density
    )


def test_observation_matrix_is_built_when_it_is_actually_needed(tmp_path, monkeypatch):
    """
    An experiment that transmits still gets its observation matrix
    """
    experiment = make_experiment(tmp_path, monkeypatch)

    size = TEST_MAX_VAL**TEST_DIMENSION
    assert experiment.observation_matrix.shape == (size, size)
    # asking twice gives the same object rather than rebuilding it
    assert experiment.observation_matrix is experiment.observation_matrix


def test_compute_channel_caches_where_it_is_told(tmp_path, monkeypatch):
    """
    An experiment caches into the directory it was given, not one derived from the environment.

    Workers filling the cache in parallel have to be able to point at a directory of their choosing,
    and a worker that quietly wrote somewhere else would recompute channels it already had.
    """
    elsewhere = tmp_path / "chosen_cache"
    monkeypatch.setenv("SCR_ROOT_DIR", str(tmp_path / "unused"))
    distortion_matrix = get_distortion_matrix(
        TEST_MAX_VAL, TEST_DIMENSION, "jeff_divergence", TEST_LIFESPAN, cache=False
    )
    experiment = Experiment(
        true_probs=[0.7, 0.3],
        distortion_metric="jeff_divergence",
        distortion_matrix=distortion_matrix,
        n_generations=1,
        n_observations=TEST_LIFESPAN,
        max_param_val=TEST_MAX_VAL,
        channel_cache_dir=elsewhere,
    )

    experiment.compute_channel(
        beta=TEST_BETA,
        source_distribution_fn=uniform_source_density,
        source_distribution_str="uniform",
    )

    cached = list(elsewhere.glob("*.npy"))
    assert len(cached) == 1
    assert not (tmp_path / "unused").exists()
    # and the name is the one the analysis code expects to find
    assert cached[0] == channel_cache_path(
        ChannelSpec(TEST_DIMENSION, TEST_MAX_VAL, TEST_BETA, "uniform", "jeff_divergence"),
        elsewhere,
    )
