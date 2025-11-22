"""
Tests for utility functions
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats.dirichlet as jdirichlet

from src.utils import (
    accuracy_reweighting_fn,
    get_observation_transition_matrix,
    kl_divergence_dirichlet,
    params_to_index,
)


def kl_params_helper(p_params, q_params):
    analytic_kl_divergence = kl_divergence_dirichlet(p_params, q_params)

    # sample from the Dirichlet distributions
    samples = jrandom.dirichlet(jrandom.key(0), p_params, 1_000_000)
    monte_carlo_kl_divergence = jnp.mean(
        jax.vmap(lambda x: jdirichlet.logpdf(x, p_params))(samples)
        - jax.vmap(lambda x: jdirichlet.logpdf(x, q_params))(samples)
    ) / jnp.log(2)  # convert to bits
    assert jnp.isclose(analytic_kl_divergence, monte_carlo_kl_divergence, atol=1e-2)


def test_kl_divergence_dirichlet():
    """
    Compute a monte carlo estimate of the KL divergence between two Dirichlet distributions
    """
    # generate random Dirichlet parameters
    p_params, q_params = jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 1.0, 1.0])
    kl_params_helper(p_params, q_params)

    p_params, q_params = jnp.array([1.0, 2.0, 3.0]), jnp.array([5.0, 4.0, 1.0])
    kl_params_helper(p_params, q_params)

    p_params, q_params = jnp.array([5.0, 5.0, 5.0]), jnp.array([4.0, 4.0, 4.0])
    kl_params_helper(p_params, q_params)

    p_params, q_params = jnp.array([3.0, 3.0, 3.0]), jnp.array([4.0, 3.0, 3.0])
    kl_params_helper(p_params, q_params)

    p_params, q_params = jnp.array([15.0, 15.0, 20.0]), jnp.array([10.0, 10.0, 10.0])
    kl_params_helper(p_params, q_params)


def test_observation_transition_matrix():
    """
    Test the observation transition matrix
    """
    true_probs = jnp.array([0.7, 0.2, 0.1])
    dimension = len(true_probs)
    max_val = 5
    observation_matrix = get_observation_transition_matrix(true_probs, max_val)
    assert observation_matrix.shape == (max_val**dimension, max_val**dimension)
    prevalence_vector = jnp.zeros(max_val**dimension)
    prevalence_vector = prevalence_vector.at[0].set(1.0)
    next_prevalence_vector = jnp.dot(observation_matrix, prevalence_vector)
    assert jnp.isclose(jnp.sum(next_prevalence_vector), 1.0, atol=1e-3)

    # check the cells of the next prevalence vector
    index_1 = params_to_index(jnp.array([2, 1, 1]), max_val)
    index_2 = params_to_index(jnp.array([1, 2, 1]), max_val)
    index_3 = params_to_index(jnp.array([1, 1, 2]), max_val)

    assert jnp.isclose(next_prevalence_vector[index_1], 0.7, atol=1e-3)
    assert jnp.isclose(next_prevalence_vector[index_2], 0.2, atol=1e-3)
    assert jnp.isclose(next_prevalence_vector[index_3], 0.1, atol=1e-3)


def test_accuracy_reweighting_fn():
    """
    Test the accuracy reweighting function
    """
    true_probs = jnp.array([0.8, 0.2])
    pseudocount_probs = jnp.ones(5**2) / 5**2
    dimension = len(true_probs)
    max_val = 5
    reweighted_probs = accuracy_reweighting_fn(
        pseudocount_probs, dimension, max_val, true_probs
    )
    reweighted_probs /= reweighted_probs.sum()
    assert reweighted_probs.shape == (max_val**dimension,)
    assert jnp.isclose(jnp.sum(reweighted_probs), 1.0, atol=1e-3)
    good_index = params_to_index(jnp.array([5, 1]), max_val)
    bad_index = params_to_index(jnp.array([1, 5]), max_val)
    assert reweighted_probs[good_index] > reweighted_probs[bad_index]


def test_index_params_conversion():
    """
    Test that we're accurately converting between indices and parameters
    """
    pass
