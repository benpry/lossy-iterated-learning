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
    index_to_params,
    kl_divergence_dirichlet,
    pad_with_zeros,
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


def conversion_helper(dimension, max_val):
    for index in range(max_val**dimension):
        params = index_to_params(index, dimension, max_val)
        assert jnp.all(
            params
            == index_to_params(params_to_index(params, max_val), dimension, max_val)
        )


def test_index_params_conversion():
    """
    Test that we're accurately converting between indices and parameters
    """
    conversion_helper(2, 5)
    conversion_helper(3, 5)
    conversion_helper(2, 10)
    conversion_helper(3, 10)


def reference_pad_with_zeros(array, smaller_max_val, larger_max_val, dimension):
    """
    The straightforward way to pad an array of probabilities, one entry at a time.

    pad_with_zeros has to do the same thing without looping in Python, so this pins down what it
    means before that optimization.
    """
    padded_array = jnp.zeros(larger_max_val**dimension)
    for i, prob in enumerate(array):
        params = index_to_params(i, dimension, smaller_max_val)
        new_index = params_to_index(params, larger_max_val)
        padded_array = padded_array.at[new_index].set(prob)

    return padded_array


def test_pad_with_zeros_matches_padding_one_entry_at_a_time():
    """
    Padding puts every probability at the index its pseudocounts have in the larger range
    """
    for smaller_max_val, larger_max_val, dimension in [(3, 4, 2), (4, 5, 2), (3, 5, 3)]:
        array = jnp.arange(1, smaller_max_val**dimension + 1, dtype=jnp.float64)
        array = array / array.sum()

        padded = pad_with_zeros(array, smaller_max_val, larger_max_val, dimension)

        expected = reference_pad_with_zeros(
            array, smaller_max_val, larger_max_val, dimension
        )
        assert jnp.array_equal(padded, expected)
        # nothing is lost or invented along the way
        assert jnp.isclose(padded.sum(), array.sum())
        assert padded.shape == (larger_max_val**dimension,)


def test_pad_with_zeros_keeps_pseudocounts_pointing_at_the_same_parameters():
    """
    An entry ends up at the index that decodes back to the pseudocounts it started from
    """
    smaller_max_val, larger_max_val, dimension = 3, 5, 2
    array = jnp.zeros(smaller_max_val**dimension).at[4].set(1.0)

    padded = pad_with_zeros(array, smaller_max_val, larger_max_val, dimension)

    moved_to = int(jnp.argmax(padded))
    assert jnp.array_equal(
        index_to_params(moved_to, dimension, larger_max_val),
        index_to_params(4, dimension, smaller_max_val),
    )
