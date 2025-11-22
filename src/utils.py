"""
Utilities and helper functions
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gamma


def logprob_true_probs(params, true_probs):
    """
    Compute the log-probability of the true probabilities given the parameters
    """
    true_probs = jnp.array(true_probs)
    params = jnp.array(params)
    return jax.scipy.stats.dirichlet.logpdf(true_probs, params)


def logps_true_probs(pseudocount_probs, dimension, max_val, true_probs):
    """
    Compute the log-probabilities of the true probabilities for each set of parameters, represented
    as a vector of pseudocount probabilities
    """
    param_array = jax.vmap(
        partial(index_to_params, dimension=dimension, max_val=max_val)
    )(jnp.arange(len(pseudocount_probs)))
    logps = jax.vmap(partial(logprob_true_probs, true_probs=true_probs))(param_array)
    return logps


def expected_logp_true_probs_from_string(
    pseudocount_probs_str, true_probs, dimension, max_val
):
    """
    Compute the expected log-probability of the true probabilities given a string that represents a
    vector of pseudocount probabilities
    """
    pseudocount_probs = jnp.fromstring(pseudocount_probs_str[1:-1], sep=" ")

    if not jnp.isclose(jnp.sum(pseudocount_probs), 1.0, atol=1e-3):
        warnings.warn(
            f"Pseudocounts do not sum to 1.0. Sum: {jnp.sum(pseudocount_probs)}"
        )

    logps = logps_true_probs(pseudocount_probs, dimension, max_val, true_probs)

    return jnp.dot(pseudocount_probs, logps)


def kl_divergence_dirichlet(p_params, q_params):
    """
    Implementation of the KL divergence between two Dirichlet distributions.
    Formula from https://statproofbook.github.io/P/dir-kl.html
    """
    # compute the kl divergence between two dirichlet distributions
    p_sum = jnp.sum(p_params)
    q_sum = jnp.sum(q_params)

    # log gamma quotient
    log_gamma_quotient = jnp.log(gamma(p_sum) / gamma(q_sum))
    sum_log_gammas = jnp.sum(jnp.log(gamma(q_params) / gamma(p_params)))
    sum_diffs = jnp.sum((p_params - q_params) * (digamma(p_params) - digamma(p_sum)))

    return (log_gamma_quotient + sum_log_gammas + sum_diffs) / jnp.log(
        2
    )  # convert to bits


def kl_divergence_dirichlet_from_indices(
    p_index, q_index, dimension, p_max_val, q_max_val
):
    """
    Implementation of the KL divergence between two Dirichlet distributions from indices
    """
    p_params = index_to_params(p_index, dimension, max_val=p_max_val)
    q_params = index_to_params(q_index, dimension, max_val=q_max_val)
    return kl_divergence_dirichlet(p_params, q_params)


@partial(jax.jit, static_argnums=(2, 3))
def jeff_divergence_dirichlet_from_indices(
    p_index, q_index, dimension, p_max_val, q_max_val
):
    """
    Implementation of Jeffreys divergence between two Dirichlet distributions
    """
    p_params = index_to_params(p_index, dimension, max_val=p_max_val)
    q_params = index_to_params(q_index, dimension, max_val=q_max_val)
    divergence = kl_divergence_dirichlet(p_params, q_params) + kl_divergence_dirichlet(
        q_params, p_params
    )
    return divergence


def params_to_index(pseudocounts, max_val=10):
    """
    Convert a list of pseudocounts to an index
    """
    dimension = len(pseudocounts)
    index = 0
    for i in range(dimension):
        count = pseudocounts[i]
        index += (count - 1) * (max_val ** (i))

    return index.astype(int)


def index_to_params(index, dimension, max_val=10):
    """
    Decode a list of pseudocounts back to an index
    """
    params = jnp.zeros(dimension, dtype=jnp.float64)
    remaining_index = index

    for i in range(dimension - 1, -1, -1):
        param = remaining_index // (max_val**i)
        params = params.at[i].set(param + 1)
        remaining_index = remaining_index - param * (max_val**i)

    return params


def compute_column(original_index, max_val, dimension, observation):
    """
    Compute the column of the observation matrix corresponding to the original index
    """
    params = index_to_params(original_index, dimension, max_val)
    new_params = params.at[observation].set(params[observation] + 1)
    new_index = params_to_index(new_params, max_val)

    column = jnp.zeros(max_val**dimension)

    new_column = column.at[new_index].set(1)
    old_column = column.at[original_index].set(1)

    return jnp.where(jnp.max(params) == max_val, old_column, new_column)


def get_observation_transition_matrix(probs, max_val):
    dimension = len(probs)
    size = max_val**dimension
    matrices = []

    # compute the transition matrix associated with each possible observation
    for observation in range(dimension):
        obs_mat = jax.vmap(
            partial(
                compute_column,
                max_val=max_val,
                dimension=dimension,
                observation=observation,
            )
        )(jnp.arange(size))
        matrices.append(obs_mat)

    # compute the observation transition matrix as a weighted sum
    full_matrix = jnp.zeros((size, size))
    for i, p in enumerate(probs):
        full_matrix += p * matrices[i]

    return full_matrix.transpose()


@partial(jax.jit, static_argnums=(1, 2, 3))
def pad_with_zeros(array, smaller_max_val, larger_max_val, dimension):
    """
    Take an array of probabilities with a smaller max value and pad it to include larger max values
    """
    size = larger_max_val**dimension
    padded_array = jnp.zeros(size)
    for i, prob in enumerate(array):
        params = index_to_params(i, dimension, smaller_max_val)
        new_index = params_to_index(params, larger_max_val)
        padded_array = padded_array.at[new_index].set(prob)

    return padded_array


def uniform_source_density(index, dimension, max_val):
    return 1.0


def decreasing_source_density(index, dimension, max_val):
    params = index_to_params(index, dimension=dimension, max_val=max_val)
    return 1.1 ** (dimension - jnp.sum(params))


def peaked_source_density(index, dimension, max_val):
    params = index_to_params(index, dimension=dimension, max_val=max_val)
    return jnp.where(
        jnp.logical_or(
            jnp.all(params == jnp.array([5, 1])), jnp.all(params == jnp.array([1, 5]))
        ),
        1000.0,
        1.0,
    )


def accuracy_reweighting_fn(
    pseudocount_probs, dimension, max_val, true_probs, temp=1.0
):
    """
    Reweight the pseudocounts based on the probability they assign to the true probabilities
    """
    logps = logps_true_probs(pseudocount_probs, dimension, max_val, true_probs)
    return jnp.exp(logps / temp)
