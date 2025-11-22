"""
Test the computation of distortion matrices
"""

import jax.numpy as jnp

from src.experiment import get_distortion_matrix
from src.utils import params_to_index


def test_distortion_matrix_shapes():
    distortion_matrix = get_distortion_matrix(
        max_val=5, dimension=2, lifespan=0, metric="jeff_divergence", cache=False
    )

    assert distortion_matrix.shape == (5**2, 5**2)

    distortion_matrix = get_distortion_matrix(
        max_val=5, dimension=2, lifespan=1, metric="jeff_divergence", cache=False
    )

    assert distortion_matrix.shape == (5**2, 4**2)

    distortion_matrix = get_distortion_matrix(
        max_val=7, dimension=2, lifespan=2, metric="jeff_divergence", cache=False
    )

    assert distortion_matrix.shape == (7**2, 5**2)

    distortion_matrix = get_distortion_matrix(
        max_val=5, dimension=3, lifespan=0, metric="jeff_divergence", cache=False
    )

    assert distortion_matrix.shape == (5**3, 5**3)


def test_distortion_matrix_values():
    distortion_matrix = get_distortion_matrix(
        max_val=5, dimension=2, lifespan=0, metric="jeff_divergence", cache=False
    )

    # the diagonal should be all 0s
    assert jnp.all(distortion_matrix.diagonal() == 0.0)

    # the matrix should be symmetric
    assert jnp.allclose(distortion_matrix, distortion_matrix.T)

    # the matrix should be non-negative
    assert jnp.all(distortion_matrix >= 0.0)

    # parameters that are very different should have higher distortion than parameters that are only a little bit different
    base_index = params_to_index(jnp.array([1, 1]), max_val=5)
    similar_index = params_to_index(jnp.array([1, 2]), max_val=5)
    different_index = params_to_index(jnp.array([1, 5]), max_val=5)
    assert (
        distortion_matrix[base_index, similar_index]
        < distortion_matrix[base_index, different_index]
    )

    base_index = params_to_index(jnp.array([3, 2]), max_val=5)
    similar_index = params_to_index(jnp.array([4, 2]), max_val=5)
    different_index = params_to_index(jnp.array([2, 4]), max_val=5)
    assert (
        distortion_matrix[base_index, similar_index]
        < distortion_matrix[base_index, different_index]
    )
