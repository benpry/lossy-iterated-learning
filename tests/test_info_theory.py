"""
Test the information-theoretic utilities
"""

import jax.numpy as jnp

from src.info_theory import blahut_arimoto, entropy


def test_blahut_arimoto():
    """
    Test the Blahut-Arimoto algorithm on a simple example
    """
    distortion_matrix = jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    source_distribution = (
        jnp.ones(distortion_matrix.shape[1]) / distortion_matrix.shape[1]
    )
    num_encodings = distortion_matrix.shape[1]

    beta = 1e6
    channel, channel_marginal, rate, distortion, iters = blahut_arimoto(
        source_distribution, distortion_matrix, beta, num_encodings, max_iters=1000
    )
    assert jnp.isclose(jnp.sum(channel) / num_encodings, 1.0, atol=1e-3)
    assert jnp.isclose(jnp.sum(channel_marginal), 1.0, atol=1e-3)
    assert jnp.isclose(rate, jnp.log2(num_encodings), atol=1e-3)
    assert jnp.isclose(distortion, 0.0, atol=1e-3)

    beta = 1e-6
    channel, channel_marginal, rate, distortion, iters = blahut_arimoto(
        source_distribution, distortion_matrix, beta, num_encodings, max_iters=1000
    )
    assert jnp.isclose(jnp.sum(channel) / num_encodings, 1.0, atol=1e-3)
    assert jnp.isclose(jnp.sum(channel_marginal), 1.0, atol=1e-3)
    assert jnp.isclose(rate, 0.0, atol=1e-3)
    assert jnp.isclose(distortion, 2 / 3, atol=1e-3)


def test_entropy():
    """
    Test the entropy function
    """
    x = jnp.ones(10) / 10
    assert jnp.isclose(entropy(x, base=2.0), jnp.log2(10), atol=1e-3)
    assert jnp.isclose(entropy(x, base=10), 1.0, atol=1e-3)

    x = jnp.array([0.0, 1.0])
    assert jnp.isclose(entropy(x, base=2.0), 0.0, atol=1e-3)

    x = jnp.array([0.5, 0.5])
    assert jnp.isclose(entropy(x, base=2.0), 1.0, atol=1e-3)

    x = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.isclose(entropy(x, base=2.0), 2.0, atol=1e-3)
