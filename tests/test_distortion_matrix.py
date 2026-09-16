"""
Test the computation of distortion matrices
"""

import jax.numpy as jnp
import numpy as np
import pytest

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


def test_distortion_matrix_cache_is_written_atomically(tmp_path, monkeypatch):
    """
    The cached matrix appears complete or not at all.

    Several workers starting at once will all try to build the same matrix, and a half written file
    would be loaded by later runs as if it were real data.
    """
    monkeypatch.setenv("SCR_ROOT_DIR", str(tmp_path))

    computed = get_distortion_matrix(
        max_val=5, dimension=2, metric="jeff_divergence", lifespan=1
    )

    cache_dir = tmp_path / "cache" / "distortion_matrices"
    cached_files = sorted(p.name for p in cache_dir.iterdir())
    # exactly the finished matrix, with no temporary file left behind
    assert len(cached_files) == 1, cached_files
    assert cached_files[0].endswith(".npy")
    assert not cached_files[0].endswith(".tmp")

    # and reading it back gives what was computed
    reloaded = get_distortion_matrix(
        max_val=5, dimension=2, metric="jeff_divergence", lifespan=1
    )
    assert jnp.array_equal(jnp.asarray(reloaded), jnp.asarray(computed))


def test_distortion_matrix_cache_is_not_left_half_written(tmp_path, monkeypatch):
    """
    A write that fails partway leaves nothing at the cache path.

    Several workers build this matrix at once, and a partial file at the cache path would be loaded
    by every later run as if it were real data.
    """
    monkeypatch.setenv("SCR_ROOT_DIR", str(tmp_path))

    real_save = np.save

    def save_half_then_fail(file, arr, *args, **kwargs):
        real_save(file, arr[: len(arr) // 2])
        raise OSError("no space left on device")

    monkeypatch.setattr("src.experiment.np.save", save_half_then_fail)

    with pytest.raises(OSError):
        get_distortion_matrix(
            max_val=5, dimension=2, metric="jeff_divergence", lifespan=1
        )

    cache_dir = tmp_path / "cache" / "distortion_matrices"
    left_behind = list(cache_dir.glob("*.npy")) if cache_dir.exists() else []
    assert not left_behind, f"a half written matrix was left at {left_behind}"
