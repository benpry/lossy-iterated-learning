"""
Test the information-theoretic utilities
"""

import jax.numpy as jnp

from src.experiment import get_distortion_matrix
from src.info_theory import (
    blahut_arimoto,
    entropy,
    remaining_change,
    remaining_movement,
)

MAX_ITERS = 100_000

# The beta where the optimal codebook for the small problem below changes shape. Convergence slows to
# a crawl here, so the movement still to come is tens of thousands of times the size of one step.
CRITICAL_BETA = 1.355
# Far more iterations than convergence needs, so a reference built this way owes nothing to whatever
# stopping rule is in force.
REFERENCE_ITERS = 5_000_000


def small_problem():
    """
    A problem small enough to iterate millions of times inside a test
    """
    distortion_matrix = get_distortion_matrix(
        max_val=5, dimension=2, lifespan=1, metric="jeff_divergence", cache=False
    )
    source_p = jnp.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]

    return source_p, distortion_matrix


def fixed_point_reference(source_p, distortion_matrix, beta):
    """
    Iterate until the channel stops moving at float64 resolution.

    A tolerance of zero means the loop can never stop on its own estimate of what is left, so this is
    an oracle for where the fixed point is that does not depend on the rule being tested.
    """
    channel, *_ = blahut_arimoto(
        source_p,
        distortion_matrix,
        beta,
        distortion_matrix.shape[1],
        max_iters=REFERENCE_ITERS,
        tolerance=0.0,
    )

    return channel


def channel_distance(channel, other):
    """
    The largest total variation distance between corresponding rows of two channels
    """
    return float(jnp.max(jnp.sum(jnp.abs(channel - other), axis=1)) / 2)


def test_blahut_arimoto_lands_within_tolerance_of_the_fixed_point():
    """
    The channel that comes back is within `tolerance` of the fixed point, even where convergence is
    slowest.

    Away from the betas where the codebook changes shape this is easy, because one step is about the
    size of the distance left to travel. Near them it is the whole problem: the channel creeps toward
    the fixed point in steps far smaller than the distance remaining, so the rule has to estimate
    that distance rather than read it off the last step.
    """
    source_p, distortion_matrix = small_problem()
    tolerance = 1e-8

    channel, _, _, _, iters, remaining = blahut_arimoto(
        source_p,
        distortion_matrix,
        CRITICAL_BETA,
        distortion_matrix.shape[1],
        max_iters=REFERENCE_ITERS,
        tolerance=tolerance,
    )
    reference = fixed_point_reference(source_p, distortion_matrix, CRITICAL_BETA)

    # it stopped because it decided it was done, not because it ran out of iterations
    assert iters < REFERENCE_ITERS
    assert remaining <= tolerance
    # and it really was done
    assert channel_distance(channel, reference) <= tolerance


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
    channel, channel_marginal, rate, distortion, iters, remaining = blahut_arimoto(
        source_distribution, distortion_matrix, beta, num_encodings
    )
    assert jnp.isclose(jnp.sum(channel) / num_encodings, 1.0, atol=1e-3)
    assert jnp.isclose(jnp.sum(channel_marginal), 1.0, atol=1e-3)
    assert jnp.isclose(rate, jnp.log2(num_encodings), atol=1e-3)
    assert jnp.isclose(distortion, 0.0, atol=1e-3)

    beta = 1e-6
    channel, channel_marginal, rate, distortion, iters, remaining = blahut_arimoto(
        source_distribution, distortion_matrix, beta, num_encodings
    )
    assert jnp.isclose(jnp.sum(channel) / num_encodings, 1.0, atol=1e-3)
    assert jnp.isclose(jnp.sum(channel_marginal), 1.0, atol=1e-3)
    assert jnp.isclose(rate, 0.0, atol=1e-3)
    assert jnp.isclose(distortion, 2 / 3, atol=1e-3)


def test_blahut_arimoto_converges_to_a_fixed_point():
    """
    The channel it returns satisfies the Blahut-Arimoto fixed point equations.

    This is a statement about the whole channel, not just its rate, distortion, or marginal: every
    row has to be the distribution the algorithm's own update rule would produce.
    """
    distortion_matrix = get_distortion_matrix(
        max_val=4, dimension=2, lifespan=1, metric="jeff_divergence", cache=False
    )
    source_p = jnp.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]
    beta = 2.0
    tolerance = 1e-10

    channel, channel_marginal, rate, distortion, iters, remaining = blahut_arimoto(
        source_p,
        distortion_matrix,
        beta,
        distortion_matrix.shape[1],
        max_iters=MAX_ITERS,
        tolerance=tolerance,
    )

    # it stopped because it converged, not because it ran out of iterations
    assert remaining < tolerance
    assert iters < MAX_ITERS

    # every row is what the update rule would produce from the marginal
    unnormalized = channel_marginal * jnp.exp2(-beta * distortion_matrix)
    fixed_point = unnormalized / jnp.sum(unnormalized, axis=1, keepdims=True)
    assert jnp.max(jnp.abs(channel - fixed_point)) < 1e-9

    # the marginal describes the channel that came back, not the iterate before it
    assert jnp.allclose(channel_marginal, jnp.matmul(source_p, channel), atol=1e-12)


def test_blahut_arimoto_reports_when_it_runs_out_of_iterations():
    """
    Stopping at the iteration cap is reported, rather than passing for convergence
    """
    distortion_matrix = get_distortion_matrix(
        max_val=4, dimension=2, lifespan=1, metric="jeff_divergence", cache=False
    )
    source_p = jnp.ones(distortion_matrix.shape[0]) / distortion_matrix.shape[0]
    tolerance = 1e-10

    channel, channel_marginal, rate, distortion, iters, remaining = blahut_arimoto(
        source_p,
        distortion_matrix,
        2.0,
        distortion_matrix.shape[1],
        max_iters=3,
        tolerance=tolerance,
    )

    assert iters == 3
    assert remaining > tolerance


def test_blahut_arimoto_does_not_underflow_at_large_beta():
    """
    Large betas must not drive the exponentials to zero and turn the channel into NaNs
    """
    # every distortion is large enough that exp2(-beta * distortion) underflows to zero
    distortion_matrix = jnp.array([[100.0, 200.0], [200.0, 100.0]])
    source_p = jnp.array([0.5, 0.5])

    channel, channel_marginal, rate, distortion, iters, remaining = blahut_arimoto(
        source_p, distortion_matrix, 1000.0, 2
    )

    assert not jnp.isnan(channel).any()
    assert jnp.allclose(jnp.sum(channel, axis=1), 1.0)
    # each source point should be sent to the encoding that distorts it least
    assert jnp.allclose(channel, jnp.eye(2))


def test_remaining_change_sums_the_geometric_series():
    """
    The estimated movement left is the sum of the remaining steps, not the size of the last one
    """
    # steps shrinking by a factor of 10: the remaining movement is 1e-4 + 1e-5 + ... = 1.111e-4
    assert jnp.isclose(remaining_change(1e-3, 1e-2), 1e-3 / 9)

    # steps shrinking by a factor of 2: the remaining movement is as large as the last step
    assert jnp.isclose(remaining_change(0.5, 1.0), 0.5)


def test_remaining_change_grows_as_the_steps_stop_shrinking():
    """
    A ratio close to 1 means a small step still hides a lot of movement to come.

    This is the case near the betas where the optimal codebook changes shape, and it is why the size
    of a single step is not a usable convergence criterion.
    """
    step = 1e-4
    slow = remaining_change(step, step / 0.999)

    assert jnp.isclose(slow, step * 0.999 / 0.001)
    # what is left to come is hundreds of times the size of the step you can see
    assert slow / step > 900


def test_remaining_change_is_infinite_when_the_steps_are_not_shrinking():
    """
    Steps that hold steady or grow give no reason to believe the channel will settle
    """
    assert jnp.isinf(remaining_change(1e-3, 1e-3))
    assert jnp.isinf(remaining_change(1e-2, 1e-3))


def test_remaining_movement_is_unknown_until_two_windows_have_gone_by():
    """
    A single window's displacement says nothing about how fast the movement is decaying
    """
    assert jnp.isinf(remaining_movement(1e-5, jnp.inf))
    assert jnp.isinf(remaining_movement(jnp.inf, jnp.inf))


def test_remaining_movement_sums_the_geometric_series_over_windows():
    """
    Given two windows to compare, it is the same geometric sum, measured over windows
    """
    assert jnp.isclose(remaining_movement(1e-3, 1e-2), 1e-3 / 9)
    assert jnp.isclose(remaining_movement(0.5, 1.0), 0.5)


def test_remaining_movement_is_zero_once_the_channel_stops_moving():
    """
    A window that moves less than float64 can resolve has no movement left to estimate.

    Rounding alone shifts a row's total variation by around 1e-16 per step, so once a whole window
    moves less than that the two displacements being compared are noise. Their ratio is as likely as
    not to come out above one, which reads as "this will never converge", and the loop would run to
    its iteration cap on a channel that is already exactly at the fixed point.
    """
    assert remaining_movement(1e-16, 1e-16) == 0.0
    assert remaining_movement(1e-16, 1e-18) == 0.0
    assert remaining_movement(0.0, 0.0) == 0.0


def test_blahut_arimoto_stops_once_the_channel_stops_moving():
    """
    A channel that reaches the fixed point outright stops, rather than running to the iteration cap
    """
    # every source point has one encoding that is plainly best, so this settles within a few steps
    distortion_matrix = jnp.array([[100.0, 200.0], [200.0, 100.0]])
    source_p = jnp.array([0.5, 0.5])
    max_iters = 100_000

    *_, iters, remaining = blahut_arimoto(
        source_p, distortion_matrix, 1000.0, 2, max_iters=max_iters, tolerance=1e-10
    )

    assert iters < max_iters
    assert remaining <= 1e-10


def test_remaining_change_is_zero_once_the_channel_stops_moving():
    """
    A channel that has stopped moving has no movement left, however it got there
    """
    assert remaining_change(0.0, 1e-3) == 0.0
    assert remaining_change(0.0, 0.0) == 0.0


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
