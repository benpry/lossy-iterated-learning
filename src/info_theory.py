"""
This file contains code for information theory algorithms.
"""

import jax
import jax.numpy as jnp


def entropy(x, base=2.0, axis=0):
    ret = jax.scipy.special.entr(x)
    ret = jnp.sum(ret, axis=axis)
    if base is not None:
        ret /= jnp.log(base)
    return ret


# How many iterations to let pass between measurements of how far the channel has travelled.
#
# The decay across a single step cannot be measured. Near the betas where the codebook changes shape
# the steps shrink by around one part in a million each, while a step size -- a maximum over rows of
# a sum of differences that cancel almost exactly -- is itself only good to a few parts in a million.
# The ratio of two consecutive steps is then dominated by its own rounding error, and comes out above
# one about as often as below it. Measuring across a window multiplies the decay being looked for by
# the number of steps in it without multiplying the noise, so the two separate cleanly.
CONVERGENCE_WINDOW = 1000

# Movement across a whole window below which the channel counts as having stopped.
#
# Rounding alone shifts a row's total variation by of order 1e-16 per step, so a window that moves
# less than this is measuring nothing but arithmetic. Two such windows have a meaningless ratio,
# above one as often as not, which would read as "this will never converge". At the tolerances this
# code is used with, a window this quiet corresponds to well under 1e-10 of movement still to come,
# so nothing real is being hidden.
SETTLED_DISPLACEMENT = 1e-13


def remaining_change(change, previous_change):
    """
    Estimate how much further a channel will move before it converges.

    Blahut-Arimoto converges geometrically, so if the channel moved `change` over one interval and
    `previous_change` over the one before, what is left is the sum of a geometric series. Near the
    betas where the shape of the optimal codebook changes, that ratio approaches one and the movement
    still to come is thousands of times the size of a single step, which is why the size of one step
    is not on its own a usable measure of convergence.

    The two arguments have to be movements over equal intervals, but nothing here requires those
    intervals to be single iterations. See `remaining_movement` for why they should not be.
    """
    is_shrinking = jnp.asarray(previous_change) > 0.0
    # a channel that was not moving before is not moving now either, so call the ratio zero
    ratio = jnp.where(
        is_shrinking, change / jnp.where(is_shrinking, previous_change, 1.0), 0.0
    )
    return jnp.where(ratio < 1.0, change * ratio / (1.0 - ratio), jnp.inf)


def remaining_movement(displacement, previous_displacement):
    """
    Estimate how much further a channel will move, from how far it travelled over two windows.

    This is `remaining_change` measured across windows of `CONVERGENCE_WINDOW` iterations rather than
    across single steps, which is what makes the ratio it depends on measurable at all. It adds the
    two cases a window-based measurement has to handle: nothing can be said before two windows have
    gone by, and a channel already sitting at the fixed point has to be recognised rather than have
    its rounding error extrapolated.
    """
    displacement = jnp.asarray(displacement)
    has_settled = displacement <= SETTLED_DISPLACEMENT
    is_measurable = jnp.isfinite(displacement) & jnp.isfinite(
        jnp.asarray(previous_displacement)
    )

    return jnp.where(
        has_settled,
        0.0,
        jnp.where(
            is_measurable,
            remaining_change(displacement, previous_displacement),
            jnp.inf,
        ),
    )


# Dilip's implementation, with slight modifications
@jax.jit
def blahut_arimoto(
    source_p,
    distortion,
    beta,
    num_encodings,
    max_iters=100_000,
    tolerance=1e-8,
    window=CONVERGENCE_WINDOW,
):
    """
    Find the channel that minimizes the rate plus beta times the distortion.

    Iterates until the movement still to come falls below `tolerance` rather than for a fixed number
    of steps, because how long that takes varies by orders of magnitude with beta: on the sweeps this
    is used for, the cheapest beta settles in a dozen iterations and the dearest takes four million.
    Returns the movement it estimates is still to come, so a caller can refuse a channel that never
    converged.

    That estimate is taken across windows of `window` iterations rather than across single steps. It
    is the distance still to travel that matters, and near the betas where the codebook changes shape
    that distance is hundreds of thousands of times one step, so it has to be inferred from the rate
    at which the steps are decaying -- and that rate is not measurable one step at a time.
    """
    # exp2(-beta * distortion) is the same on every iteration and is the most expensive part of a
    # step, so compute it once. Subtracting each row's smallest value cancels out in the
    # normalization below, and keeps the exponentials from underflowing to zero when beta is large.
    scaled_distortion = beta * distortion
    distortion_kernel = jnp.exp2(
        -(scaled_distortion - jnp.min(scaled_distortion, axis=1, keepdims=True))
    )

    # Start from a uniform channel, with the current window opening on it and no window behind us.
    # The channel is carried twice: once as the current iterate and once as the point the open window
    # started from, which the iterate is measured against when the window closes.
    unknown_displacement = jnp.asarray(jnp.inf, dtype=distortion.dtype)
    initial_channel = jnp.ones_like(distortion) / num_encodings
    init_val = (
        initial_channel,
        initial_channel,
        unknown_displacement,
        unknown_displacement,
        0,
    )

    def body_fun(val):
        channel, window_start, displacement, previous_displacement, iters = val
        channel_marginal = jnp.matmul(source_p, channel)
        new_channel = channel_marginal * distortion_kernel
        new_channel /= jnp.sum(new_channel, axis=1, keepdims=True)
        new_iters = iters + 1

        def close_the_window():
            # The largest total variation distance any source point's distribution over encodings has
            # moved since the window opened. This tracks the whole channel rather than its marginal,
            # and it bounds the source-weighted version, so it stays meaningful for rows the source
            # rarely visits.
            travelled = (
                jnp.max(jnp.sum(jnp.abs(new_channel - window_start), axis=1)) / 2
            )
            return new_channel, travelled, displacement

        def leave_it_open():
            return window_start, displacement, previous_displacement

        # comparing against the window start costs a pass over the channel, so only do it when the
        # window actually closes rather than on every iteration
        new_window_start, new_displacement, new_previous_displacement = jax.lax.cond(
            new_iters % window == 0, close_the_window, leave_it_open
        )

        return (
            new_channel,
            new_window_start,
            new_displacement,
            new_previous_displacement,
            new_iters,
        )

    def cond_fun(val):
        _, _, displacement, previous_displacement, iters = val
        # before two windows have closed this is infinite, so the loop keeps going
        return (iters < max_iters) & (
            remaining_movement(displacement, previous_displacement) > tolerance
        )

    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val=init_val)
    channel, _, displacement, previous_displacement, iters = final_val

    # Recompute the marginal from the channel being returned. The marginal from inside the loop
    # belongs to the previous iterate, which would make the rate below describe two different
    # channels at once.
    channel_marginal = jnp.matmul(source_p, channel)
    D = jnp.matmul(source_p, channel * distortion).sum()
    R = entropy(channel_marginal, base=2.0) - jnp.average(
        entropy(channel, base=2.0, axis=1), weights=source_p, axis=0
    )

    return (
        channel,
        channel_marginal,
        R,
        D,
        iters,
        remaining_movement(displacement, previous_displacement),
    )
