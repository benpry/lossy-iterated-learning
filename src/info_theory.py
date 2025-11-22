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


# Dilip's implementation, with slight modifications
@jax.jit
def blahut_arimoto(source_p, distortion, beta, num_encodings, max_iters=100):
    # initialize with a uniform channel
    channel = jnp.ones_like(distortion) / num_encodings
    channel_marginal = jnp.matmul(source_p, channel)
    iters, D, prev_D = 0, jnp.inf, 0
    init_val = (channel, channel_marginal, D, prev_D, iters)

    def body_fun(val):
        channel, channel_marginal, D, prev_D, iters = val
        iters += 1
        prev_D = D
        channel_marginal = jnp.matmul(source_p, channel)
        channel = channel_marginal * jnp.exp2(-beta * distortion)
        channel /= jnp.sum(channel, axis=1, keepdims=True)
        D = jnp.matmul(source_p, channel * distortion).sum()
        return channel, channel_marginal, D, prev_D, iters

    def cond_fun(val):
        channel, channel_marginal, D, prev_D, iters = val
        return iters < max_iters

    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val=init_val)
    channel, channel_marginal, D, prev_D, iters = final_val

    R = entropy(channel_marginal, base=2.0) - jnp.average(
        entropy(channel, base=2.0, axis=1), weights=source_p, axis=0
    )

    return channel, channel_marginal, R, D, iters
