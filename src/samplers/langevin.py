import jax
import chex

import jax.numpy as jnp

from typing import Callable, Any

Param = Any


def langevin_sampling(
    rng: chex.PRNGKey,
    init_param: Param,
    log_pdf: Callable[[chex.PRNGKey, Param], chex.Array],
    step_size: float,
    num_steps: int,
    grad_opt: Callable[[Param], Param],
):
    """Langevin dynamics sampling of a distribution.

    Args:
        rng: Jax random key.
        init_param: Initial parameter.
        log_pdf: A function that returns the (unnormalized) log pdf of a parameter.
        step_size: The initial step size of the Langevin dynamics.
        num_steps: The number of steps of the Langevin dynamics.
        grad_opt: A function that transforms the gradient, e.g., to clip it.

    Returns:
        The updated parameter.
    """

    # From https://icml.cc/2011/papers/398_icmlpaper.pdf
    def _update_parameters(state, i):
        param, rng = state
        # Compute the gradient of the unnormalized log prior
        rng, rng_ = jax.random.split(rng)
        grad_log_prior = jax.grad(lambda k, p: log_pdf(k, p).sum(), argnums=(1))(
            rng_, param
        )

        # write a clip the gradient
        grad_log_prior = grad_opt(grad_log_prior)
        step = step_size / i

        # Langevin dynamics update rule
        rng, rng_ = jax.random.split(rng)
        noise = jax.random.normal(rng_, param.shape) * jnp.sqrt(2 * step)
        updated_param = param + step * grad_log_prior + noise
        state = (updated_param, rng)
        return state, None

    updated_param, _ = jax.lax.scan(
        _update_parameters,
        (init_param, rng),
        jnp.arange(1, num_steps + 1),
    )
    return updated_param[0]
