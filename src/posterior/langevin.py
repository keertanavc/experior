import jax
import jax.numpy as jnp

from src.commons import Params, PRNGKey
from typing import Callable
from functools import partial


# TODO multiple passes over data
@partial(jax.jit, static_argnames=("log_prior", "log_likelihood"))
def sglangevin(
    rng: PRNGKey,
    data: jnp.ndarray,
    params: Params,
    log_prior: Callable[[Params], float],
    log_likelihood: Callable[[Params, jnp.ndarray], float],
    batch_size: int = 1,
    delta: float = 0.01,  # TODO make this dyanmic
):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler.
    """
    size = data.shape[0]
    assert size % batch_size == 0, "Batch size must divide data size."
    epochs = size // batch_size

    def _update(curr_param, noise_eps_batch):
        """
        One step of SGLD.
        """
        noise, eps, batch = noise_eps_batch
        prior_grad = jax.grad(log_prior)(curr_param)
        likelihood_grad = jax.grad(log_likelihood)(curr_param, batch)
        update = eps / 2.0 * (
            prior_grad + size / batch.shape[0] * likelihood_grad
        ) + noise * jnp.sqrt(eps)

        return curr_param + update, None

    epsilons = (2.0 * delta * jnp.arange(1, epochs + 1) / epochs).reshape(-1, 1)
    noises = jax.random.normal(rng, (epochs, params.shape[0]))
    key, rng = jax.random.split(rng)
    data = jax.random.permutation(key, data).reshape(epochs, batch_size, -1)
    noise_eps_batches = (noises, epsilons, data)

    params, _ = jax.lax.scan(_update, params, noise_eps_batches)
    return params
