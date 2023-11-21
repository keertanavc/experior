import jax
import jax.numpy as jnp

from src.configs import ExperiorConfig
from src.posterior import sglangevin


def test_sglangevin(conf: ExperiorConfig):
    rng = jax.random.PRNGKey(0)
    key, rng = jax.random.split(rng)
    data = jax.random.normal(key, (100, 5))

    key, rng = jax.random.split(rng)
    initial_params = jax.random.normal(key, (5,))

    log_prior = lambda params: -0.5 * jnp.sum(params**2)

    def log_likelihood(params, data):
        return -0.5 * jnp.sum((data - params) ** 2)

    key, rng = jax.random.split(rng)
    params = sglangevin(
        key, data, initial_params, log_prior, log_likelihood, delta=0.01
    )

    assert params.shape == initial_params.shape
    assert jnp.isnan(params).sum() == 0
