import jax
import chex
import jax.numpy as jnp

from jax.tree_util import Partial
from gymnax.environments import spaces

from .common import EnvParams
from src.utils import uniform_sample_ball


class UnitBall(spaces.Space):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def sample(self, rng: chex.PRNGKey, size: int = 1) -> chex.Array:
        return uniform_sample_ball(rng, size, self.d)

    def contains(self, x: chex.Array) -> bool:
        return jnp.all(jnp.linalg.norm(x, axis=-1) <= 1.0)


def get_linear_gaussian_bandit(max_episodes: int = 1000) -> EnvParams:
    d = 10

    def prior_fn(key, size):
        return uniform_sample_ball(key, size, d) * jnp.array([1.0, 1.0] + [0.0] * 8)

    def ref_prior_fn(key, size):
        return jax.random.normal(key, (size, d))

    def Q_funciton(params, context, action):
        params = params.reshape(-1, d)
        context = context.reshape(-1, d)
        action = action.reshape(-1, d)  # TODO fix these
        return jnp.einsum("nd,nd -> n", params, (context * action))

    def reward_dist_fn(key, params, context, action):
        return Q_funciton(params, context, action) + jax.random.normal(key)

    init_context_dist_fn = Partial(lambda key, size: jax.random.normal(key, (size, d)))

    def best_action_value_fn(params, context):
        params = params.reshape(-1, d)
        context = context.reshape(-1, d)  # TODO fix these
        norm = jnp.linalg.norm(context * params, axis=-1)
        return (context * params) / norm[..., None], norm

    return EnvParams(
        prior_fn=Partial(prior_fn),
        ref_prior_fn=Partial(ref_prior_fn),
        reward_dist_fn=Partial(reward_dist_fn),
        Q_function=Partial(Q_funciton),
        best_action_value_fn=Partial(best_action_value_fn),
        init_context_dist_fn=Partial(init_context_dist_fn),
        max_episodes=max_episodes,
        action_space=UnitBall(d),
        observation_space=None,
        state_space=None,
    )
