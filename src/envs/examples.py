import jax
import jax.numpy as jnp

from jax.tree_util import Partial
from gymnax.environments import spaces

from src.utils import uniform_sample_ball
from .bandit import EnvParams

toy_deterministic_bandit = EnvParams(
    action_space=spaces.Discrete(3),
    observation_space=None,
    state_space=None,
    reward_prior_fn=Partial(lambda key: jnp.array([0.0, 1.0, 0.0])),
    reward_dist_fn=Partial(
        lambda key, params, context, action: jax.random.bernoulli(key, params).astype(
            jnp.float32
        )[action]
    ),
    context_dist_fn=Partial(lambda key: jnp.array([0.0])),  # fixed context
    best_action_reward_fn=Partial(
        lambda params, context: (jnp.argmax(params), jnp.max(params))
    ),
    max_steps_in_episode=100,
)

linear_gaussian_bandit = EnvParams(
    reward_prior_fn=Partial(
        lambda key: (
            uniform_sample_ball(key, 1, 10) * jnp.array([1.0, 1.0] + [0.0] * 8)
        ).reshape(
            10,
        )
    ),
    reward_dist_fn=Partial(
        lambda key, params, context, action: (context * action) @ params
        + jax.random.normal(key)
    ),
    context_dist_fn=Partial(lambda key: jax.random.normal(key, (10,))),
    best_action_reward_fn=Partial(
        lambda params, context: (
            (params * context) / jnp.linalg.norm(params * context),
            jnp.linalg.norm(params * context),
        ),
    ),
    max_steps_in_episode=1000,
    action_space=spaces.Box(-1.0, 1.0, (10,)),
    observation_space=None,
    state_space=None,
)
