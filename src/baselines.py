import jax
import jax.numpy as jnp

from src.models import Policy
from src.configs import ExperiorConfig

from typing import Dict


class BernoulliTS(Policy):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def _ts(self, rng_key, actions, rewards):
        # actions.shape = (T, ), rewards.shape = (T, )
        alpha, beta = jnp.ones(self.num_actions), jnp.ones(self.num_actions)

        for a, r in zip(actions, rewards):
            alpha += jnp.where(jnp.arange(self.num_actions) == a, r, 0)
            beta += jnp.where(jnp.arange(self.num_actions) == a, 1 - r, 0)

        sampled_theta = jax.random.beta(rng_key, alpha, beta)
        action_probs = jnp.where(sampled_theta == jnp.max(sampled_theta), 1, 0)
        return jnp.log(action_probs)

    def __call__(self, rng_key, time_steps, actions, rewards):
        b_size = actions.shape[0]
        keys = jax.random.split(rng_key, b_size)
        return jax.vmap(self._ts)(keys, actions, rewards)

    def __str__(self):
        return "BernoulliTS"


def get_baselines(conf: ExperiorConfig) -> Dict[str, Policy]:
    return {"BernoulliTS": BernoulliTS(conf.prior.num_actions)}
