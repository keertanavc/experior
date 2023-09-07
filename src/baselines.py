import jax
import jax.numpy as jnp

from src.models import Policy
from src.configs import ExperiorConfig

from typing import Dict

## NOTE: The models can get time_steps = 0, they should ignore those


class BernoulliTS(Policy):
    def __init__(self, num_actions: int):
        """Assumes binary rewards"""
        self.num_actions = num_actions

    def __call__(self, rng_key, time_steps, actions, rewards):
        max_t = time_steps.max(axis=1).reshape(-1, 1)  # shape: (batch_size, 1)
        b_size = time_steps.shape[0]
        num_actions = self.num_actions

        # shape: (batch_size, T)
        idx = (time_steps[:, :] <= max_t) & (time_steps[:, :] > 0)

        weights = jnp.ones_like(actions, dtype=jnp.float32) * idx
        # count the number of times each action was taken for each batch
        action_counts = (jnp.eye(num_actions)[actions] * weights[:, :, None]).sum(
            axis=1
        )

        # shape: (batch_size, T,  num_actions)
        aug_rewards = (rewards * idx)[:, :, None] * (
            actions[:, :, None] == jnp.arange(num_actions)[None, None, :]
        )

        # shape: (batch_size, num_actions)
        sum_rewards = jnp.sum(aug_rewards, axis=1)

        alpha = sum_rewards + 1
        beta = action_counts - sum_rewards + 1
        sampled_theta = jax.random.beta(rng_key, alpha, beta)
        action_probs = jnp.where(
            sampled_theta == jnp.max(sampled_theta, axis=1, keepdims=True), 1, 0
        )
        return jnp.log(action_probs)

    def __str__(self):
        return "BernoulliTS"


def get_baselines(conf: ExperiorConfig) -> Dict[str, Policy]:
    return {"BernoulliTS": BernoulliTS(conf.prior.num_actions)}
