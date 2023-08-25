from typing import Any
import jax
import jax.numpy as jnp

class Policy:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def __call__(self, rng_key, timesteps, actions, rewards):
        """Returns the log-probability of the actions.

        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).
        """
        raise NotImplementedError

class BernoulliTS(Policy):
    def __init__(self, num_actions: int):
        super().__init__(num_actions)

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
