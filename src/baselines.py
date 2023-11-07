import jax
import jax.numpy as jnp

from src.models import Policy

## NOTE: The models can get time_steps = 0, they should ignore those


class BernoulliTS(Policy):
    def __init__(self, expert_policy: jnp.ndarray):
        """Assumes binary rewards"""
        self.num_actions = expert_policy.shape[0]

    def __call__(self, rng_key, time_steps, actions, rewards):
        max_t = time_steps.max(axis=1).reshape(-1, 1)  # shape: (batch_size, 1)
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


class SuccElim(Policy):
    def __init__(self, expert_policy: jnp.ndarray):
        """Assumes binary rewards"""
        self.expert = expert_policy
        self.avail_actions = None

    def __call__(self, rng_key, time_steps, actions, rewards):
        num_actions = self.expert.shape[0]
        b_size = time_steps.shape[0]

        if self.avail_actions is None:
            self.avail_actions = jnp.ones(shape=(b_size, num_actions), dtype=bool)

        max_t = time_steps.max(axis=1).reshape(-1, 1)

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
        div_action_cnt = jnp.where(
            action_counts == 0, jnp.ones_like(action_counts), action_counts
        )
        means = jnp.sum(aug_rewards, axis=1) / div_action_cnt
        radius = jnp.sqrt(2 * jnp.log(max_t + 1) / div_action_cnt)
        lcb = means - radius
        ucb = means + radius
        self.avail_actions = jnp.where(
            self.avail_actions, ucb > lcb, self.avail_actions
        )

        policy = self.expert.reshape(1, -1) * self.avail_actions
        policy = policy / policy.sum(axis=1, keepdims=True)
        return jnp.log(policy)

    def __str__(self):
        return "SuccessiveElimination"
