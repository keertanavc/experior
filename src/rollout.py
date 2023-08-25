import jax

import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=('policy_fn', 'horizon'))
def policy_rollout(policy_fn, rng_key, mu_vectors, horizon):
    """Returns a rollout of the policy, i.e. a sequence of actions and rewards.

    Args:
      policy_fn: A function that takes rng_key, timesteps, actions, rewards and
      returns a log-probability distribution over actions.
      rng_key: A JAX random key.
      mu_vectors: Samples from the prior over means, shape (n_samples, num_actions).
      horizon: The length of the rollout.

    # TODO reduce the variance by outputting the means and log_probs of all the actions.
    # TODO optimize more
    Returns:
        actions: The sequence of actions, shape (n_samples, horizon).
        rewards: The sequence of rewards, shape (n_samples, horizon).
        log_probs: The log-probabilities of the taken actions, shape (n_samples, horizon, num_actions).
    """
    n_envs = mu_vectors.shape[0]

    def policy_step(state_input, _):
        i, time_steps, actions, rewards, rng = state_input
        # TODO make the transformer input dynamic
        # shape: (n_envs, num_actions)
        rng, key = jax.random.split(rng)

        # shape: (n_envs, num_actions)
        log_prob = policy_fn(key, time_steps, actions, rewards)
        rng, key = jax.random.split(rng)
        new_action = jax.random.categorical(key, log_prob)

        rng, key = jax.random.split(rng)
        new_reward = jax.random.bernoulli(
            key, mu_vectors[jnp.arange(n_envs), new_action])

        actions = actions.at[:, i+1].set(new_action)
        time_steps = time_steps.at[:, i+1].set(i+1)
        rewards = rewards.at[:, i+1].set(new_reward)
        carry = (i + 1, time_steps, actions, rewards, rng)
        return carry, log_prob

    # TODO the first step is considered 0 with zero action and reward
    init_val = (-1,
                jnp.zeros((n_envs, horizon), dtype=jnp.int32),
                jnp.zeros((n_envs, horizon), dtype=jnp.int32),
                jnp.zeros((n_envs, horizon), dtype=jnp.float32),
                rng_key)

    (_, _, actions, rewards, _), log_probs = jax.lax.scan(
        policy_step, init_val, (), length=horizon)

    return actions, rewards, jnp.transpose(log_probs, axes=(1, 0, 2))
