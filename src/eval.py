import jax
import jax.numpy as jnp

from src.rollout import policy_rollout


def uniform_bayes_regret(rng_key, policy_fn, num_actions, horizon, n_envs):
    rng_key, key = jax.random.split(rng_key)
    mu_vectors = jax.random.uniform(key, shape=(n_envs, num_actions))

    rng_key, key = jax.random.split(rng_key)
    actions, _, _ = policy_rollout(policy_fn, key, mu_vectors, horizon)

    means = mu_vectors[jnp.arange(n_envs)[:, None], actions]

    # shape: (n_envs, 1)
    max_means = jnp.max(mu_vectors, axis=1).reshape(-1, 1)

    # shape: (n_envs, horizon)
    cum_max = jax.lax.cumsum(jnp.repeat(max_means, horizon, axis=1), axis=1)
    cum_rewards = jax.lax.cumsum(means, axis=1)  # shape: (n_envs, horizon)

    cum_regret = cum_max - cum_rewards
    return cum_regret.mean(axis=0)
