import jax
import jax.numpy as jnp
from src.configs import TransformerPolicyConfig
from src.models import Policy
from src.baselines import thompson_sampling

import numpy as np


def uniform_bayes_regret(rng_key, policy_actions, num_actions, horizon, n_envs):
    key1, key2 = jax.random.split(rng_key)
    mu_vectors = jax.random.uniform(
        key1, shape=(n_envs, num_actions))

    # actions, rewards, log_policy_probs = policy_state.apply_fn({'params': policy_state.params},
    #                                                            key2,
    #                                                            mu_vectors,
    #                                                            method=Policy.policy_rollout)

    means = mu_vectors[jnp.arange(n_envs)[:, None], policy_actions]
    max_means = jnp.max(mu_vectors, axis=1).reshape(-1,
                                                    1)  # shape: (n_envs, 1)
    cum_max = jax.lax.cumsum(jnp.repeat(
        max_means, horizon, axis=1), axis=1)  # shape: (n_envs, horizon)
    cum_rewards = jax.lax.cumsum(means, axis=1)  # shape: (n_envs, horizon)

    cum_regret = cum_max - cum_rewards
    return cum_regret.mean(axis=0)



def uniform_bayes_regret_ts(rng_key, num_actions, horizon, n_envs):
    rng_key, key = jax.random.split(rng_key)
    mu_vectors = jax.random.uniform(
        key, shape=(n_envs, num_actions))

    # actions, rewards, log_policy_probs = policy_state.apply_fn({'params': policy_state.params},
    #                                                            key2,
    #                                                            mu_vectors,
    #                                                            method=Policy.policy_rollout)
    actions = np.zeros((n_envs, horizon))
    for i in range(n_envs):
        rng_key, key = jax.random.split(rng_key)
        a, _ = thompson_sampling(mu_vectors[i], horizon, rng_key)
        actions[i, :] = a

    actions = np.asarray(actions).astype(int)
    mu_vectors = np.asarray(mu_vectors)
    means = mu_vectors[np.arange(n_envs)[:, None], actions]
    max_means = jnp.max(mu_vectors, axis=1).reshape(-1,
                                                    1)  # shape: (n_envs, 1)
    cum_max = jax.lax.cumsum(jnp.repeat(
        max_means, horizon, axis=1), axis=1)  # shape: (n_envs, horizon)
    cum_rewards = jax.lax.cumsum(means, axis=1)  # shape: (n_envs, horizon)

    cum_regret = cum_max - cum_rewards
    return cum_regret.mean(axis=0)
