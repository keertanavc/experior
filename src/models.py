import jax

import flax.linen as nn
import jax.numpy as jnp
import jax.scipy.stats as jstats

from functools import partial

from src.configs import TransformerPolicyConfig
from src.commons import TransformerBlock


class BetaPrior(nn.Module):
    """A beta prior distribution over arm rewards for a Bernoulli bandit."""

    num_actions: int
    epsilon: float = 1e-3
    init_range: float = 5

    def setup(self):
        self.alphas_sq = self.param('alphas_sq', lambda rng, x: self.init_range * jax.random.uniform(
            rng) * jnp.ones(x), (self.num_actions,))  # TODO initialization
        self.betas_sq = self.param(
            'betas_sq', lambda rng, x: self.init_range * jax.random.uniform(rng) * jnp.ones(x), (self.num_actions,))

    def log_prob(self, mu):
        """Returns the log probability of a given mean vector."""
        alphas = jnp.power(self.alphas_sq, 2) + self.epsilon
        betas = jnp.power(self.betas_sq, 2) + self.epsilon
        return jstats.beta.logpdf(mu, alphas, betas)

    def __call__(self, mu):
        return self.log_prob(mu)

    def sample(self, rng_key, size):
        """Returns a sample from the prior."""
        alphas = self.alphas_sq ** 2 + self.epsilon
        betas = self.betas_sq ** 2 + self.epsilon
        return jax.random.beta(rng_key, a=alphas, b=betas, shape=(size, self.num_actions))


class Policy(nn.Module):
    """A policy that takes the history of actions and rewards as input and outputs a probability distribution
    over actions. Inspired by:
    https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
    """

    config: TransformerPolicyConfig

    @nn.compact
    def __call__(self, timesteps, actions, rewards):
        """Returns the log-probability distribution over actions for a given history of steps.

        Args:
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        B, T = timesteps.shape
        assert T <= self.config.horizon, f"Expected a history of at most {self.config.horizon} steps, got {T}"

        # shape: (B, T, h_dim)
        time_embedding = nn.Embed(num_embeddings=self.config.horizon,
                                  features=self.config.h_dim, dtype=self.config.dtype)(timesteps)

        action_embedding = nn.Embed(num_embeddings=self.config.num_actions,
                                    features=self.config.h_dim, dtype=self.config.dtype)(actions) + time_embedding

        reward_embedding = nn.Dense(
            features=self.config.h_dim, dtype=self.config.dtype)(rewards[..., jnp.newaxis]) + time_embedding

        # sequence of (r0, a0, r1, a1, ...)
        h = jnp.stack([reward_embedding, action_embedding],
                      axis=2).reshape(B, T * 2, self.config.h_dim)

        class_token = self.param('class_token', nn.initializers.normal(
            stddev=1e-6), (1, 1, self.config.h_dim))
        class_token = jnp.tile(class_token, (B, 1, 1))
        # shape: (B, T * 2 + 1, h_dim)
        h = jnp.concatenate([class_token, h], axis=1)

        h = nn.LayerNorm(dtype=self.config.dtype)(h)

        h = nn.Sequential([TransformerBlock(h_dim=self.config.h_dim,
                                            num_heads=self.config.num_heads,
                                            drop_p=self.config.drop_p,
                                            dtype=self.config.dtype) for _ in range(self.config.n_blocks)])(h)

        h = h[:, 0].reshape(B, self.config.h_dim)
        action_logits = nn.Dense(
            features=self.config.num_actions, dtype=self.config.dtype)(h)
        log_probs = nn.log_softmax(action_logits)
        return log_probs  # shape: (B, num_actions)


@partial(jax.jit, static_argnames=('policy_fn', 'horizon'))
def policy_rollout(policy_fn, rng_key, mu_vectors, horizon):
    """Returns a rollout of the policy, i.e. a sequence of actions and rewards.

    Args:
      policy_fn: A function that takes timesteps, actions, rewards and returns
        a log-probability distribution over actions.
      rng_key: A JAX random key.
      mu_vectors: Samples from the prior over means, shape (n_samples, num_actions).
      horizon: The length of the rollout.

    # TODO reduce the variance by outputting the means and log_probs of all the actions.
    # TODO optimize more
    Returns:
        actions: The sequence of actions, shape (n_samples, horizon).
        rewards: The sequence of rewards, shape (n_samples, horizon).
        log_probs: The log-probabilities of the taken actions, shape (n_samples, horizon).
    """
    def rollout_1d(mu, key):
        # mu shape: (num_actions,)
        def policy_step(state_input, _):
            i, time_steps, actions, rewards, rng = state_input

            # t = jax.lax.dynamic_slice(time_steps, [0], [i+1])
            # a = jax.lax.dynamic_slice(actions, [0], [i+1])
            # r = jax.lax.dynamic_slice(rewards, [0], [i+1])
            # shape: (num_actions,)
            log_prob = policy_fn(time_steps, actions, rewards)
            rng, key = jax.random.split(rng)
            new_action = jax.random.categorical(key, log_prob)

            rng, key = jax.random.split(rng)
            new_reward = jax.random.bernoulli(key, mu[new_action])

            actions = actions.at[i+1].set(new_action)
            rewards = rewards.at[i+1].set(new_reward)
            carry = (i + 1, time_steps, actions, rewards, rng)
            return carry, log_prob[new_action]

        # TODO the first step is considered 0 with zero action and reward
        init_val = (-1,
                    jnp.arange(horizon),
                    jnp.zeros(horizon, dtype=jnp.int32),
                    jnp.zeros(horizon, dtype=jnp.float32),
                    key)

        (_, _, actions, rewards, _), log_probs = jax.lax.scan(
            policy_step, init_val, (), length=horizon)
        return actions, rewards, log_probs

    n_samples = mu_vectors.shape[0]
    rng_keys = jax.random.split(rng_key, n_samples)
    return jax.vmap(rollout_1d, in_axes=(0, 0), out_axes=(0, 0, 0))(mu_vectors, rng_keys)

    # means = mu_vectors[jnp.arange(n_samples)[:, None], actions] # shape: (n_samples, horizon)
