import jax

import flax.linen as nn
import jax.numpy as jnp
import jax.scipy.stats as jstats

from flax.training import train_state

from src.configs import TransformerPolicyConfig, BetaPriorConfig
from src.commons import TransformerBlock


class BetaPrior(nn.Module):
    """A beta prior distribution over arm rewards for a Bernoulli bandit."""

    config: BetaPriorConfig
    epsilon: float = 1e-3

    def setup(self):
        if self.config.init_alpha is None:
            alpha_init_fn = lambda rng, shape: 5. * jax.random.uniform(rng) * jnp.ones(shape)
        else:
            alpha_init_fn = lambda rng, shape: self.config.init_alpha * jnp.ones(shape)

        if self.config.init_beta is None:
            beta_init_fn = lambda rng, shape: 5. * jax.random.uniform(rng) * jnp.ones(shape)
        else:
            beta_init_fn = lambda rng, shape: self.config.init_beta * jnp.ones(shape)

        self.alphas_sq = self.param('alphas_sq', alpha_init_fn, (self.config.num_actions,))
        self.betas_sq = self.param('betas_sq', beta_init_fn, (self.config.num_actions,))

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
        return jax.random.beta(rng_key, a=alphas, b=betas,
                               shape=(size, self.config.num_actions))

    @classmethod
    def create_state(cls, rng_key, optimizer, conf: BetaPriorConfig) -> train_state.TrainState:
        """Returns an initial state for the prior."""
        prior_model = cls(config=conf)
        variables = prior_model.init(
            rng_key, jnp.ones((1, conf.num_actions)))

        prior_state = train_state.TrainState.create(
            apply_fn=prior_model.apply, params=variables['params'], tx=optimizer)

        return prior_state


class Policy(nn.Module):
    """A policy that takes the history of actions and rewards as input and outputs a probability distribution
    over actions. Inspired by:
    https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
    """

    config: TransformerPolicyConfig

    @nn.compact
    def __call__(self, rng_key, timesteps, actions, rewards):
        """Returns the log-probability distribution over actions for a given history of steps.

        Args:
            rng_key: A JAX random key.
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

    @classmethod
    def create_state(cls, rng_key, optimizer,
                     conf: TransformerPolicyConfig) -> train_state.TrainState:
        """Returns an initial state for the policy."""
        policy = cls(config=conf)
        key1, key2 = jax.random.split(rng_key)
        variables = policy.init(key1, key2, jnp.ones((1, 2), dtype=jnp.int32),
                                jnp.ones((1, 2), dtype=jnp.int32), jnp.ones((1, 2)))

        policy_state = train_state.TrainState.create(
            apply_fn=policy.apply, params=variables['params'], tx=optimizer)

        return policy_state
