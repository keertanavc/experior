import jax

import flax.linen as nn
import jax.numpy as jnp
import jax.scipy.stats as jstats

from flax.training import train_state

from src.configs import ExperiorConfig
from src.commons import TransformerBlock

from abc import ABC, abstractmethod


##################### Priors #####################
def get_prior(conf: ExperiorConfig):
    if conf.prior.name == "beta":
        return BetaPrior
    else:
        raise NotImplementedError


class Prior(ABC):
    @abstractmethod
    def sample(self, rng_key, size):
        """Returns a sample from the prior."""
        pass


class BetaPrior(nn.Module, Prior):
    """A beta prior distribution over arm rewards for a Bernoulli bandit."""

    config: ExperiorConfig

    def setup(self):
        if self.config.prior.init_alpha is None:

            def alpha_init_fn(rng, shape):
                return 5.0 * jax.random.uniform(rng) * jnp.ones(shape)

        else:

            def alpha_init_fn(rng, shape):
                return self.config.prior.init_alpha * jnp.ones(shape)

        if self.config.prior.init_beta is None:

            def beta_init_fn(rng, shape):
                return 5.0 * jax.random.uniform(rng) * jnp.ones(shape)

        else:

            def beta_init_fn(rng, shape):
                return self.config.prior.init_beta * jnp.ones(shape)

        self.alphas_sq = self.param(
            "alphas_sq", alpha_init_fn, (self.config.prior.num_actions,)
        )
        self.betas_sq = self.param(
            "betas_sq", beta_init_fn, (self.config.prior.num_actions,)
        )

    def log_prob(self, mu):
        """Returns the log probability of a given mean vector."""
        alphas = jnp.power(self.alphas_sq, 2) + self.config.prior.epsilon
        betas = jnp.power(self.betas_sq, 2) + self.config.prior.epsilon
        return jstats.beta.logpdf(mu, alphas, betas)

    def __call__(self, mu):
        return self.log_prob(mu)

    def sample(self, rng_key, size):
        """Returns a sample from the prior."""
        alphas = self.alphas_sq**2 + self.config.prior.epsilon
        betas = self.betas_sq**2 + self.config.prior.epsilon
        return jax.random.beta(
            rng_key, a=alphas, b=betas, shape=(size, self.config.prior.num_actions)
        )

    @classmethod
    def create_state(
        cls, rng_key, optimizer, conf: ExperiorConfig
    ) -> train_state.TrainState:
        """Returns an initial state for the prior."""
        prior_model = cls(config=conf)
        variables = prior_model.init(rng_key, jnp.ones((1, conf.prior.num_actions)))

        prior_state = train_state.TrainState.create(
            apply_fn=prior_model.apply, params=variables["params"], tx=optimizer
        )

        return prior_state


##################### Policies #####################
def get_policy(conf: ExperiorConfig):
    if conf.policy.name == "transformer":
        return TransformerPolicy
    elif conf.policy.name == "softelim":
        return SoftElimPolicy
    else:
        raise NotImplementedError


class Policy(ABC):
    @abstractmethod
    def __call__(self, rng_key, timesteps, actions, rewards):
        """Returns the log-probability distribution over actions for a given history of steps.
        Note: It gets the history over all the horizon, but it should ignore the steps with zero timesteps.

        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        pass


class SoftElimPolicy(nn.Module, Policy):
    """The SoftElim policy in https://arxiv.org/pdf/2006.05094.pdf"""

    config: ExperiorConfig
    name = "SoftElimPolicy"

    def setup(self):
        self.w = self.param(
            "w",
            lambda rng, shape: jnp.ones(shape),
            (self.config.prior.num_actions,),
        )

    def __call__(self, rng_key, timesteps, actions, rewards):
        """
        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        max_t = timesteps.max(axis=1).reshape(-1, 1)  # shape: (batch_size, 1)
        num_actions = self.config.prior.num_actions
        # TODO check if the maximum value of timesteps is less than the max_horizon

        # shape: (batch_size, T)
        idx = (timesteps[:, :] <= max_t) & (timesteps[:, :] > 0)

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
        means = jnp.nan_to_num(jnp.sum(aug_rewards, axis=1) / action_counts)
        S = 2 * (means.max(axis=1).reshape(-1, 1) - means) ** 2 * action_counts

        return nn.log_softmax(-S / (self.w**2))

    @classmethod
    def create_state(
        cls, rng_key, optimizer, conf: ExperiorConfig
    ) -> train_state.TrainState:
        """Returns an initial state for the policy."""
        policy = cls(config=conf)
        key1, key2 = jax.random.split(rng_key)
        variables = policy.init(
            key1,
            key2,
            jnp.ones((1, 2), dtype=jnp.int32),
            jnp.ones((1, 2), dtype=jnp.int32),
            jnp.ones((1, 2)),
        )

        policy_state = train_state.TrainState.create(
            apply_fn=policy.apply, params=variables["params"], tx=optimizer
        )

        return policy_state


class TransformerPolicy(nn.Module, Policy):
    """A policy that takes the history of actions and rewards as input and outputs a probability distribution
    over actions. Inspired by:
    https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
    """

    config: ExperiorConfig
    name = "TransformerPolicy"

    @nn.compact
    def __call__(self, rng_key, timesteps, actions, rewards):
        """Returns the log-probability distribution over actions for a given history of steps.
        # TODO add masking the inputs with timestep 0
        # TODO test_horizon instead of max horizon
        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        B, T = timesteps.shape
        # TODO check if the maximum value of timesteps is less than the max_horizon

        # shape: (B, T, h_dim)
        time_embedding = nn.Embed(
            num_embeddings=self.config.trainer.max_horizon + 1,
            features=self.config.policy.h_dim,
            dtype=self.config.policy.dtype,
        )(timesteps)

        action_embedding = (
            nn.Embed(
                num_embeddings=self.config.prior.num_actions,
                features=self.config.policy.h_dim,
                dtype=self.config.policy.dtype,
            )(actions)
            + time_embedding
        )

        reward_embedding = (
            nn.Dense(features=self.config.policy.h_dim, dtype=self.config.policy.dtype)(
                rewards[..., jnp.newaxis]
            )
            + time_embedding
        )

        # sequence of (r0, a0, r1, a1, ...)
        h = jnp.stack([reward_embedding, action_embedding], axis=2).reshape(
            B, T * 2, self.config.policy.h_dim
        )

        class_token = self.param(
            "class_token",
            nn.initializers.normal(stddev=1e-6),
            (1, 1, self.config.policy.h_dim),
        )
        class_token = jnp.tile(class_token, (B, 1, 1))
        # shape: (B, T * 2 + 1, h_dim)
        h = jnp.concatenate([class_token, h], axis=1)

        h = nn.LayerNorm(dtype=self.config.policy.dtype)(h)

        h = nn.Sequential(
            [
                TransformerBlock(
                    h_dim=self.config.policy.h_dim,
                    num_heads=self.config.policy.num_heads,
                    drop_p=self.config.policy.drop_p,
                    dtype=self.config.policy.dtype,
                )
                for _ in range(self.config.policy.n_blocks)
            ]
        )(h)

        h = h[:, 0].reshape(B, self.config.policy.h_dim)
        action_logits = nn.Dense(
            features=self.config.prior.num_actions, dtype=self.config.policy.dtype
        )(h)
        log_probs = nn.log_softmax(action_logits)
        return log_probs  # shape: (B, num_actions)

    @classmethod
    def create_state(
        cls, rng_key, optimizer, conf: ExperiorConfig
    ) -> train_state.TrainState:
        """Returns an initial state for the policy."""
        policy = cls(config=conf)
        key1, key2 = jax.random.split(rng_key)
        variables = policy.init(
            key1,
            key2,
            jnp.ones((1, 2), dtype=jnp.int32),
            jnp.ones((1, 2), dtype=jnp.int32),
            jnp.ones((1, 2)),
        )

        policy_state = train_state.TrainState.create(
            apply_fn=policy.apply, params=variables["params"], tx=optimizer
        )

        return policy_state
