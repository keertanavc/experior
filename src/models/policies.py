import jax

import flax.linen as nn
import jax.numpy as jnp

from src.configs import (
    SoftElimPolicyConfig,
    TransformerPolicyConfig,
    BetaTSPolicyConfig,
)
from src.commons import TransformerBlock, TrainState
from .priors import BetaPrior

from abc import ABC, abstractmethod


##################### Policies #####################
def get_policy(name: str):
    if name == "transformer":
        return TransformerPolicy
    elif name == "softelim":
        return SoftElimPolicy
    elif name == "beta_ts":
        return BetaTSPolicy
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


class BetaTSPolicy(nn.Module, Policy):
    """A differentiable TS policy with Beta prior distribution."""

    config: BetaTSPolicyConfig
    name = "BetaTSPolicy"

    def setup(self):
        self.prior = BetaPrior(config=self.config.prior)

    def prior_log_p(self, mu_vectors):
        """Returns the log probability of the prior for a given mean vector.

        Args:
            mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).

        Returns:
            The log probability of the prior, shape (batch_size, ).
        """
        return self.prior.log_prob(mu_vectors)

    def prior_opt_policy(self, rng_key, size=1000):
        mu_vectors = self.prior.sample(rng_key, size)

        # shape: (n_samples, num_actions)
        opt_actions = jnp.eye(self.config.num_actions)[jnp.argmax(mu_vectors, axis=1)]

        return opt_actions.mean(axis=0)

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
        sum_rewards = jnp.sum(aug_rewards, axis=1)
        prior_alpha = (
            self.prior.alphas_sq.reshape(1, -1) ** 2 + self.config.prior.epsilon
        )
        prior_beta = self.prior.betas_sq.reshape(1, -1) ** 2 + self.config.prior.epsilon
        post_alpha = sum_rewards + prior_alpha
        post_beta = action_counts - sum_rewards + prior_beta

        sampled_theta = jax.random.beta(rng_key, post_alpha, post_beta)

        # shape: (batch_size, num_actions)
        action_probs = jnp.where(
            sampled_theta == jnp.max(sampled_theta, axis=1, keepdims=True), 1, 0
        )

        return jnp.log(action_probs)

    @classmethod
    def create_state(cls, rng_key, optimizer, conf: BetaTSPolicyConfig) -> TrainState:
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

        policy_state = TrainState.create(
            apply_fn=policy.apply, params=variables["params"], tx=optimizer
        )

        return policy_state


class SoftElimPolicy(nn.Module, Policy):
    """The SoftElim policy in https://arxiv.org/pdf/2006.05094.pdf"""

    config: SoftElimPolicyConfig
    name = "SoftElimPolicy"

    def setup(self):
        self.w = self.param(
            "w",
            lambda rng, shape: jnp.ones(shape),
            (self.config.num_actions,),
        )

        # TODO observation: it seems having a separate w for each action does not add much. why?

    def __call__(self, rng_key, timesteps, actions, rewards):
        """
        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        max_t = timesteps.max(axis=1).reshape(-1, 1)  # shape: (batch_size, 1)
        num_actions = self.config.num_actions

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
        div_action_cnt = jnp.where(
            action_counts == 0, jnp.ones_like(action_counts), action_counts
        )
        means = jnp.sum(aug_rewards, axis=1) / div_action_cnt

        # shape: (batch_size, num_actions)
        S = 2 * (means.max(axis=1).reshape(-1, 1) - means) ** 2 * action_counts

        return nn.log_softmax(-S / (self.w.reshape(1, -1) ** 2))

    @classmethod
    def create_state(cls, rng_key, optimizer, conf: SoftElimPolicyConfig) -> TrainState:
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

        policy_state = TrainState.create(
            apply_fn=policy.apply, params=variables["params"], tx=optimizer
        )

        return policy_state


class TransformerPolicy(nn.Module, Policy):
    """A policy that takes the history of actions and rewards as input and outputs a probability distribution
    over actions. Inspired by:
    https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
    """

    config: TransformerPolicyConfig
    name = "TransformerPolicy"

    @nn.compact
    def __call__(self, rng_key, timesteps, actions, rewards):
        """Returns the log-probability distribution over actions for a given history of steps.
        # TODO add masking the inputs with timestep 0
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
            num_embeddings=self.config.horizon + 1,
            features=self.config.h_dim,
            dtype=self.config.dtype,
        )(timesteps)

        action_embedding = (
            nn.Embed(
                num_embeddings=self.config.num_actions,
                features=self.config.h_dim,
                dtype=self.config.dtype,
            )(actions)
            + time_embedding
        )

        reward_embedding = (
            nn.Dense(features=self.config.h_dim, dtype=self.config.dtype)(
                rewards[..., jnp.newaxis]
            )
            + time_embedding
        )

        # sequence of (r0, a0, r1, a1, ...)
        h = jnp.stack([reward_embedding, action_embedding], axis=2).reshape(
            B, T * 2, self.config.h_dim
        )

        class_token = self.param(
            "class_token",
            nn.initializers.normal(stddev=1e-6),
            (1, 1, self.config.h_dim),
        )
        class_token = jnp.tile(class_token, (B, 1, 1))
        # shape: (B, T * 2 + 1, h_dim)
        h = jnp.concatenate([class_token, h], axis=1)

        h = nn.LayerNorm(dtype=self.config.dtype)(h)

        h = nn.Sequential(
            [
                TransformerBlock(
                    h_dim=self.config.h_dim,
                    num_heads=self.config.num_heads,
                    drop_p=self.config.drop_p,
                    dtype=self.config.dtype,
                )
                for _ in range(self.config.n_blocks)
            ]
        )(h)

        h = h[:, 0].reshape(B, self.config.h_dim)
        action_logits = nn.Dense(
            features=self.config.num_actions, dtype=self.config.dtype
        )(h)
        log_probs = nn.log_softmax(action_logits)
        return log_probs  # shape: (B, num_actions)

    @classmethod
    def create_state(
        cls, rng_key, optimizer, conf: TransformerPolicyConfig
    ) -> TrainState:
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

        policy_state = TrainState.create(
            apply_fn=policy.apply, params=variables["params"], tx=optimizer
        )

        return policy_state
