import jax

import flax.linen as nn
import jax.numpy as jnp

from flax.training import train_state

from src.configs import ExperiorConfig
from src.commons import TransformerBlock

from abc import ABC, abstractmethod


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
        div_action_cnt = jnp.where(
            action_counts == 0, jnp.ones_like(action_counts), action_counts
        )
        means = jnp.sum(aug_rewards, axis=1) / div_action_cnt
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
        Args:
            rng_key: A JAX random key.
            timesteps: The history of timesteps, shape (batch_size, T).
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).

        """
        B, T = timesteps.shape

        # TODO check if the maximum value of timesteps is less than the max_horizon
        max_horizon = max(
            self.config.trainer.test_horizon, self.config.trainer.train_horizon
        )

        # shape: (B, T, h_dim)
        time_embedding = nn.Embed(
            num_embeddings=max_horizon + 1,
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
