import jax

import jax.numpy as jnp

from src.configs import ExperiorConfig
from abc import ABC, abstractmethod


##################### Gradient Estimation Baselines #####################


class VarBaseline(ABC):
    @abstractmethod
    def __call__(self, actions, rewards, mu_vectors):
        """
        Args:
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).
            mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).
        """
        pass


class ZeroBaseline(VarBaseline):
    def __call__(self, actions, rewards, mu_vectors):
        """
        Args:
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).
            mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).
        """

        return jnp.zeros_like(actions, dtype=jnp.float32)


class OptBaseline(VarBaseline):
    def __call__(self, actions, rewards, mu_vectors):
        """Returns the baseline opt for the variance reduction.
        See https://arxiv.org/pdf/2006.05094.pdf

        Args:
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).
            mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).

        Returns:
            The baseline opt, shape (batch_size, T).
        """

        max_means = jnp.max(mu_vectors, axis=1, keepdims=True)  # shape: (n_samples, 1)

        T = actions.shape[1]

        # shape: (batch_size, T)
        return jax.lax.cumsum(max_means.repeat(T, axis=1), axis=1, reverse=True)


def get_var_baseline(conf: ExperiorConfig) -> VarBaseline:
    if conf.trainer.policy_grad.var_baseline == "opt":
        return OptBaseline()
    elif conf.trainer.policy_grad.var_baseline is None:
        return ZeroBaseline()
    else:
        raise NotImplementedError


##################### Policy Gradient Estimation #####################


class PolicyGradEstimator:
    def __init__(self, conf: ExperiorConfig):
        self.baseline = get_var_baseline(conf)

    def __call__(self, actions, rewards, log_probs, mu_vector):
        """
        Args:
            actions: The history of actions, shape (batch_size, T).
            rewards: The history of rewards, shape (batch_size, T).
            log_probs: The log-probabilities of all actions, shape (batch_size, T, num_actions).
            mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).
        """
        raise NotImplementedError


class Reinforce(PolicyGradEstimator):
    def __init__(self, conf: ExperiorConfig):
        super().__init__(conf)

    def __call__(self, actions, rewards, log_probs, mu_vectors):
        """Returns the REINFORCE gradient estimator."""

        n_samples = actions.shape[0]
        num_actions = actions.shape[1]

        # shape: (n_samples, horizon)
        means = mu_vectors[jnp.arange(n_samples)[:, None], actions]

        # shape: (n_samples, horizon)
        rtg = jax.lax.cumsum(means, axis=1, reverse=True)

        # baseline for variance reduction, shape: (n_samples, horizon)
        baseline = self.baseline(actions, rewards, mu_vectors)

        i, j = jnp.meshgrid(
            jnp.arange(n_samples), jnp.arange(num_actions), indexing="ij"
        )
        policy_log = log_probs[i, j, actions]  # shape: (n_samples, horizon)
        loss = -(jax.lax.stop_gradient(rtg - baseline) * policy_log).sum(axis=1).mean()
        true_loss_val = -(rtg * policy_log).sum(axis=1).mean()

        return loss, jax.lax.stop_gradient(true_loss_val)


class Binforce(PolicyGradEstimator):
    def __init__(self, conf: ExperiorConfig):
        super().__init__(conf)

    def __call__(self, actions, rewards, log_probs, mu_vectors):
        """Returns the biased REINFORCE gradient estimator."""

        n_samples = actions.shape[0]

        # shape: (n_samples, horizon)
        means = mu_vectors[jnp.arange(n_samples)[:, None], actions]

        # shape: (n_samples, horizon)
        rtg = jax.lax.cumsum(means, axis=1, reverse=True)

        # baseline for variance reduction, shape: (n_samples, horizon)
        baseline = self.baseline(actions, rewards, mu_vectors)

        # shape: (n_samples, horizon, num_actions)
        action_rtg = (rtg - means)[:, :, None] + mu_vectors[:, None, :]
        policy_probs = jax.lax.stop_gradient(jnp.exp(log_probs))
        loss = (
            -(
                jax.lax.stop_gradient(action_rtg - baseline[:, :, None])
                * log_probs
                * policy_probs
            )
            .sum(axis=2)
            .sum(axis=1)
            .mean()
        )

        true_loss_val = (
            -(action_rtg * log_probs * policy_probs).sum(axis=2).sum(axis=1).mean()
        )

        return loss, jax.lax.stop_gradient(true_loss_val)


def get_policy_grad_estimator(conf: ExperiorConfig) -> PolicyGradEstimator:
    if conf.trainer.policy_grad.name == "reinforce":
        return Reinforce(conf)
    elif conf.trainer.policy_grad.name == "binforce":
        return Binforce(conf)
    else:
        raise NotImplementedError
