import jax

import flax.linen as nn
import jax.numpy as jnp
import jax.scipy.stats as jstats

from flax.training import train_state

from src.configs import ExperiorConfig
from src.commons import Array

from abc import ABC, abstractmethod
from typing import Callable

##################### Priors #####################


def get_prior(conf: ExperiorConfig):
    if conf.prior.name == "beta":
        return BetaPrior
    elif conf.prior.name == "MaxEnt":
        return MaxEntPrior
    else:
        raise NotImplementedError


class Prior(ABC):
    @abstractmethod
    def sample(self, rng_key, size):
        """Returns a sample from the prior."""
        pass


class UniformPrior(Prior):
    """A uniform prior distribution over arm rewards for a Bernoulli bandit."""

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def sample(self, rng_key, size):
        return jax.random.uniform(rng_key, shape=(size, self.num_actions))


class MaxEntPrior(nn.Module, Prior):
    """A maximum entropy prior distribution over arm rewards for a Bernoulli bandit."""

    config: ExperiorConfig
    expert_policy: Array  # expert distribution, shape: (num_actions,)

    def setup(self):
        self.ref_dist = self.config.prior.ref_dist
        self.lambdas = self.param(
            "lambdas",
            nn.initializers.lecun_normal(),
            (self.config.prior.num_actions, 1),
        )

    def unnormalized_prob(self, mu):
        """Returns the unnormalized probability of a given mean vector.

        Args:
            mu: A batch of mean vectors, shape (batch_size, num_actions).

        Returns:
            The unnormalized probability of each mean vector, shape (batch_size,).
        """
        # shape: (batch_size, num_actions)
        best_act = jnp.eye(self.config.prior.num_actions)[jnp.argmax(mu, axis=1)]
        I_max = best_act - jnp.array(self.expert_policy).reshape(
            1, self.config.prior.num_actions
        )
        return jnp.exp(-I_max @ self.lambdas).reshape(
            -1,
        )

    def __call__(self, mu):
        return self.unnormalized_prob(mu)

    def sample(self, rng_key, size):
        """Returns a sample from the reference distribution."""
        return self.ref_dist(rng_key, (size, self.config.prior.num_actions))

    @classmethod
    def create_state(
        cls, rng_key, optimizer, conf: ExperiorConfig, expert_policy: Array
    ) -> train_state.TrainState:
        """Returns an initial state for the prior."""
        prior_model = cls(config=conf, expert_policy=expert_policy)
        variables = prior_model.init(rng_key, jnp.ones((1, conf.prior.num_actions)))

        prior_state = train_state.TrainState.create(
            apply_fn=prior_model.apply, params=variables["params"], tx=optimizer
        )

        return prior_state


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
        eps = self.config.prior.epsilon
        alphas = jnp.power(self.alphas_sq, 2) + eps
        betas = jnp.power(self.betas_sq, 2) + eps

        # clip mu to avoid numerical issues
        mu = jnp.clip(mu, eps, 1.0 - eps)
        return jstats.beta.logpdf(mu, alphas, betas).sum(axis=1)

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
