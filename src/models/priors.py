import jax

import flax.linen as nn
import jax.numpy as jnp
import jax.scipy.stats as jstats

from flax.training import train_state

from src.configs import ExperiorConfig

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


class UniformPrior(Prior):
    """A uniform prior distribution over arm rewards for a Bernoulli bandit."""

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def sample(self, rng_key, size):
        return jax.random.uniform(rng_key, shape=(size, self.num_actions))


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
