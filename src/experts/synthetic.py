import jax
import chex
import jax.numpy as jnp

from abc import ABC, abstractmethod

from src.configs import SyntheticExpertConfig
from src.models import get_prior


class Expert(ABC):
    @abstractmethod
    def policy(self, rng: chex.PRNGKey):
        """Returns the expert policy, shape (num_actions, )"""
        pass

    def __call__(self, rng: chex.PRNGKey):
        return self.policy(rng)


class SyntheticExpert(Expert):
    def __init__(self, conf: SyntheticExpertConfig):
        self.conf = conf
        rng = jax.random.PRNGKey(42)
        self.prior_state = get_prior(conf.prior.name).create_state(
            rng, None, conf.prior
        )

    def policy(self, rng: chex.PRNGKey):
        mu_vectors = self.prior_state.apply_fn(
            {"params": self.prior_state.params},
            rng,
            self.conf.mc_samples,
            method="sample",
        )

        # shape: (n_samples, num_actions)
        opt_actions = jnp.eye(self.conf.prior.num_actions)[
            jnp.argmax(mu_vectors, axis=1)
        ]

        return opt_actions.mean(axis=0)

    def __call__(self, rng: chex.PRNGKey):
        return self.policy(rng)
