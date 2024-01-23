from chex._src.pytypes import PRNGKey
from gymnax.environments import CartPole as Cartp, EnvParams
from .env import Environment


class CartPole(Cartp, Environment):
    def init_env(self, key: PRNGKey, params: EnvParams) -> EnvParams:
        return params  # TODO: fix this
