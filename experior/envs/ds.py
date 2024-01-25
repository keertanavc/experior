from chex._src.pytypes import PRNGKey
from gymnax.environments import EnvParams
from gymnax.environments.bsuite.deep_sea import DeepSea as DS
from .env import Environment


class DeepSea(DS, Environment):
    def init_env(self, key: PRNGKey, params: EnvParams) -> EnvParams:
        return params  # TODO: fix this
