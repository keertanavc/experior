import jax 
from chex._src.pytypes import PRNGKey
from gymnax.environments import ContinuousMountainCar as CMC, EnvParams
from .env import Environment

class ContinuousMountainCar(CMC, Environment):
    def init_env(self, key: PRNGKey, params: EnvParams) -> EnvParams:
        return params