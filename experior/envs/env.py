import chex

from gymnax.environments import environment, EnvParams, EnvState
from typing import Union


class Environment(environment.Environment):
    def init_env(self, key: chex.PRNGKey, params: EnvParams) -> EnvParams:
        """Initialize environment state."""
        raise NotImplementedError

    def optimal_policy(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> Union[int, float, chex.Array]:
        raise NotImplementedError
