from typing import Tuple

import jax.numpy as jnp
import jax

from flax.training import train_state

from typing import Any
import optax

PRNGKey = Any
Params = Any
Variables = Any
OptState = optax.OptState

Shape = Tuple[int, ...]
Dtype = Any
Array = Any

    
# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
              hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


class TrainState(train_state.TrainState):
    stats: Variables
    init_stats: Variables
    init_params: Params

    def init_opt_state(self):  # Initializes the optimizer state. TODO make sure it's correct
        new_opt_state = self.tx.init(self.params)
        return self.replace(opt_state=new_opt_state)

    def init_param_state(self):
        return self.replace(stats=self.init_stats, params=self.init_params)