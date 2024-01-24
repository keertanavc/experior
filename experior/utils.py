import optax
import jax
import chex

import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Callable, NamedTuple, Optional, Union

from flax import core
from flax import struct

def linear_schedule_eps(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)





# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (
            hasattr(key_or_seed, "shape")
            and (not key_or_seed.shape)
            and hasattr(key_or_seed, "dtype")
            and key_or_seed.dtype == jnp.int32
        ):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


def moving_average(data: jnp.array, window_size: int):
    """Smooth data by calculating the moving average over a specified window size."""
    return jnp.convolve(data, jnp.ones(window_size) / window_size, mode="valid")


def process_ppo_output(ppo_output, window=5000):
    r = ppo_output['metrics']['returned_episode_returns']
    # mean over all actors first
    avg_over_actors = r.mean(-1)
    
    # then mean + std performance over various parallel environments 
    r_mean = moving_average(avg_over_actors.mean(-1).reshape(-1), window)
    r_std = moving_average(avg_over_actors.std(-1).reshape(-1), window)
    
    return r_mean, r_std
    

class VecTrainState(struct.PyTreeNode):
    """Train state to handle parallel updates."""

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = jax.vmap(self.tx.update)(
            grads, self.opt_state, self.params
        )
        new_params = jax.vmap(optax.apply_updates)(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = jax.vmap(tx.init)(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
