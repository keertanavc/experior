import optax
import chex

import flax.linen as nn
import jax.numpy as jnp

from typing import Callable, Sequence
from flax.training import train_state

from pydantic import BaseModel as BaseModel


class BaseConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class MLP(nn.Module):
    features: Sequence[int]
    kernel_init: Callable[
        [chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array
    ] = nn.initializers.xavier_uniform()
    bias_init: Callable[
        [chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array
    ] = nn.initializers.normal(stddev=1e-6)
    activation: str = "relu"
    dtype: chex.ArrayDType = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "gelu":
            activation_fn = nn.gelu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Expected a valid activation, got {self.activation}")
        for i, feat in enumerate(self.features):
            x = nn.Dense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
            )(x)
            if i != len(self.features) - 1:
                x = activation_fn(x)
        return x


class TransformerBlock(nn.Module):
    h_dim: int
    num_heads: int
    dtype: chex.ArrayDType
    drop_p: float

    @nn.compact
    def __call__(self, x, mask=None):  # x.shape = (batch_size, seq_len, h_dim)
        x = x + nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype)(
            x, mask=mask
        )
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = x + MLP(
            features=[4 * self.h_dim, self.h_dim], activation="gelu", dtype=self.dtype
        )(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


class TrainState(train_state.TrainState):
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
        if self.tx:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
            new_params = optax.apply_updates(self.params, updates)
        else:
            new_params = self.params
            new_opt_state = self.opt_state

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params) if tx else None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
