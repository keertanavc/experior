import flax.linen as nn
from src.utils import PRNGKey, Shape, Dtype, Array
from typing import Callable, Sequence

import jax.numpy as jnp


class MLP(nn.Module):
    features: Sequence[int]
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], Array
    ] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )
    activation: str = "relu"
    dtype: Dtype = jnp.float32

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
    dtype: Dtype
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
