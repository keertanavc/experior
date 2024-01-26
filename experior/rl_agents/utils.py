import optax
import jax
import chex
import distrax

import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from flax.linen.initializers import constant, orthogonal
from typing import Sequence
from typing import Union, Callable, Optional


class QNetwork(nn.Module):
    n_actions: int
    n_hidden: int
    n_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, unobs_context: jnp.ndarray = None):
        if unobs_context is None:
            unobs_context = jnp.ones((x.shape[0], self.n_features))
        assert unobs_context.shape[-1] == self.n_features
        if len(unobs_context.shape) == 2:
            assert unobs_context.shape[0] == x.shape[0]
        elif len(unobs_context.shape) > 2:
            raise NotImplementedError

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_features)(x)
        x = jnp.multiply(x, unobs_context)  # TODO maybe add some nonlinearity here
        x = nn.Dense(self.n_actions)(x)
        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise NotImplementedError
        x = x.reshape((x.shape[0], -1))
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ContinuousActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise NotImplementedError
        # x = x.reshape((x.shape[0], -1))

        # Actor features
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Actor output
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic features
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)

        # Critic output
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# adam slgd optimizer
def adam_lmc(
    b1: float = 0.9,
    b2: float = 0.999,
    bias_factor: float = 1.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the Adam LMC algorithm.

    References:
      [Ishfaq et al, 2023](https://arxiv.org/pdf/2305.18246.pdf)

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
      A `GradientTransformation` object.
    """
    from optax._src.utils import canonicalize_dtype, cast_tree

    mu_dtype = canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.update_moment(updates, state.mu, b1, 1)
        nu = optax.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_int32_increment(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, m, v: g + bias_factor * m / (jnp.sqrt(v + eps_root) + eps),
            updates,
            mu,
            nu,
        )
        mu = cast_tree(mu, mu_dtype)
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def add_slgd_noise(
    learning_rate, temperature, rng_key: chex.PRNGKey
) -> optax.GradientTransformation:
    learning_rate_fn = (
        lambda count: learning_rate(count) if callable(learning_rate) else learning_rate
    )
    temp_fn = lambda count: temperature(count) if callable(temperature) else temperature

    def init_fn(params):
        del params
        return optax.AddNoiseState(count=jnp.zeros([], jnp.int32), rng_key=rng_key)

    def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)
        count_inc = optax.safe_int32_increment(state.count)
        lr = learning_rate_fn(count_inc)
        temp = temp_fn(count_inc)
        variance = 2 * lr * temp
        standard_deviation = jnp.sqrt(variance)
        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates,
            jax.tree_util.tree_unflatten(treedef, all_keys[1:]),
        )
        updates = jax.tree_util.tree_map(
            lambda g, n: g + standard_deviation.astype(g.dtype) * n, updates, noise
        )
        return updates, optax.AddNoiseState(count=count_inc, rng_key=all_keys[0])

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(learning_rate: optax.ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def adam_slgd(
    learning_rate: Union[float, Callable[[int], float]],
    temperature: Union[float, Callable[[int], float]],
    rng_key: chex.PRNGKey,
    b1: float = 0.9,
    b2: float = 0.999,
    bias_factor: float = 1.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> optax.GradientTransformation:
    adam = (
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        )
        if bias_factor <= 0.0
        else adam_lmc(
            b1=b1,
            b2=b2,
            bias_factor=bias_factor,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        )
    )
    return optax.chain(
        adam,
        scale_by_learning_rate(learning_rate),
        add_slgd_noise(learning_rate, temperature, rng_key),
    )
