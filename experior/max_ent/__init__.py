from typing import Any, Callable

from flax import core
from flax import struct

import optax
import chex
import jax

import jax.numpy as jnp


class MaxEntTrainState(struct.PyTreeNode):
    step: int
    lambda_: float
    epsilon: float
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    log_prior_fn: Callable = struct.field(pytree_node=False)
    scaling_factor: float = 0.0
    max_param_value: float = 10.0

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = jax.vmap(self.tx.update)(
            grads, self.opt_state, self.params
        )
        new_params = jax.vmap(optax.apply_updates)(self.params, updates)

        # TODO clip the new params
        new_params = jax.tree_util.tree_map(
            lambda p: jnp.clip(p, -self.max_param_value, self.max_param_value),
            new_params,
        )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    # def reset_opt_state(self):
    #     return self.replace(opt_state=jax.vmap(self.tx.init)(self.params))

    @classmethod
    def create(cls, *, rng, n_trajectory, lambda_, epsilon, tx, num_envs, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        params = jax.random.normal(rng, (num_envs, n_trajectory))
        opt_state = jax.vmap(tx.init)(params)

        def log_prior_fn(params, traj_log_likelihoods):
            # TODO makes sure the scale is not too large
            log_prior = jnp.einsum(
                "ep, p->e",
                jnp.exp(params - jnp.log(n_trajectory)),
                jnp.exp(traj_log_likelihoods),
            )
            return log_prior

        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            lambda_=lambda_,
            epsilon=epsilon,
            log_prior_fn=log_prior_fn,
            **kwargs,
        )

    # TODO init emp entropy
    def make_max_ent_update_step(
        self, sampled_log_likelihoods: chex.Array, init_emp_ent: float = 0.0
    ):
        # sampled_log_likelihoods shape: (num_envs, prior_n_samples, n_trajectory)
        def max_ent_update_step(state, _):
            def single_loss_fn(params, log_likelihoods):
                # log_likelihoods shape: (prior_n_samples, n_trajectory)
                # params shape: (n_trajectory,)
                n_trajectory = params.shape[0]
                alphas = jnp.exp(params)
                m_alpha = jnp.exp(log_likelihoods) @ alphas
                loss = (
                    -jax.scipy.special.logsumexp(
                        m_alpha, axis=0, b=1.0 / m_alpha.shape[0]
                    )
                    + self.lambda_ * jnp.log(alphas / self.lambda_).sum()
                    + n_trajectory
                    * self.lambda_
                    * (1 - self.epsilon - jnp.log(n_trajectory) + init_emp_ent)
                )
                return -loss.mean()

            multi_loss_fn = lambda params: jax.vmap(single_loss_fn)(
                params, sampled_log_likelihoods
            ).mean()
            loss, grad = jax.value_and_grad(multi_loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grad)
            return new_state, {"loss": loss}

        scaling_factor = jax.lax.stop_gradient(
            jnp.einsum(
                "ep, enp->en", jnp.exp(self.params), jnp.exp(sampled_log_likelihoods)
            ).mean(axis=-1)
        )
        return (
            self.replace(scaling_factor=scaling_factor),
            max_ent_update_step,
        )
