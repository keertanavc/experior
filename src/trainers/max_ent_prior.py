import chex
import jax
import optax
import jax.numpy as jnp

from typing import Any, Optional
from src.envs import *
from src.commons import BaseConfig
from src.experts import NoisyRationalExpertConfig, make_expert_log_likelihood_fn


class MaxEntropyPriorConfig(BaseConfig):
    name: str = "MaxEntropyPrior"
    env: Environment
    env_param: EnvParams
    epsilon: float
    lr: float
    max_grad_norm: float
    mc_samples: int
    expert_config: NoisyRationalExpertConfig
    epochs: int
    max_prior_param_value: Optional[float] = None
    lambda_: Optional[float] = None


def make_max_entropy_prior(conf: MaxEntropyPriorConfig):
    expert_log_likelihood_fn = make_expert_log_likelihood_fn(conf.expert_config)

    def _emp_entropy_rho(trajectory: Trajectory):
        # assuming standard normal distribution for the contexts
        # TODO handle horizon > 1 for RL and make it configurable
        return (
            -jax.scipy.stats.norm.logpdf(trajectory.context).sum()
            / trajectory.context.shape[0]
        )

    def max_entropy_train(rng: chex.PRNGKey, expert_trajectory: Trajectory):
        rng, rng_ = jax.random.split(rng)
        sampled_unobserved_contexts = conf.env_param.ref_prior_fn(rng_, conf.mc_samples)

        rng, rng_ = jax.random.split(rng)
        log_likelihood_fn = lambda unobserved_context: expert_log_likelihood_fn(
            rng_, unobserved_context, expert_trajectory
        )
        # prior_n_samples x n_trajectory
        traj_log_likelihoods = jax.lax.map(
            log_likelihood_fn, sampled_unobserved_contexts
        )

        emp_entropy = _emp_entropy_rho(expert_trajectory)
        n_trajectory = expert_trajectory.context.shape[0]

        rng, rng_ = jax.random.split(rng)
        params = {"log_alphas": jax.random.normal(rng_, (n_trajectory, 1))}
        tx = optax.chain(
            optax.clip_by_global_norm(conf.max_grad_norm), optax.adam(conf.lr, eps=1e-5)
        )
        opt_state = tx.init(params)

        def _update_step(runner_state, _):
            opt_state, params = runner_state

            def _loss_fn(params):
                alphas = jnp.exp(params["log_alphas"])
                m_alpha = jnp.exp(traj_log_likelihoods) @ alphas
                loss = (
                    -jax.scipy.special.logsumexp(
                        m_alpha, axis=0, b=1.0 / m_alpha.shape[0]
                    )
                    + conf.lambda_ * jnp.log(alphas / conf.lambda_).sum()
                    + n_trajectory
                    * conf.lambda_
                    * (1 - conf.epsilon - jnp.log(n_trajectory) + emp_entropy)
                )  # TODO fix
                return -loss.mean()

            loss, grad = jax.value_and_grad(_loss_fn)(params)
            updates, opt_state = tx.update(grad, opt_state)
            params = optax.apply_updates(params, updates)

            # constrain the params
            if conf.max_prior_param_value is not None:
                params = jax.tree_map(
                    lambda x: jnp.clip(
                        x, -conf.max_prior_param_value, conf.max_prior_param_value
                    ),
                    params,
                )
            return (opt_state, params), {"loss": loss}

        runner_state = (opt_state, params)
        runner_state, metrics = jax.lax.scan(
            _update_step,
            init=runner_state,
            xs=None,
            length=conf.epochs,
        )

        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "prior_params": runner_state[1],
        }

    def unnoramlized_log_prior(
        prior_params: Any,
        expert_trajectory: Trajectory,
        rng: chex.PRNGKey,
        unobserved_context: UnobservedContext,
    ):
        log_likelihoods = expert_log_likelihood_fn(
            rng, unobserved_context, expert_trajectory
        )
        return jnp.exp(log_likelihoods) @ jnp.exp(prior_params["log_alphas"])

    return max_entropy_train, unnoramlized_log_prior
