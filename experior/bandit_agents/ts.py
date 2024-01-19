import jax.numpy as jnp
import jax
import flax
import optax
import chex
import flashbax as fbx

from typing import Callable, Union, Dict

from experior.envs import Environment
from experior.utils import VecTrainState, QNetwork, adam_slgd
from experior.experts import Trajectory
from experior.experts.default_discrete_action import expert_log_likelihood_fn


def make_thompson_sampling(
    env: Environment,
    langevin_learning_rate: float,
    langevin_batch_size: float,
    num_envs: int,
    steps: int,
):
    def langevin_posterior_sampling(
        rng: chex.PRNGKey,
        last_param: chex.Array,
        trajectory: Trajectory,
        indices: chex.Array,
        log_prior_pdf: Callable[[chex.PRNGKey, chex.Array], chex.Array],
    ):
        # using https://proceedings.mlr.press/v119/mazumdar20a/mazumdar20a.pdf
        def _update_parameters(runner_state, _):
            rng, param, trajectory = runner_state
            rng, rng_ = jax.random.split(rng)
            # Compute the gradient of the unnormalized log prior
            grad_log_prior = jax.grad(
                lambda k, p: log_prior_pdf(k, p).sum(), argnums=(1)
            )(rng_, param)

            rng, rng_ = jax.random.split(rng)
            obs = trajectory.obs
            actions = trajectory.action
            rewards = trajectory.reward
            # Compute the gradient of the log likelihood for a batch of transitions that is non nan
            batch_indices = jax.random.bernoulli(
                rng_, p=langevin_batch_ratio, shape=(obs.shape[0],)
            )
            # TODO maybe better step size
            step_size = langevin_learning_rate / (indices.sum() + 1)
            ind = jnp.logical_and(batch_indices, indices).reshape(-1, 1)
            reward_log_likelihood_fn = (
                lambda p, c, r, a: -0.5 * (env_params.Q_function(p, c, a) - r) ** 2
            )

            def log_likelihood_fn(p):
                log_like = jax.vmap(reward_log_likelihood_fn, in_axes=(None, 0, 0, 0))(
                    p, obs, rewards, actions
                )
                return (log_like * ind).sum()

            grad_log_likelihood = jax.grad(log_likelihood_fn)(param)
            # Sum the gradients to get the gradient of the log posterior
            grad_log_posterior = (
                grad_log_prior + 1.0 / langevin_batch_ratio * grad_log_likelihood
            )

            # write a clip the gradient
            grad_log_posterior = jnp.clip(
                grad_log_posterior,
                -config["LANGEVIN_GRAD_CLIP"],
                config["LANGEVIN_GRAD_CLIP"],
            )

            # Langevin dynamics update rule
            rng, rng_ = jax.random.split(rng)
            noise = jax.random.normal(rng_, param.shape) * jnp.sqrt(2 * step_size)
            updated_param = param + step_size * grad_log_posterior + noise
            runner_state = (rng, updated_param, trajectory)
            return runner_state, None

        runner_state = (rng, last_param, trajectory)
        runner_state, _ = jax.lax.scan(
            _update_parameters, runner_state, None, config["LANGEVIN_NUM_STEPS"]
        )
        return runner_state[1]

    def train(rng, log_prior_fn):
        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None))(
            jax.random.split(rng_, num_envs), env.default_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(
            jax.random.split(rng_, num_envs), env_params
        )

        # replay buffer
        rng, rng_ = jax.random.split(rng)
        buffer = fbx.make_item_buffer(
            max_length=steps, min_length=1, sample_batch_size=langevin_batch_size
        )

        rng, rng_ = jax.random.split(rng)
        buffer_state = buffer.init(
            {
                "obs": obs,
                "action": jax.vmap(lambda k, p: env.action_space(p).sample(k))(
                    jax.random.split(rng_, num_envs), env_params
                ),
                "reward": jnp.zeros((num_envs,)),
            }
        )

        def _env_step(runner_state, i):
            env_state, rng, params, trajectories, time_steps = runner_state
            context = env_state.current_context

            # SELECT ACTION (Thompson Sampling)
            posterior_rng = jax.random.split(rng, config["NUM_ENVS"])
            indices = time_steps < i
            params = jax.vmap(
                langevin_posterior_sampling, in_axes=(0, 0, 0, None, None)
            )(posterior_rng, params, trajectories, indices, log_prior_fn)
            action = env_params.best_action_value_fn(params, env_state.current_context)[
                0
            ]

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)

            actions = trajectories.action.at[:, i, :].set(action)
            contexts = trajectories.context.at[:, i, :].set(context)
            rewards = trajectories.reward.at[:, i, :].set(reward)
            trajectories = Trajectory(action=actions, context=contexts, reward=rewards)
            runner_state = (env_state, rng, params, trajectories, time_steps)
            return runner_state, None

        max_steps = env_params.max_episodes
        time_steps = jnp.arange(max_steps)
        trajectories = Trajectory(
            action=jnp.empty(
                (
                    env_state.last_action.shape[0],
                    max_steps,
                    env_state.last_action.shape[-1],
                )
            ),
            context=jnp.empty(
                (
                    env_state.current_context.shape[0],
                    max_steps,
                    env_state.current_context.shape[-1],
                )
            ),
            reward=jnp.empty(
                (
                    env_state.last_reward.shape[0],
                    max_steps,
                    env_state.last_reward.shape[-1],
                )
            ),
        )
        runner_state = (env_state, rng, init_params, trajectories, time_steps)
        runner_state, _ = jax.lax.scan(_env_step, runner_state, jnp.arange(max_steps))

        return {"runner_state": runner_state, "metrics": runner_state[3]}

    return train
