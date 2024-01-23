import jax

import chex
from typing import NamedTuple


from experior.envs import Environment


class Trajectory(NamedTuple):
    action: chex.Array
    reward: chex.Array
    obs: chex.Array
    value: chex.Array = None
    log_prob: chex.Array = None
    done: chex.Array = None
    info: chex.Array = None


def generate_optimal_trajectories(
    rng: chex.PRNGKey, env: Environment, num_trajectories: int, horizon: int
):
    rng, rng_ = jax.random.split(rng)
    params = env.default_params
    params = jax.vmap(env.init_env, in_axes=(0, None))(
        jax.random.split(rng_, num_trajectories), params
    )

    def optimal_rollout(key, params):
        obs, state = env.reset_env(key, params)

        def _env_step(runner_state, _):
            state, rng, obs = runner_state
            rng, rng_ = jax.random.split(rng)
            action = env.optimal_policy(rng_, state, params)
            rng, rng_ = jax.random.split(rng)
            next_obs, next_state, reward, done, info = env.step(
                rng_, state, action, params
            )
            runner_state = (next_state, rng, next_obs)
            return runner_state, Trajectory(action, reward, obs)

        runner_state = (state, key, obs)
        runner_state, trajectory = jax.lax.scan(_env_step, runner_state, None, horizon)
        return trajectory

    expert_trajectories = jax.vmap(optimal_rollout, in_axes=(0, 0))(
        jax.random.split(rng, num_trajectories), params
    )

    return expert_trajectories


def expert_log_likelihood_fn(
    beta: float,
    # shape: (n_trajectory, n_horizon, n_action), for continuous actions,
    # n_action is the number of sampled actions
    q_values: chex.Array,
    taken_q_values: chex.Array,  # shape: (n_trajectory, n_horizon)
):
    # n_trajectory x n_horizon
    denominator = jax.scipy.special.logsumexp(beta * q_values, axis=2)

    numerator = beta * taken_q_values  # n_trajectory x n_horizon
    # TODO in the case of large horizon, this might get large
    return (numerator - denominator).sum(axis=1)  # n_trajectory
