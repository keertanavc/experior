import jax.numpy as jnp
import jax
import chex
import optax

from experior.envs import Environment
from experior.experts import Trajectory, expert_log_likelihood_fn
from experior.max_ent import MaxEntTrainState
from .utils import RewardModel, LinearDiscreteRewardModel
from .ts import make_thompson_sampling


def make_max_ent_thompson_sampling(
    env: Environment,
    reward_model: RewardModel,
    num_envs: int,
    total_steps: int,
    max_ent_epsilon: float,
    max_ent_lambda: float,
    max_ent_learning_rate: float,
    max_ent_prior_n_samples: int,
    max_ent_steps: int,
    expert_beta: float,
    langevin_learning_rate: float,
    langevin_batch_size: float,
    langevin_updates_per_step: int,
    init_emp_ent: float = 0.0,
    langevin_grad_clip: float = 50.0,
):
    def train(rng, expert_trajectories: Trajectory, meta_params: chex.Array = None):
        rng, rng_ = jax.random.split(rng)
        obs, _ = env.reset(rng_, env.default_params)
        rng, rng_ = jax.random.split(rng)
        sampled_params = jax.vmap(reward_model.init, (0, None))(
            jax.random.split(rng_, max_ent_prior_n_samples), obs
        )
        if not isinstance(reward_model, LinearDiscreteRewardModel):
            raise NotImplementedError  # TODO implement continuous actions

        n_trajectory = expert_trajectories.obs.shape[0]
        n_horizon = expert_trajectories.obs.shape[1]

        def get_expert_log_likelihood(param):
            traj_i, horizon_j = jnp.meshgrid(
                jnp.arange(n_trajectory), jnp.arange(n_horizon), indexing="ij"
            )
            actions = expert_trajectories.action.reshape(n_trajectory, n_horizon)

            # shape: (n_trajectory, n_horizon, n_action)
            mean_values = jax.vmap(reward_model.apply, in_axes=(None, 0))(
                param, expert_trajectories.obs
            )

            taken_values = mean_values[traj_i, horizon_j, actions]
            return expert_log_likelihood_fn(expert_beta, mean_values, taken_values)

        # shape: (max_ent_prior_n_samples, n_trajectory)
        sampled_log_likelihoods = jax.vmap(get_expert_log_likelihood)(sampled_params)

        rng, rng_ = jax.random.split(rng)
        max_ent_state = MaxEntTrainState.create(
            rng=rng_,
            n_trajectory=n_trajectory,
            lambda_=max_ent_lambda,
            epsilon=max_ent_epsilon,
            tx=optax.adam(max_ent_learning_rate),
            num_envs=1,
        )

        update_step = max_ent_state.make_max_ent_update_step(
            sampled_log_likelihoods[None, ...], init_emp_ent
        )

        max_ent_state, max_ent_loss = jax.lax.scan(
            update_step, max_ent_state, None, max_ent_steps
        )

        prior_log_pdf = jax.tree_util.Partial(
            lambda p, _: max_ent_state.log_prior_fn(
                max_ent_state.params, get_expert_log_likelihood(p)
            )
        )

        ts_train = make_thompson_sampling(
            env,
            reward_model,
            num_envs,
            total_steps,
            langevin_learning_rate,
            langevin_batch_size,
            langevin_updates_per_step,
            prior_log_pdf,
            langevin_grad_clip,
        )
        state, metric = ts_train(rng, meta_params=meta_params)
        metric["max_ent_loss"] = max_ent_loss["loss"]
        return state, max_ent_state, metric

    return train
