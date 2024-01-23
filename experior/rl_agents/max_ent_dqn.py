import jax.numpy as jnp
import jax
import flax
import optax
import flashbax as fbx

from typing import Callable, Union, Dict

from experior.envs import Environment
from experior.utils import VecTrainState
from .utils import QNetwork, adam_slgd
from experior.experts import Trajectory
from experior.experts import expert_log_likelihood_fn
from experior.max_ent import MaxEntTrainState


class Q_TrainState(VecTrainState):
    target_params: flax.core.FrozenDict


# TODO maybe try more optimizers (adamw, with graident clip, etc)
def make_max_ent_dqn_train(
    env: Environment,
    q_network: QNetwork,
    learning_rate: Union[float, Callable[[int], float]],
    temperature: Union[float, Callable[[int], float]],
    num_envs: int,
    buffer_size: int,
    batch_size: int,
    steps: int,
    train_frequency: int,
    target_network_frequency: int,
    max_ent_lambda: float,
    max_ent_epsilon: float,
    max_ent_learning_rate: Union[float, Callable[[int], float]],
    max_ent_prior_n_samples: int,
    max_ent_updates_per_step: int,
    max_ent_updates_frequency: int,
    expert_beta: float,
    slgd_updates_per_step: int = 1,
    discount_factor: float = 1.0,
):
    # TODO have fixed rng key for env for all the agents
    def train(rng, expert_trajectories: Trajectory):
        # init q-network
        rng, rng_ = jax.random.split(rng)
        env_params = env.default_params
        obs, _ = env.reset(rng_, env_params)
        rng, rng_ = jax.random.split(rng)
        tx = adam_slgd(
            learning_rate=learning_rate, temperature=temperature, rng_key=rng_
        )
        rng, rng_ = jax.random.split(rng)
        q_state = Q_TrainState.create(
            apply_fn=q_network.apply,
            params=jax.vmap(q_network.init, in_axes=(0, None))(
                jax.random.split(rng_, num_envs), obs[None, ...]
            ),
            target_params=jax.vmap(q_network.init, in_axes=(0, None))(
                jax.random.split(rng_, num_envs), obs[None, ...]
            ),
            tx=tx,
        )
        rng, rng_ = jax.random.split(rng)
        unobs_tx = adam_slgd(
            learning_rate=learning_rate, temperature=temperature, rng_key=rng_
        )  # TODO grad clip
        rng, rng_ = jax.random.split(rng)
        unobs_state = VecTrainState.create(
            apply_fn=None,
            params=jax.random.normal(rng_, (num_envs, q_network.n_features)),
            tx=unobs_tx,
        )

        # max entropy prior
        rng, rng_ = jax.random.split(rng)
        n_trajectory = expert_trajectories.obs.shape[0]
        horizon = expert_trajectories.obs.shape[1]  # TODO assumes horizon
        max_ent_state = MaxEntTrainState.create(
            rng=rng_,
            n_trajectory=n_trajectory,
            lambda_=max_ent_lambda,
            epsilon=max_ent_epsilon,
            tx=optax.adam(max_ent_learning_rate),
            num_envs=num_envs,
        )

        # set up the environment
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None))(
            jax.random.split(rng_, num_envs), env_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            jax.random.split(rng_, num_envs), env_params
        )

        # replay buffer
        rng, rng_ = jax.random.split(rng)
        buffer = fbx.make_item_buffer(
            max_length=buffer_size,
            min_length=batch_size,
            sample_batch_size=batch_size,
        )

        # TODO only for int action types
        buffer_state = buffer.init(
            {
                "obs": obs,
                "action": jnp.zeros((num_envs,), dtype=jnp.int32),
                "reward": jnp.zeros((num_envs,)),
                "done": jnp.zeros((num_envs,), dtype=jnp.bool_),
                "next_obs": obs,
            }
        )

        # useful indices
        batch_i, envs_j = jnp.meshgrid(
            jnp.arange(batch_size), jnp.arange(num_envs), indexing="ij"
        )
        traj_i, horizon_j = jnp.meshgrid(
            jnp.arange(n_trajectory), jnp.arange(horizon), indexing="ij"
        )

        # likelihoods and unobserved_contexts
        rng, rng_ = jax.random.split(rng)
        sampled_unobserved_contexts = jax.random.normal(
            rng_, (max_ent_prior_n_samples, q_network.n_features)
        )

        def get_expert_log_likelihood(q_params, unobs_context):
            # q_params and unobs_context shape: (1, ...)
            # expert_trajectories.obs shape: (n_trajectory, n_horizon, obs_dim)
            # expert_q_values shape: (n_trajectory, n_horizon, n_action)
            expert_q_values = jax.vmap(q_state.apply_fn, in_axes=(None, 0, None))(
                q_params, expert_trajectories.obs, unobs_context
            )
            expert_q_values_taken = expert_q_values[
                traj_i, horizon_j, expert_trajectories.action.squeeze()
            ]
            # shape: (n_trajectory,)
            return expert_log_likelihood_fn(
                expert_beta, expert_q_values, expert_q_values_taken
            )

        def update_q_network(
            q_state: Q_TrainState,
            unobs_state: VecTrainState,
            max_ent_state: MaxEntTrainState,
            batch: Dict,
        ):
            # shapes: (batch_size, num_envs, ...)
            obs, action, next_obs, reward, done = (
                batch["obs"],
                batch["action"],
                batch["next_obs"],
                batch["reward"],
                batch["done"],
            )

            # shape: (batch_size, num_envs, n_actions)
            q_next_target = jax.vmap(q_state.apply_fn, in_axes=(0, 1, 0), out_axes=(1))(
                q_state.target_params,
                next_obs,
                unobs_state.params,
            )
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size, num_envs)
            q_value = reward + discount_factor * q_next_target * (1.0 - done)

            def td_max_ent_loss(q_params, unobs_params):
                # (batch_size, num_envs, n_actions)
                q_pred = jax.vmap(q_state.apply_fn, in_axes=(0, 1, 0), out_axes=(1))(
                    q_params, obs, unobs_params
                )
                q_pred = q_pred[batch_i, envs_j, action.squeeze()]

                # shape: (num_envs, n_trajectory)
                traj_log_likelihoods = jax.vmap(
                    get_expert_log_likelihood, in_axes=(0, 0)
                )(
                    q_state.params, unobs_params
                )  # TODO note that we don't want to take grad w.r.t. q_params for traj_log_likelihoods

                # shape: (num_envs,)
                max_ent_log_pdf = jax.vmap(max_ent_state.log_prior_fn)(
                    max_ent_state.params, traj_log_likelihoods
                )

                # the loss mean over all envs
                loss = (
                    0.5 * jnp.sum((q_pred - q_value) ** 2, axis=0) - max_ent_log_pdf
                ).mean()
                return loss

            loss_value, (grad, unobs_grad) = jax.value_and_grad(
                td_max_ent_loss, argnums=(0, 1)
            )(q_state.params, unobs_state.params)
            q_state = q_state.apply_gradients(grads=grad)
            unobs_state = unobs_state.apply_gradients(grads=unobs_grad)
            # returns average loss over all envs
            return loss_value, q_state, unobs_state

        def _env_step(runner_state, i):
            (
                obs,
                env_state,
                q_state,
                unobs_state,
                max_ent_state,
                rng,
                buffer_state,
            ) = runner_state

            # train max entropy prior
            # shape: (num_envs, max_ent_prior_n_samples, n_trajectory)
            # TODO we may need to reset the optimizer here
            def _max_ent_train():
                sampled_log_likelihoods = jax.vmap(
                    jax.vmap(get_expert_log_likelihood, in_axes=(None, 0)),
                    in_axes=(0, None),
                )(q_state.params, sampled_unobserved_contexts)

                max_ent_update_step = max_ent_state.make_max_ent_update_step(
                    sampled_log_likelihoods, init_emp_ent=0.0  # TODO fix
                )

                max_ent_s = max_ent_state.reset_opt_state()  # TODO fix

                max_ent_s, metrics = jax.lax.scan(
                    max_ent_update_step,
                    init=max_ent_s,
                    xs=None,
                    length=max_ent_updates_per_step,
                )
                return max_ent_s

            def _max_ent_not_train():
                return max_ent_state

            max_ent_state = jax.lax.cond(
                i % max_ent_updates_frequency == 0, _max_ent_train, _max_ent_not_train
            )
            # action from q network
            action = jnp.argmax(
                jax.vmap(q_state.apply_fn)(
                    q_state.params,
                    obs[:, None, ...],
                    unobs_state.params,
                ),
                axis=-1,
            ).reshape(
                (num_envs,)
            )  # TODO only for int action types
            rng, rng_ = jax.random.split(rng)
            next_obs, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, 0)
            )(jax.random.split(rng_, num_envs), env_state, action, env_params)

            # add to replay buffer
            buffer_state = buffer.add(
                buffer_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "next_obs": next_obs,
                },
            )

            def _train():
                # sample from replay buffer
                k, rng_ = jax.random.split(rng)
                batch = buffer.sample(buffer_state, rng_)

                # train q network (update multiple times)
                def _update_q_network(runner_state, _):
                    (qs, us) = runner_state
                    l, qs, us = update_q_network(
                        qs, us, max_ent_state, batch.experience
                    )
                    runner_state = (qs, us)
                    return runner_state, l

                (q_s, u_s), loss = jax.lax.scan(
                    _update_q_network,
                    (q_state, unobs_state),
                    None,
                    slgd_updates_per_step,
                )
                return k, loss.mean(), q_s, u_s

            def _no_train():
                return rng, 0.0, q_state, unobs_state

            # training
            rng, loss, q_state, unobs_state = jax.lax.cond(
                i % train_frequency == 0,
                _train,
                _no_train,
            )

            q_state = jax.lax.cond(
                i % target_network_frequency == 0,
                lambda: q_state.replace(target_params=q_state.params),
                lambda: q_state,
            )

            runner_state = (
                next_obs,
                env_state,
                q_state,
                unobs_state,
                max_ent_state,
                rng,
                buffer_state,
            )
            return runner_state, {"loss": loss, "reward": reward, "done": done}

        runner_state = (
            obs,
            env_state,
            q_state,
            unobs_state,
            max_ent_state,
            rng,
            buffer_state,
        )
        runner_state, output = jax.lax.scan(_env_step, runner_state, jnp.arange(steps))

        return runner_state, output

    return train
