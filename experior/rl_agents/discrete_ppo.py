import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Union, Callable
from experior.experts import Trajectory
from experior.envs import Environment
from experior.utils import VecTrainState

from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


def make_discrete_ppo_train(
    env: Environment,
    actor_critic_network: nn.Module,
    learning_rate: Union[float, Callable[[int], float]],
    num_envs: int,
    steps: int,  # total steps
    num_actors: int,
    train_frequency: int,
    max_grad_norm: float,
    epochs_per_iteration: int,
    num_minibatches: int,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_eps: float = 0.2,
    unroll_steps: int = 1,  # TODO what is this?
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):

        def linear_schedule(count):
          frac = (
              1.0
              - (count // (num_minibatches * epochs_per_iteration))
              / update_nums
          )
          return learning_rate * frac
        # init the actor-critic network
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        init_x = jnp.zeros((1,) + env.observation_space(env.default_params).shape)
        rng, rng_ = jax.random.split(rng)
        train_state = VecTrainState.create(
            apply_fn=actor_critic_network.apply,
            params=jax.vmap(actor_critic_network.init, in_axes=(0, None))(
                jax.random.split(rng_, num_envs), init_x
            ),
            tx=tx,
        )

        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None))(
            jax.random.split(rng_, num_envs * num_actors), env.default_params
        )

        # shape: (num_envs, num_actors, ...)
        env_params = jax.tree_util.tree_map(
            lambda x: x.reshape((num_envs, num_actors, *x.shape[1:])), env_params
        )
        rng, rng_ = jax.random.split(rng)
        obsv, env_state = jax.vmap(jax.vmap(env.reset))(
            jax.random.split(rng_, num_envs * num_actors).reshape(
                num_envs, num_actors, -1
            ),
            env_params,
        )  # shape: (num_envs, num_actors, ...)

        def _train_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                # train_state shape: (num_envs, ...)
                # env_state, last_obs shape: (num_envs, num_actors, ...)
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, rng_ = jax.random.split(rng)
                pi, value = jax.vmap(lambda p, obs: train_state.apply_fn(p, obs), in_axes=(0, 0), out_axes=(0))(
                    train_state.params, last_obs
                )  # shape: (num_envs, num_actors, action_dim), (num_envs, num_actors, 1)
                action = pi.sample(seed=rng_)  # shape: (num_envs, num_actors)
                log_prob = pi.log_prob(action)  # shape: (num_envs, num_actors)

                # STEP ENV
                rng, rng_ = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(jax.vmap(env.step))(
                    jax.random.split(rng_, num_envs * num_actors).reshape(
                        num_envs, num_actors, -1
                    ),
                    env_state,
                    action,
                    env_params,
                )  # shape: (num_envs, num_actors, ...)
                transition = Trajectory(
                    action, reward, last_obs, value, log_prob, done, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_frequency
            )  # traj_batch shape: (train_frequency, num_envs, num_actors, ...)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = jax.vmap(lambda p, obs: train_state.apply_fn(p, obs))(
                train_state.params, last_obs
            )

            def _calculate_gae(traj_batch: Trajectory, last_val):
                def _get_advantages(gae_and_next_value, trajectory: Trajectory):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        trajectory.done,
                        trajectory.value,
                        trajectory.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=unroll_steps,
                )  # shape: (train_frequency, num_envs, num_actors, 1)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state: VecTrainState, batch_info):
                    traj_batch, advantages, targets = batch_info
                    # shape: (num_envs, minibatch_size, ...)

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = jax.vmap(
                            lambda p, obs: train_state.apply_fn(p, obs)
                        )(
                            params, traj_batch.obs
                        )  # shape: (num_envs, minibatch_size, action_dim), (num_envs, minibatch_size, 1)
                        log_prob = pi.log_prob(
                            traj_batch.action
                        )  # shape: (num_envs, minibatch_size)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-clip_eps, clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - clip_eps,
                                1.0 + clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor + vf_coef * value_loss - ent_coef * entropy
                        )
                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0))
                    (_, losses), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, rng_ = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = train_frequency * num_actors
                permutation = jax.random.permutation(rng_, batch_size)
                batch = (
                    traj_batch,
                    advantages,
                    targets,
                )  # shape: (train_frequency, num_envs, num_actors, ...)
                batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_envs, batch_size, *x.shape[3:])),
                    batch,
                )  # shape: (num_envs, train_frequency * num_actors, ...)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_envs, num_minibatches, -1] + list(x.shape[2:])
                    ),
                    shuffled_batch,
                )  # shape: (num_envs, num_minibatches, minibatch_size, ...)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 0, 1), minibatches
                )  # shape: (num_minibatches, num_envs, minibatch_size, ...)

                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, losses = jax.lax.scan(
                _update_epoch, update_state, None, epochs_per_iteration
            )
            train_state = update_state[0]
            metric = {"metrics": traj_batch.info, "losses": losses}
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, rng_ = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, rng_)
        update_nums = steps // train_frequency // num_actors + 1
        runner_state, output = jax.lax.scan(
            _train_step, runner_state, None, update_nums
        )  # shape: (update_nums, train_frequency, num_envs, num_actors, ...)
        return runner_state, output

    return train
