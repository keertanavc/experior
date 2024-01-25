import jax.numpy as jnp
import jax
import flax
import optax
import flashbax as fbx
import flax.linen as nn

from typing import Callable, Union

from experior.envs import Environment
from experior.utils import VecTrainState
from .utils import adam_slgd

from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


class Q_TrainState(VecTrainState):
    target_params: flax.core.FrozenDict


# TODO maybe try more optimizers (adamw, with graident clip, etc)
def make_lmc_dqn_train(
    env: Environment,
    q_network: nn.Module,
    learning_rate: Union[float, Callable[[int], float]],
    temperature: Union[float, Callable[[int], float]],
    num_envs: int,
    buffer_size: int,
    batch_size: int,
    steps: int,
    train_frequency: int,
    target_network_frequency: int,
    discount_factor: float = 1.0,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):
        # init q-network
        rng, rng_ = jax.random.split(rng)
        env_params = env.default_params
        obs, _ = env.reset(rng_, env_params)
        rng, k1, k2 = jax.random.split(rng, 3)
        tx = adam_slgd(learning_rate=learning_rate, temperature=temperature, rng_key=k1)
        q_state = Q_TrainState.create(
            apply_fn=q_network.apply,
            params=jax.vmap(q_network.init, in_axes=(0, None))(
                jax.random.split(k2, num_envs), obs[None, ...]
            ),
            target_params=jax.vmap(q_network.init, in_axes=(0, None))(
                jax.random.split(k2, num_envs), obs[None, ...]
            ),
            tx=tx,
        )

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

        def update_q_network(q_state, batch):
            # shapes: (batch_size, num_envs, ...)
            obs, action, next_obs, reward, done = (
                batch["obs"],
                batch["action"],
                batch["next_obs"],
                batch["reward"],
                batch["done"],
            )

            # shape: (batch_size, num_envs, n_actions)
            q_next_target = jax.vmap(q_state.apply_fn, in_axes=(0, 1), out_axes=(1))(
                q_state.target_params, next_obs
            )
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size, num_envs)
            q_value = reward + discount_factor * q_next_target * (1.0 - done)

            i, j = jnp.meshgrid(
                jnp.arange(batch_size), jnp.arange(num_envs), indexing="ij"
            )

            def td_loss(params):
                # (batch_size, num_envs, n_actions)
                q_pred = jax.vmap(q_state.apply_fn, in_axes=(0, 1), out_axes=(1))(
                    params, obs
                )
                q_pred = q_pred[i, j, action.squeeze()]

                # the loss mean over all envs
                return jnp.mean((q_pred - q_value) ** 2), q_pred

            (loss_value, q_pred), grad = jax.value_and_grad(td_loss, has_aux=True)(
                q_state.params
            )
            q_state = q_state.apply_gradients(grads=grad)
            return loss_value, q_pred, q_state

        def _env_step(runner_state, i):
            obs, env_state, q_state, rng, buffer_state = runner_state
            # action from q network
            action = jnp.argmax(
                jax.vmap(q_state.apply_fn)(q_state.params, obs[:, None, ...]),
                axis=-1,
            ).reshape(
                (num_envs,)
            )  # TODO only for int action types
            rng, rng_ = jax.random.split(rng)
            next_obs, env_state, reward, done, info = jax.vmap(
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
                # train q network
                l, qp, qs = update_q_network(q_state, batch.experience)
                return k, l, qp, qs

            def _no_train():
                return rng, 0.0, jnp.zeros((batch_size, num_envs)), q_state

            # training
            rng, loss, q_pred, q_state = jax.lax.cond(
                i % train_frequency == 0,
                _train,
                _no_train,
            )

            q_state = jax.lax.cond(
                i % target_network_frequency == 0,
                lambda: q_state.replace(target_params=q_state.params),
                lambda: q_state,
            )

            runner_state = (next_obs, env_state, q_state, rng, buffer_state)
            return runner_state, {
                "loss": loss,
                "reward": reward,
                "done": done,
                "info": info,
            }

        runner_state = (obs, env_state, q_state, rng, buffer_state)
        runner_state, output = jax.lax.scan(_env_step, runner_state, jnp.arange(steps))

        return runner_state, output

    return train
