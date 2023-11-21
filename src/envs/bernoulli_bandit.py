"""
Code adapted from
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py
"""

from typing import Optional, Tuple, List

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState:
    last_action: int
    last_reward: int
    exp_reward_best: float
    reward_probs: chex.Array
    time: float


@struct.dataclass
class EnvParams:
    reward_probs: chex.Array
    max_steps_in_episode: int


class BernoulliBandit(environment.Environment):
    """
    JAX version of a Bernoulli bandit environment as in Wang et al. 2017
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(reward_probs=jnp.array([0.1, 0.9]), max_steps_in_episode=100)

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Sample bernoulli reward, increase counter, construct input."""
        reward = jax.random.bernoulli(key, state.reward_probs[action]).astype(jnp.int32)
        state = EnvState(
            action,
            reward,
            state.exp_reward_best,
            state.reward_probs,
            state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""

        state = EnvState(
            0,
            0,
            jnp.max(params.reward_probs),
            params.reward_probs,
            0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        action_one_hot = jax.nn.one_hot(
            state.last_action, self.num_actions(params)
        ).squeeze()
        return jnp.hstack([state.last_reward, action_one_hot, state.time])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "BernoulliBandit"

    def num_actions(self, params: EnvParams) -> int:
        """Number of actions possible in environment."""
        return params.reward_probs.shape[0]

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions(params))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # TODO: check if this is correct
        low = jnp.array(
            [0] + [0 for i in range(self.num_actions(params))] + [0.0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [2]
            + [2 for i in range(self.num_actions(params))]
            + [params.max_steps_in_episode],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (4,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        # TODO: check if this is correct
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions(params)),
                "last_reward": spaces.Discrete(2),
                "exp_reward_best": spaces.Box(0, 1, (1,), jnp.float32),
                "reward_probs": spaces.Box(
                    0, 1, (self.num_actions(params),), jnp.float32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
