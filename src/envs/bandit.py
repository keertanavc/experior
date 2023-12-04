"""
Code adapted from
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py
"""
import chex
import jax
import jax.numpy as jnp

from jax import lax
from typing import Tuple

from gymnax.environments import environment, spaces
from gymnax.environments.environment import EnvParams
from .common import *


class StochasticBayesBandit(environment.Environment):
    def __init__(self):
        super().__init__()
        self._n_action = None

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Action, params: EnvParams
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Sample context, reward, increase counter, construct input."""
        action = action.reshape(state.last_action.shape)
        k1, k2 = jax.random.split(key)
        current_context = state.current_context
        reward = params.reward_dist_fn(
            k1, state.unobserved_context, current_context, action
        )
        next_context = params.init_context_dist_fn(k2, 1)
        state = EnvState(
            next_context.reshape(-1,),
            action,
            reward,
            state.unobserved_context,
            state.time + 1,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        """Reset environment state by sampling initial position."""
        k1, k2, k3 = jax.random.split(key, 3)
        reward_params = params.prior_fn(k1, 1)
        context = params.init_context_dist_fn(k2, 1)
        state = EnvState(
            context.reshape(-1, ),
            self.action_space(params).sample(k3).reshape(-1, ),
            jnp.array([0.0]),
            reward_params,
            0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> Observation:
        """Concatenate context, reward, action and time stamp."""
        return jnp.hstack(
            [
                state.current_context.reshape(1, -1),
                state.last_reward.reshape(1, -1),
                state.last_action.reshape(1, -1),
                jnp.array(state.time).reshape(-1, 1),
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_episodes
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "StochasticBayesBandit"

    def action_space(self, params: EnvParams) -> spaces.Space:
        return params.action_space

    def observation_space(self, params: EnvParams) -> spaces.Space:
        return params.observation_space

    def state_space(self, params: EnvParams) -> spaces.Space:
        return params.state_space
