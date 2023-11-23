"""
Code adapted from
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py
"""
import chex
from gymnax.environments.environment import EnvParams
import jax
import jax.numpy as jnp

from flax import struct
from jax import lax
from jax.tree_util import Partial
from typing import Tuple, Callable, Union, Optional

from gymnax.environments import environment, spaces


ContextType = chex.Array
RewardFuncParamsType = chex.Array
RewardType = chex.Array
ActionType = Union[int, chex.Array]


@struct.dataclass
class EnvState:
    current_context: ContextType
    last_action: ActionType
    last_reward: float
    reward_params: RewardFuncParamsType
    time: float


@struct.dataclass
class EnvParams:
    reward_prior_fn: Callable[[chex.PRNGKey], RewardFuncParamsType]
    reward_dist_fn: Callable[
        [chex.PRNGKey, RewardFuncParamsType, ContextType, ActionType], RewardType
    ]  # sample reward given the context, action, and parameters
    best_action_reward_fn: Callable[[RewardFuncParamsType, ContextType], ActionType]
    context_dist_fn: Callable[[chex.PRNGKey], ContextType]
    max_steps_in_episode: int
    action_space: Optional[spaces.Space] = struct.field(pytree_node=False)
    observation_space: Optional[spaces.Space] = struct.field(pytree_node=False)
    state_space: Optional[spaces.Space] = struct.field(pytree_node=False)


class StochasticBayesBandit(environment.Environment):
    def __init__(self):
        super().__init__()
        self._n_action = None

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        prior_fn = Partial(lambda key: jnp.array([0.1, 0.9]))
        reward_dist_fn = Partial(
            lambda key, params, context, action: jax.random.bernoulli(
                key, params
            ).astype(jnp.float32)[action]
        )
        best_action_reward_fn = Partial(
            lambda params, context: (jnp.argmax(params), jnp.max(params))
        )
        context_dist_fn = Partial(lambda key: jnp.array([0.0]))  # fixed context

        return EnvParams(
            reward_prior_fn=prior_fn,
            reward_dist_fn=reward_dist_fn,
            context_dist_fn=context_dist_fn,
            best_action_reward_fn=best_action_reward_fn,
            max_steps_in_episode=100,
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: ActionType, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Sample context, reward, increase counter, construct input."""
        k1, k2 = jax.random.split(key)
        current_context = state.current_context
        reward_dist = params.reward_dist_fn
        reward = reward_dist(k1, state.reward_params, current_context, action)
        next_context = params.context_dist_fn(k2)
        state = EnvState(
            next_context,
            action,
            reward,
            state.reward_params,
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
        k1, k2, k3 = jax.random.split(key, 3)
        reward_params = params.reward_prior_fn(k1)
        context = params.context_dist_fn(k2)
        state = EnvState(
            context,
            self.action_space(params).sample(k3),
            0.0,
            reward_params,
            0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Concatenate context, reward, action and time stamp."""
        return jnp.hstack(
            [state.current_context, state.last_reward, state.last_action, state.time]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
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
