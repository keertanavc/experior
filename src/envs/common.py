import chex

from flax import struct
from typing import NamedTuple, Union, Tuple, Callable, Optional, Dict
from gymnax.environments import spaces


class Trajectory(NamedTuple):
    action: chex.Array
    context: chex.Array
    reward: chex.Array


Context = chex.Array
UnobservedContext = chex.Array
Reward = chex.Array
Observation = Union[chex.Array, Dict]
Action = Union[int, chex.Array]


@struct.dataclass
class EnvState:
    current_context: Context
    last_action: Action
    last_reward: Reward
    unobserved_context: UnobservedContext
    time: float


@struct.dataclass
class EnvParams:
    prior_fn: Callable[[chex.PRNGKey, chex.Shape], UnobservedContext]
    ref_prior_fn: Callable[[chex.PRNGKey, chex.Shape], UnobservedContext]
    reward_dist_fn: Callable[[chex.PRNGKey, UnobservedContext, Context, Action], Reward]
    Q_function: Callable[[UnobservedContext, Context, Action], Reward]
    best_action_value_fn: Callable[[UnobservedContext, Context], Tuple[Action, Reward]]
    init_context_dist_fn: Callable[[chex.PRNGKey, chex.Shape], Context]
    max_episodes: int
    action_space: Optional[spaces.Space] = struct.field(pytree_node=False)
    observation_space: Optional[spaces.Space] = struct.field(pytree_node=False)
    state_space: Optional[spaces.Space] = struct.field(pytree_node=False)

    transition_dist_fn: Optional[
        Callable[[chex.PRNGKey, Context, Action], Context]
    ] = None
    horizon: Optional[int] = 1
