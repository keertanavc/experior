import chex
from typing import NamedTuple


class Trajectory(NamedTuple):
    action: chex.Array
    state: chex.Array
    reward: chex.Array
    obs: chex.Array
