import jax
import jax.numpy as jnp

from src.configs import ExperiorConfig
from src.envs import Bandit, toy_deterministic_bandit, linear_gaussian_bandit
from src.utils import PRNGSequence


def test_bernoulli_bandit(conf: ExperiorConfig):
    env = Bandit()
    rng = PRNGSequence(0)
    obs, state = env.reset(next(rng), toy_deterministic_bandit)
    obs, state, reward, done, info = env.step(
        next(rng), state, 1, toy_deterministic_bandit
    )

    assert obs.shape == (4,)
    assert state.last_action == 1
    assert state.last_reward == 1
    assert (state.reward_params == jnp.array([0.0, 1.0, 0.0])).all()
    assert state.time == 1
    assert reward == 1


def test_linear_gaussian_bandit(conf: ExperiorConfig):
    env = Bandit()
    rng = PRNGSequence(0)
    obs, state = env.reset(next(rng), linear_gaussian_bandit)
    obs, state, reward, done, info = env.step(
        next(rng), state, jax.random.normal(next(rng), (10,)), linear_gaussian_bandit
    )

    assert obs.shape == (22,)
    assert state.current_context.shape == (10,)
    assert state.last_action.shape == (10,)
