import jax.numpy as jnp

from src.configs import ExperiorConfig
from src.envs import BernoulliBandit, BernoulliEnvParams
from src.utils import PRNGSequence


def test_bernoulli_bandit(conf: ExperiorConfig):
    params = BernoulliEnvParams(
        reward_probs=jnp.array([0.0, 1.0, 0.0]), max_steps_in_episode=100
    )

    env = BernoulliBandit()
    rng = PRNGSequence(0)
    obs, state = env.reset(next(rng), params)
    obs, state, reward, done, info = env.step(next(rng), state, 1, params)

    assert obs.shape == (5,)
    assert state.last_action == 1
    assert state.last_reward == 1
    assert state.exp_reward_best == 1.0
    assert state.time == 1
    assert reward == 1
