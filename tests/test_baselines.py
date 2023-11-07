import jax.numpy as jnp
import numpy as np

import os, json

from src.baselines import BernoulliTS
from src.utils import PRNGSequence
from src.eval import bayes_regret

from tests import TEST_CONFIG


def test_bernoulli_ts():
    key = PRNGSequence(42)
    num_actions = 3
    expert_policy = jnp.ones((num_actions,)) / num_actions
    policy = BernoulliTS(expert_policy)
    regret = bayes_regret(
        next(key),
        policy,
        num_actions,
        TEST_CONFIG.trainer.test_horizon,
        TEST_CONFIG.trainer.policy_trainer.mc_samples,
    )

    metrics = {"ts_regret": regret.tolist()}

    with open(os.path.join(TEST_CONFIG.out_dir, "ts_test"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    # Create some mock data for testing
    actions = jnp.array([[0, 1, 2, 1, 0], [2, 2, 0, 1, 1]])
    rewards = jnp.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]])
    time_steps = jnp.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    # Test the __call__ method
    log_probs = policy(next(key), time_steps, actions, rewards)

    # Assert some basic expectations about the output
    assert log_probs.shape == (2, num_actions)
    assert jnp.all(log_probs <= 0)  # Since they are log-probabilities
    assert jnp.all(log_probs >= -np.inf)
