import jax.numpy as jnp
import numpy as np

from src.baselines import BernoulliTS
from src.utils import PRNGSequence

def test_bernoulli_ts():
    num_actions = 3
    policy = BernoulliTS(num_actions=num_actions)

    # Create some mock data for testing
    rng_key = PRNGSequence(42)
    batch_size = 2
    T = 5
    actions = jnp.array([[0, 1, 2, 1, 0], [2, 2, 0, 1, 1]])
    rewards = jnp.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]])

    # Test the __call__ method
    log_probs = policy(next(rng_key), None, actions, rewards)

    # Assert some basic expectations about the output
    assert log_probs.shape == (batch_size, num_actions)
    assert jnp.all(log_probs <= 0) # Since they are log-probabilities
    assert jnp.all(log_probs >= -np.inf)
