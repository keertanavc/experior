
import jax
import jax.numpy as jnp

from src.configs import TransformerPolicyConfig
from src.models import Policy, BetaPrior, policy_rollout
from src.utils import PRNGSequence

# TODO best way to write test for jax?

def test_policy():
    # Configuration
    config = TransformerPolicyConfig(horizon=10,
                                     h_dim=64,
                                     dtype=jnp.float32,
                                     num_heads=4,
                                     drop_p=0.1,
                                     n_blocks=3,
                                     num_actions=5)

    # Creating test data
    batch_size = 2
    T = 5
    rng = PRNGSequence(0)
    timesteps = jnp.arange(
        batch_size * T).reshape(batch_size, T) % config.horizon
    actions = jax.random.randint(
        next(rng), (batch_size, T), 0, config.num_actions)
    rewards = jax.random.normal(next(rng), (batch_size, T))

    # Initialize the policy
    policy = Policy(config=config)
    params = policy.init(next(rng), timesteps, actions, rewards)

    # Test call
    action_probs = jnp.exp(policy.apply(params, timesteps, actions, rewards))
    assert action_probs.shape == (batch_size, config.num_actions)
    assert jnp.all(action_probs >= 0) and jnp.all(action_probs <= 1)
    assert jnp.allclose(action_probs.sum(axis=1), 1.0)


def test_policy_rollout():
    config = TransformerPolicyConfig(horizon=50,
                                     h_dim=32,
                                     dtype=jnp.float32,
                                     num_heads=2,
                                     drop_p=0.1,
                                     n_blocks=1,
                                     num_actions=5)
    rng = PRNGSequence(0)
    prior = BetaPrior(num_actions=config.num_actions)
    params = prior.init(next(rng), jnp.ones((1, config.num_actions)))
    n_sample = 100

    mu_vectors = prior.apply(params, rng_key=next(
        rng), size=n_sample, method=prior.sample)

    mu_vectors = jax.lax.stop_gradient(mu_vectors)
    policy = Policy(config=config)
    params = policy.init(next(rng), jnp.ones((
        1, 2), dtype=jnp.int32), jnp.ones((1, 2), dtype=jnp.int32), jnp.ones((1, 2)))

    policy_fn = lambda timesteps, actions, rewards: policy.apply(params, timesteps, actions, rewards)

    actions, rewards, log_policy_probs = policy_rollout(policy_fn, next(rng),
                                                        mu_vectors, config.horizon)

    assert actions.shape == (n_sample, config.horizon) and rewards.shape == (
        n_sample, config.horizon) and log_policy_probs.shape == (n_sample, config.horizon)

    assert jnp.all(jnp.in1d(actions, jnp.arange(config.num_actions)))
    probs = jnp.exp(log_policy_probs)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1)


def test_rollout():
    num_actions = 2
    num_samples = 2
    horizon = 3
    policy_fn = lambda timesteps, actions, rewards: jnp.array([jnp.log(0.01), jnp.log(0.99)])
    rng = PRNGSequence(1234)
    mu_vectors = jnp.array([[0.1, 0.9], [0.9, 0.1]])
    actions, rewards, log_policy_probs = policy_rollout(policy_fn, next(rng),
                                                        mu_vectors, horizon)

    print(actions)
    print(rewards)
    print(log_policy_probs)

    assert actions.shape == (num_samples, horizon) and \
    rewards.shape == (num_samples, horizon) and \
    log_policy_probs.shape == (num_samples, horizon)

    assert jnp.all(jnp.in1d(actions, jnp.arange(num_actions)))
    probs = jnp.exp(log_policy_probs)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1)
