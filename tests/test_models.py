import jax
import optax
import tqdm

import jax.numpy as jnp

from src.configs import TransformerPolicyConfig, BetaPriorConfig
from src.models import TransformerPolicy, BetaPrior
from src.rollout import policy_rollout
from src.utils import PRNGSequence

from flax.training import train_state

# TODO best way to write test for jax?


def test_policy():
    # Configuration
    config = TransformerPolicyConfig(
        horizon=10,
        h_dim=64,
        dtype=jnp.float32,
        num_heads=4,
        drop_p=0.1,
        n_blocks=3,
        num_actions=5,
    )

    # Creating test data
    batch_size = 2
    T = 5
    rng = PRNGSequence(0)
    timesteps = jnp.arange(batch_size * T).reshape(batch_size, T) % config.horizon
    actions = jax.random.randint(next(rng), (batch_size, T), 0, config.num_actions)
    rewards = jax.random.normal(next(rng), (batch_size, T))

    # Initialize the policy
    policy = TransformerPolicy(config=config)
    params = policy.init(next(rng), next(rng), timesteps, actions, rewards)

    # Test call
    action_probs = jnp.exp(policy.apply(params, next(rng), timesteps, actions, rewards))

    assert action_probs.shape == (batch_size, config.num_actions)
    assert jnp.all(action_probs >= 0) and jnp.all(action_probs <= 1)
    assert jnp.allclose(action_probs.sum(axis=1), 1.0)


def test_policy_rollout():
    config = TransformerPolicyConfig(
        horizon=50,
        h_dim=32,
        dtype=jnp.float32,
        num_heads=2,
        drop_p=0.1,
        n_blocks=1,
        num_actions=5,
    )
    rng = PRNGSequence(0)
    conf = BetaPriorConfig(num_actions=config.num_actions)
    prior = BetaPrior(conf)
    params = prior.init(next(rng), jnp.ones((1, config.num_actions)))
    n_sample = 100

    mu_vectors = prior.apply(
        params, rng_key=next(rng), size=n_sample, method=prior.sample
    )

    mu_vectors = jax.lax.stop_gradient(mu_vectors)
    policy = TransformerPolicy(config=config)
    params = policy.init(
        next(rng),
        next(rng),
        jnp.ones((1, 2), dtype=jnp.int32),
        jnp.ones((1, 2), dtype=jnp.int32),
        jnp.ones((1, 2)),
    )

    def policy_fn(key, timesteps, actions, rewards):
        return policy.apply(params, key, timesteps, actions, rewards)

    actions, rewards, log_policy_probs = policy_rollout(
        policy_fn, next(rng), mu_vectors, config.horizon
    )

    assert (
        actions.shape == (n_sample, config.horizon)
        and rewards.shape == (n_sample, config.horizon)
        and log_policy_probs.shape == (n_sample, config.horizon, conf.num_actions)
    )

    assert jnp.all(jnp.in1d(actions, jnp.arange(config.num_actions)))
    probs = jnp.exp(log_policy_probs)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1)
    assert jnp.allclose(probs.sum(axis=2), 1.0)


def test_rollout():
    num_actions = 3
    num_samples = 2
    horizon = 4

    def policy_fn(k, t, a, r):
        return (
            jnp.array([jnp.log(0.01), jnp.log(0.01), jnp.log(0.98)])
            .reshape(1, -1)
            .repeat(t.shape[0], axis=0)
        )

    rng = PRNGSequence(1234)
    mu_vectors = jnp.array([[0.1, 0.8, 0.1], [0.9, 0.1, 0.1]])
    actions, rewards, log_policy_probs = policy_rollout(
        policy_fn, next(rng), mu_vectors, horizon
    )

    assert (
        actions.shape == (num_samples, horizon)
        and rewards.shape == (num_samples, horizon)
        and log_policy_probs.shape == (num_samples, horizon, num_actions)
    )

    assert jnp.all(jnp.in1d(actions, jnp.arange(num_actions)))
    probs = jnp.exp(log_policy_probs)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1)
    assert jnp.allclose(probs.sum(axis=2), 1.0)


def test_prior():
    rng = PRNGSequence(123)
    K = 3
    n_samples = 1000
    epochs = 1000
    data = jax.random.beta(
        next(rng), a=2 * jnp.ones(K), b=jnp.ones(K), shape=(n_samples, K)
    )

    conf = BetaPriorConfig(num_actions=K, init_alpha=2, init_beta=1)
    model = BetaPrior(conf)
    variables = model.init(next(rng), jnp.ones((1, K)))
    tx = optax.adam(learning_rate=1e-2)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx
    )

    @jax.jit
    def update_step(state, batch):
        def loss_fn(params):
            log_prob = state.apply_fn({"params": params}, batch)
            return -jnp.mean(log_prob)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    with tqdm.tqdm(total=epochs) as pbar:
        for _ in range(epochs):
            state, loss = update_step(state, data)
            pbar.set_postfix({"loss": loss})
            pbar.update(1)

    assert jnp.isclose(
        state.params["alphas_sq"] ** 2 + BetaPrior.epsilon, 2, atol=1e-1
    ).all()
    assert jnp.isclose(
        state.params["betas_sq"] ** 2 + BetaPrior.epsilon, 1, atol=1e-1
    ).all()


def test_prior_sample():
    K = 4
    size = 100
    conf = BetaPriorConfig(num_actions=K)
    model = BetaPrior(conf)
    rng = PRNGSequence(123)
    variables = model.init(next(rng), jnp.ones((1, K)))
    samples = model.apply(
        {"params": variables["params"]},
        rng_key=next(rng),
        size=size,
        method=model.sample,
    )

    assert samples.shape == (size, K)
    assert samples.min() >= 0 and samples.max() <= 1
