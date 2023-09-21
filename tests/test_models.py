import jax
import optax
import tqdm

import jax.numpy as jnp

from src.models import get_policy, get_prior
from src.rollout import policy_rollout
from src.utils import PRNGSequence
from src.configs import ExperiorConfig, BetaTSPolicyConfig

from tests import TEST_CONFIG

from copy import deepcopy

# TODO best way to write test for jax?
# TODO add test config similar to train


def test_transformer_policy():
    return _test_policy(TEST_CONFIG)


def test_beta_ts_policy():
    conf = deepcopy(TEST_CONFIG)
    conf.policy = BetaTSPolicyConfig(
        prior=conf.prior, num_actions=conf.prior.num_actions
    )
    return _test_policy(conf)


def _test_policy(conf: ExperiorConfig):
    # Creating test data
    batch_size = conf.trainer.policy_trainer.batch_size
    T = conf.trainer.test_horizon
    num_actions = conf.prior.num_actions

    rng = PRNGSequence(0)
    timesteps = jnp.arange(batch_size * T).reshape(batch_size, T) % T
    actions = jax.random.randint(next(rng), (batch_size, T), 0, num_actions)
    rewards = jax.random.bernoulli(next(rng), p=0.5, shape=(batch_size, T))

    # Initialize and call the policy
    policy_state = get_policy(conf.policy.name).create_state(
        next(rng), optimizer=optax.adam(learning_rate=1e-2), conf=conf.policy
    )

    action_log_probs = policy_state.apply_fn(
        {"params": policy_state.params}, next(rng), timesteps, actions, rewards
    )
    action_probs = jnp.exp(action_log_probs)

    assert action_probs.shape == (batch_size, num_actions)
    assert jnp.all(action_probs >= 0) and jnp.all(action_probs <= 1)
    assert jnp.allclose(action_probs.sum(axis=1), 1.0)


def test_softelim_policy():
    conf = deepcopy(TEST_CONFIG)
    conf.trainer.policy_trainer.batch_size = 2
    conf.trainer.test_horizon = 5
    conf.prior.num_actions = 2
    conf.policy.num_actions = 2
    conf.trainer.train_horizon = 5
    conf.policy.horizon = 5
    conf.policy.name = "softelim"
    conf.trainer.policy_trainer.mc_samples = 2

    rng = PRNGSequence(0)
    timesteps = jnp.array([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5]])
    actions = jnp.array([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]])
    rewards = jnp.array([[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0]])

    # Initialize and call the policy
    policy_state = get_policy(conf.policy.name).create_state(
        next(rng), optimizer=optax.adam(learning_rate=1e-2), conf=conf.policy
    )

    action_log_probs = policy_state.apply_fn(
        {"params": policy_state.params}, next(rng), timesteps, actions, rewards
    )
    action_probs = jnp.exp(action_log_probs)
    S = jnp.exp(jnp.array([[0.0, 0.0], [0.0, -2 * (0.5 - 1 / 3) ** 2 * 3]]))
    probs = S / S.sum(axis=1, keepdims=True)
    assert jnp.isclose(action_probs, probs).all()


def test_transformer_policy_rollout():
    return _test_policy_rollout(TEST_CONFIG)


def test_softelim_policy_rollout():
    conf = deepcopy(TEST_CONFIG)
    conf.policy.name = "softelim"

    return _test_policy_rollout(conf)


def _test_policy_rollout(conf: ExperiorConfig):
    rng = PRNGSequence(0)
    num_actions = conf.prior.num_actions
    n_sample = conf.trainer.policy_trainer.mc_samples

    prior_cls = get_prior(conf.prior.name)

    prior_state = prior_cls.create_state(
        next(rng), optimizer=optax.adam(learning_rate=1e-2), conf=conf.prior
    )

    mu_vectors = prior_state.apply_fn(
        {"params": prior_state.params},
        rng_key=next(rng),
        size=n_sample,
        method=prior_cls.sample,
    )

    mu_vectors = jax.lax.stop_gradient(mu_vectors)
    policy_state = get_policy(conf.policy.name).create_state(
        next(rng), optimizer=optax.adam(learning_rate=1e-2), conf=conf.policy
    )

    def policy_fn(key, timesteps, actions, rewards):
        return policy_state.apply_fn(
            {"params": policy_state.params}, key, timesteps, actions, rewards
        )

    horizon = conf.trainer.train_horizon
    actions, rewards, log_policy_probs = policy_rollout(
        policy_fn, next(rng), mu_vectors, horizon
    )

    assert (
        actions.shape == (n_sample, horizon)
        and rewards.shape == (n_sample, horizon)
        and log_policy_probs.shape == (n_sample, horizon, num_actions)
    )

    assert jnp.all(jnp.in1d(actions, jnp.arange(num_actions)))
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
    num_actions = TEST_CONFIG.prior.num_actions
    n_samples = 1000
    epochs = 1000
    data = jax.random.beta(
        next(rng),
        a=2 * jnp.ones(num_actions),
        b=jnp.ones(num_actions),
        shape=(n_samples, num_actions),
    )

    tx = optax.adam(learning_rate=1e-2)
    state = get_prior(TEST_CONFIG.prior.name).create_state(
        next(rng), tx, conf=TEST_CONFIG.prior
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
        state.params["alphas_sq"] ** 2 + TEST_CONFIG.prior.epsilon, 2, atol=1e-1
    ).all()
    assert jnp.isclose(
        state.params["betas_sq"] ** 2 + TEST_CONFIG.prior.epsilon, 1, atol=1e-1
    ).all()


def test_prior_sample():
    size = 100
    num_actions = TEST_CONFIG.prior.num_actions
    rng = PRNGSequence(123)

    prior_cls = get_prior(TEST_CONFIG.prior.name)
    state = prior_cls.create_state(
        next(rng), optax.adam(learning_rate=1e-2), conf=TEST_CONFIG.prior
    )

    samples = state.apply_fn(
        {"params": state.params},
        rng_key=next(rng),
        size=size,
        method=prior_cls.sample,
    )

    assert samples.shape == (size, num_actions)
    assert samples.min() >= 0 and samples.max() <= 1
