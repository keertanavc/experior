import optax

from src.utils import PRNGSequence
import tqdm

from flax.training import train_state
from src.models import BetaPrior

import jax
import jax.numpy as jnp

def test_prior():
    rng = PRNGSequence(123)
    K = 3
    n_samples = 1000
    epochs = 1000
    data = jax.random.beta(next(rng), a=2 * jnp.ones(K),
                           b=jnp.ones(K), shape=(n_samples, K))

    model = BetaPrior(num_actions=K)
    variables = model.init(next(rng), jnp.ones((1, K)))
    tx = optax.adam(learning_rate=1e-2)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)

    @jax.jit
    def update_step(state, batch):
        def loss_fn(params):
            log_prob = state.apply_fn({'params': params}, batch)
            return -jnp.mean(log_prob)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    with tqdm.tqdm(total=epochs) as pbar:
        for _ in range(epochs):
            state, loss = update_step(state, data)
            pbar.set_postfix({'loss': loss})
            pbar.update(1)

    assert jnp.isclose(state.params['alphas_sq']
                       ** 2 + BetaPrior.epsilon, 2, atol=1e-1).all()
    assert jnp.isclose(state.params['betas_sq'] **
                       2 + BetaPrior.epsilon, 1, atol=1e-1).all()


def test_prior_sample():
    K = 4
    size = 100
    model = BetaPrior(num_actions=K)
    rng = PRNGSequence(123)
    variables = model.init(next(rng), jnp.ones((1, K)))
    samples = model.apply({'params': variables['params']}, rng_key=next(
        rng), size=size, method=model.sample)
    
    assert samples.shape == (size, K)
    assert samples.min() >= 0 and samples.max() <= 1
