import jax
import jax.random as random
import jax.numpy as jnp


def thompson_sampling(mu_vectors, T, key):
    K = len(mu_vectors)
    alpha = jnp.ones(K)  # Prior alpha for Beta distribution
    beta = jnp.ones(K)  # Prior beta for Beta distribution

    rewards = []
    actions = []

    for t in range(T):
        key, rng = random.split(key)
        # Sample from the current posterior (Beta distribution)
        sampled_theta = random.beta(rng, alpha, beta)
        # Select the action with the highest sampled value
        action = jnp.argmax(sampled_theta)
        # Sample the reward from Bernoulli distribution with mean mu_vectors[action]
        key, rng = random.split(key)
        reward = random.bernoulli(rng, mu_vectors[action])

        # Update the posterior based on the observed reward
        alpha += jnp.where(jnp.arange(K) == action, reward, 0)
        beta += jnp.where(jnp.arange(K) == action, 1 - reward, 0)

        rewards.append(reward)
        actions.append(action)

    return jnp.array(actions), jnp.array(rewards)
