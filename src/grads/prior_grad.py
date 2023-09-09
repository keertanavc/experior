import jax.numpy as jnp
import jax


def prior_grad_estimator(actions, mu_vectors, prior_log_p):
    """
    Args:
        actions: The history of actions, shape (batch_size, T).
        mu_vectors: The mean vectors of the prior, shape (batch_size, num_actions).
        prior_log_p: The log-probabilities of mu_vectors, shape (batch_size, num_actions).
    """
    # shape: (n_samples, horizon)
    means = mu_vectors[jnp.arange(mu_vectors.shape[0])[:, None], actions]
    T = actions.shape[1]

    max_means = jnp.max(mu_vectors, axis=1)  # shape: (n_samples, )

    indep_log_p = prior_log_p.sum(axis=1)
    return -(
        jax.lax.stop_gradient(T * max_means - means.sum(axis=1)) * indep_log_p
    ).mean()
