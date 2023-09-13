import jax.numpy as jnp
import jax


def prior_max_loss(actions, mu_vectors, prior_log_p):
    """The loss function for minimax training of the prior.
    Args:
        actions: The history of actions taken by the policy, shape (n_samples, T).
        mu_vectors: The mean vectors of the prior, shape (n_samples, num_actions).
        prior_log_p: The log-probabilities of mu_vectors, shape (n_samples, ).
    """
    # shape: (n_samples, horizon)
    means = mu_vectors[jnp.arange(mu_vectors.shape[0])[:, None], actions]
    T = actions.shape[1]

    max_means = jnp.max(mu_vectors, axis=1)  # shape: (n_samples, )

    indep_p = jnp.exp(prior_log_p)  # shape: (n_samples, )
    return -(
        jax.lax.stop_gradient(T * max_means - means.sum(axis=1))
        * indep_p
        / jax.lax.stop_gradient(indep_p)
    ).mean()



