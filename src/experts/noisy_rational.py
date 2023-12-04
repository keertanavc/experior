import chex
import jax
import jax.numpy as jnp

from src.envs import *
from src.commons import BaseConfig

from gymnax.environments import spaces


class NoisyRationalExpertConfig(BaseConfig):
    name: str = "NoisyRationalExpert"
    env_param: EnvParams
    beta: float
    action_n_samples: int


def make_expert_log_likelihood_fn(conf: NoisyRationalExpertConfig):
    def expert_log_likelihood_fn(
        key: chex.PRNGKey, unobserved_context: UnobservedContext, trajectory: Trajectory
    ):
        contexts = trajectory.context  # n_trajectory x context_dim
        actions = trajectory.action  # n_trajectory x action_dim

        # n_trajectory x 1
        q_star_values = conf.env_param.Q_function(unobserved_context, contexts, actions)

        if isinstance(conf.env_param.action_space, spaces.Discrete):
            sampled_actions = jnp.arange(conf.env_param.action_space.n)  # TODO fix
        else:
            sampled_actions = conf.env_param.action_space.sample(
                key, size=conf.action_n_samples
            )  # action_n_samples x action_dim

        all_means = jax.vmap(conf.env_param.Q_function, in_axes=(None, None, 0))(
            unobserved_context, contexts, sampled_actions
        )  # action_n_samples x n_trajectory x 1

        denominator = jax.scipy.special.logsumexp(
            conf.beta * all_means, axis=0
        )  # n_trajectory x 1

        numerator = conf.beta * q_star_values
        return numerator - denominator

    return expert_log_likelihood_fn


def generate_optimal_expert_trajectories(
    rng: chex.PRNGKey, sample_size: int, bandit_params: EnvParams
) -> Trajectory:
    k1, k2 = jax.random.split(rng)
    contexts = bandit_params.init_context_dist_fn(k1, sample_size)
    unobserved_contexts = bandit_params.prior_fn(k2, sample_size)
    opt_actions, _ = bandit_params.best_action_value_fn(unobserved_contexts, contexts)
    trajectories = Trajectory(action=opt_actions, context=contexts, reward=None)
    return trajectories
