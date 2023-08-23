from src.configs import TransformerPolicyConfig, OptimizerConfig
from src.models import Policy, BetaPrior

from src.utils import PRNGKey

import jax
import jax.numpy as jnp

import optax
from flax.training import train_state


class BayesRegretTrainer:

    def __init__(self, policy_config: TransformerPolicyConfig, optimizer_config: OptimizerConfig):
        self.pc = policy_config
        self.oc = optimizer_config
        self.policy_state, self.prior_state = None, None
        self._update_step = None

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        prior_model = BetaPrior(num_actions=self.pc.num_actions)
        variables = prior_model.init(
            prior_key, jnp.ones((1, self.pc.num_actions)))
        prior_tx = optax.adam(learning_rate=self.oc.prior_lr)
        self.prior_state = train_state.TrainState.create(
            apply_fn=prior_model.apply, params=variables['params'], tx=prior_tx)

        policy = Policy(config=self.pc)
        variables = policy.init(policy_key, jnp.ones(
            (1, 2), dtype=jnp.int32), jnp.ones((1, 2), dtype=jnp.int32), jnp.ones((1, 2)))
        policy_tx = optax.adam(learning_rate=self.oc.policy_lr)
        self.policy_state = train_state.TrainState.create(
            apply_fn=policy.apply, params=variables['params'], tx=policy_tx)

        @jax.jit
        def _update_step(key, policy_state, prior_state):

            key, rng_key = jax.random.split(key)
            mu_vectors = prior_state.apply_fn(
                {'params': prior_state.params}, rng_key=rng_key, size=self.oc.mc_samples, method=prior_model.sample)

            mu_vectors = jax.lax.stop_gradient(mu_vectors)

            def loss_fn(policy_params, prior_params):
                actions, rewards, log_policy_probs = policy_state.apply_fn({'params': policy_params},
                                                                           key,
                                                                           mu_vectors,
                                                                           method=policy.policy_rollout)

                log_prior_probs = prior_state.apply_fn(
                    {'params': prior_params}, mu_vectors)  # shape: (n_samples, num_actions)

                # TODO: make sure this is correct
                log_prior_probs = jnp.nan_to_num(
                    log_prior_probs, copy=False, neginf=-10, posinf=10)

                # shape: (n_samples, horizon)
                means = mu_vectors[jnp.arange(self.oc.mc_samples)[
                    :, None], actions]

                max_means = jnp.max(mu_vectors, axis=1)  # shape: (n_samples,)

                # shape: (n_samples, horizon)
                rtg = jax.lax.cumsum(means, axis=1, reverse=True)
                # make sure we don't propagate gradients through the return-to-go
                rtg = jax.lax.stop_gradient(rtg)

                policy_loss = - (rtg * log_policy_probs).sum(axis=1).mean()

                T = means.shape[1]
                prior_loss = - (jax.lax.stop_gradient(T * max_means -
                                means.sum(axis=1)) * log_prior_probs.sum(axis=1)).mean()

                return prior_loss + policy_loss, {'policy_loss': policy_loss,
                                                  'prior_loss': prior_loss,
                                                  'max_means': max_means.mean(),
                                                  'rtg': rtg.mean(),
                                                  'reward': rewards.mean(),
                                                  'prior_log_probs': log_prior_probs.mean(),
                                                  'policy_log_probs': log_policy_probs.mean()}

            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, argnums=(0, 1))(policy_state.params, prior_state.params)

            policy_grads, prior_grads = grads
            new_policy_state = policy_state.apply_gradients(grads=policy_grads)
            new_prior_state = prior_state.apply_gradients(grads=prior_grads)

            return new_policy_state, new_prior_state, loss, aux

        self._update_step = _update_step

    def train_step(self, key: PRNGKey):
        self.policy_state, self.prior_state, loss, aux = self._update_step(
            key, self.policy_state, self.prior_state)
        return loss, aux
