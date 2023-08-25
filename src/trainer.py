import jax
import optax
import os
import json

import jax.numpy as jnp

from src.configs import ExperiorConfig
from src.models import Policy, BetaPrior
from src.rollout import policy_rollout
from src.utils import PRNGKey
from src.baselines import BernoulliTS
from src.eval import uniform_bayes_regret


class BayesRegretTrainer:

    def __init__(self, conf: ExperiorConfig):
        self.conf = conf
        self.policy_state, self.prior_state = None, None
        self._update_step = None

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        prior_opt = optax.adamw(learning_rate=self.conf.train.prior_lr)
        self.prior_state = BetaPrior.create_state(
            prior_key, prior_opt, self.conf.prior)

        policy_opt = optax.adamw(learning_rate=self.conf.train.policy_lr)
        self.policy_state = Policy.create_state(
            policy_key, policy_opt, self.conf.policy)

        @jax.jit
        def _update_step(key, policy_state, prior_state, mu_vectors):

            # sample envs
            key, rng_key = jax.random.split(key)

            def loss_fn(policy_params, prior_params):
                def policy_fn(k, t, a, r):
                    return policy_state.apply_fn(
                        {'params': policy_params}, k, t, a, r)

                # shape of policy_log_p: (n_samples, horizon, num_actions)
                a, r, policy_log_p = policy_rollout(
                    policy_fn, rng_key, mu_vectors, self.conf.policy.horizon)

                # shape: (n_samples, horizon)
                means = mu_vectors[jnp.arange(self.conf.train.batch_size)[
                    :, None], a]

                max_means = jnp.max(mu_vectors, axis=1)  # shape: (n_samples,)

                # shape: (n_samples, horizon)
                rtg = jax.lax.cumsum(means, axis=1, reverse=True)

                # shape: (n_samples, horizon, num_actions) TODO check this
                action_rtg = (rtg - means)[:, :, None] + mu_vectors[:, None, :]
                policy_probs = jax.lax.stop_gradient(jnp.exp(policy_log_p))

                loss = - (action_rtg * policy_log_p *
                          policy_probs).sum(axis=2).sum(axis=1).mean()
                out = {'policy_loss': loss}

                if not self.conf.fix_prior:
                    # shape: (n_envs, num_actions)
                    prior_log_p = prior_state.apply_fn(
                        {'params': prior_params}, mu_vectors)

                    # TODO: make sure this is correct
                    prior_log_p = jnp.nan_to_num(
                        prior_log_p, copy=False, neginf=-10, posinf=10)
                    T = means.shape[1]
                    prior_loss = - (jax.lax.stop_gradient(T * max_means -
                                    means.sum(axis=1)) * prior_log_p.sum(axis=1)).mean()
                    out['prior_loss'] = prior_loss
                    out['prior_log_probs'] = prior_log_p.mean()
                    out['max_means'] = max_means.mean()

                return loss, {'loss': loss,
                              **out,
                              'reward': r.mean(),
                              'policy_log_probs': policy_log_p.mean()}

            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, argnums=(0, 1))(policy_state.params, prior_state.params)

            policy_grads, prior_grads = grads
            policy_state = policy_state.apply_gradients(grads=policy_grads)
            if not self.conf.fix_prior:
                prior_state = prior_state.apply_gradients(
                    grads=prior_grads)

            return policy_state, prior_state, loss, aux

        self._update_step = _update_step

    def sample_envs(self, rng: PRNGKey):
        mu_vectors = self.prior_state.apply_fn(
            {'params': self.prior_state.params}, rng_key=rng,
            size=self.conf.train.monte_carlo_samples, method=BetaPrior.sample)

        return jax.lax.stop_gradient(mu_vectors)

    def train_step(self, key: PRNGKey, mu_vectors):
        self.policy_state, self.prior_state, loss, aux = self._update_step(
            key, self.policy_state, self.prior_state, mu_vectors)
        return loss, aux

    def save_metrics(self, rng: PRNGKey):
        save_path = os.path.join(self.conf.out_dir, "metrics.json")

        def policy_fn(key, t, a, r):
            return self.policy_state.apply_fn(
                {'params': self.policy_state.params}, key, t, a, r)

        ts = BernoulliTS(self.conf.policy.num_actions)
        key1, key2 = jax.random.split(rng)
        our_regret = uniform_bayes_regret(
            key1, policy_fn, self.conf.policy.num_actions, self.conf.policy.horizon, self.conf.train.monte_carlo_samples)
        ts_regret = uniform_bayes_regret(
            key2, ts, self.conf.policy.num_actions, self.conf.policy.horizon, self.conf.train.monte_carlo_samples)

        metrics = {'our_regret': our_regret.tolist(),
                   'ts_regret': ts_regret.tolist()}

        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

    # def save_states(self, step: int):
    # from flax.training import checkpoints  # TODO migrate to orbax
    #     checkpoints.save_checkpoint_multiprocess(ckpt_dir=self.conf.ckpt_dir,
    #                                              target=self.policy_state,
    #                                              step=step,
    #                                              keep_every_n_steps=self.conf.keep_every_steps)
    #     checkpoints.save_checkpoint(
    #         self.conf.checkpoint_dir, self.prior_state, step)
