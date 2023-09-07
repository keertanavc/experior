import jax
import optax
import os
import json
import wandb

import jax.numpy as jnp
import orbax.checkpoint

from flax.training import orbax_utils
from tqdm import tqdm

from src.configs import ExperiorConfig
from src.models import get_policy, get_prior
from src.rollout import policy_rollout
from src.utils import PRNGKey, PRNGSequence
from src.baselines import get_baselines
from src.eval import uniform_bayes_regret


class BayesRegretTrainer:
    def __init__(self, conf: ExperiorConfig):
        self.conf = conf
        self.policy_state, self.prior_state = None, None
        self._update_step = None

        self.ckpt_manager = None

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        prior_opt = optax.adamw(learning_rate=self.conf.trainer.prior_lr)
        prior_cls = get_prior(self.conf)
        self.prior_state = prior_cls.create_state(prior_key, prior_opt, self.conf)

        policy_opt = optax.adamw(learning_rate=self.conf.trainer.policy_lr)
        policy_cls = get_policy(self.conf)
        self.policy_state = policy_cls.create_state(policy_key, policy_opt, self.conf)

        if not self.conf.test_run:
            ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
                save_interval_steps=self.conf.save_every_steps,
                keep_period=self.conf.keep_every_steps,
            )

            self.ckpt_manager = orbax.checkpoint.CheckpointManager(
                self.conf.ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), ckpt_options
            )

        @jax.jit
        def _update_step(key, policy_state, prior_state, mu_vectors):
            # sample envs
            key, rng_key = jax.random.split(key)

            def loss_fn(policy_params, prior_params):
                def policy_fn(k, t, a, r):
                    return policy_state.apply_fn({"params": policy_params}, k, t, a, r)

                # shape of policy_log_p: (n_samples, horizon, num_actions)
                a, r, policy_log_p = policy_rollout(
                    policy_fn, rng_key, mu_vectors, self.conf.policy.horizon
                )

                # shape: (n_samples, horizon)
                means = mu_vectors[jnp.arange(self.conf.trainer.batch_size)[:, None], a]

                max_means = jnp.max(mu_vectors, axis=1)  # shape: (n_samples,)

                # shape: (n_samples, horizon)
                rtg = jax.lax.cumsum(means, axis=1, reverse=True)

                # shape: (n_samples, horizon, num_actions)
                #  TODO check this (do we need to sum over actions)
                action_rtg = (rtg - means)[:, :, None] + mu_vectors[:, None, :]
                policy_probs = jax.lax.stop_gradient(jnp.exp(policy_log_p))

                loss = (
                    -(action_rtg * policy_log_p * policy_probs)
                    .sum(axis=2)
                    .sum(axis=1)
                    .mean()
                )
                out = {"policy_loss": loss}

                if not self.conf.fix_prior:
                    # shape: (n_envs, num_actions)
                    prior_log_p = prior_state.apply_fn(
                        {"params": prior_params}, mu_vectors
                    )

                    # TODO: make sure this is correct
                    prior_log_p = jnp.nan_to_num(
                        prior_log_p, copy=False, neginf=-10, posinf=10
                    )
                    T = means.shape[1]
                    prior_loss = -(
                        jax.lax.stop_gradient(T * max_means - means.sum(axis=1))
                        * prior_log_p.sum(axis=1)
                    ).mean()
                    out["prior_loss"] = prior_loss
                    out["prior_log_probs"] = prior_log_p.mean()
                    out["max_means"] = max_means.mean()

                return loss, {
                    "loss": loss,
                    **out,
                    "reward": r.mean(),
                    "policy_log_probs": policy_log_p.mean(),
                }

            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, argnums=(0, 1)
            )(policy_state.params, prior_state.params)

            policy_grads, prior_grads = grads
            policy_state = policy_state.apply_gradients(grads=policy_grads)
            if not self.conf.fix_prior:
                prior_state = prior_state.apply_gradients(grads=prior_grads)

            return policy_state, prior_state, loss, aux

        self._update_step = _update_step

    def sample_envs(self, rng: PRNGKey):
        prior_cls = get_prior(self.conf)
        mu_vectors = self.prior_state.apply_fn(
            {"params": self.prior_state.params},
            rng_key=rng,
            size=self.conf.trainer.monte_carlo_samples,
            method=prior_cls.sample,
        )

        return jax.lax.stop_gradient(mu_vectors)

    def train_step(self, key: PRNGKey, batch):
        self.policy_state, self.prior_state, loss, aux = self._update_step(
            key, self.policy_state, self.prior_state, batch
        )
        return loss, aux

    def train(self, rng: PRNGKey):
        if self.conf.wandb.resume:
            # load from latest checkpoint
            ckpt = self.load_states()
            self.policy_state = ckpt["policy_model"]
            self.prior_state = ckpt["prior_model"]
            current_epoch = ckpt["epoch"] + 1
            rng = ckpt["rng"]
        else:
            current_epoch = 1
        pbar = tqdm(range(current_epoch, self.conf.trainer.epochs + 1))

        for epoch in pbar:
            rng, key = jax.random.split(rng)
            mu_vectors = self.sample_envs(key)
            b_size = self.conf.trainer.batch_size
            n_batches = self.conf.trainer.monte_carlo_samples // b_size

            for i in range(n_batches):
                batch = mu_vectors[i * b_size : (i + 1) * b_size]
                rng, key = jax.random.split(rng)
                _, aux = self.train_step(key, batch)
                log_str = " ".join(
                    ["{}: {: .4f}".format(k, v) for (k, v) in aux.items()]
                )
                pbar.set_postfix_str(log_str)
            pbar.update(1)

            # TODO log average metrics for the epoch
            if epoch % self.conf.wandb.log_every_steps == 0 and not self.conf.test_run:
                wandb.log(aux, step=epoch)

            if not self.conf.test_run:
                self.save_states(epoch, rng)

    def save_metrics(self, rng: PRNGKey):
        save_path = os.path.join(self.conf.out_dir, "metrics.json")

        def policy_fn(key, t, a, r):
            return self.policy_state.apply_fn(
                {"params": self.policy_state.params}, key, t, a, r
            )

        models = get_baselines(self.conf)
        models[self.conf.policy.name] = policy_fn

        metrics = {}

        for name, model in models.items():
            rng, key = jax.random.split(rng)
            regret = uniform_bayes_regret(
                key,
                model,
                self.conf.prior.num_actions,
                self.conf.trainer.max_horizon,
                self.conf.trainer.monte_carlo_samples,
            )
            metrics[name] = regret.tolist()

        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

    def save_states(self, epoch, rng):
        ckpt = {
            "policy_model": self.policy_state,
            "prior_model": self.prior_state,
            "epoch": epoch,
            "rng": rng,
        }

        save_args = orbax_utils.save_args_from_target(ckpt)
        self.ckpt_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args})

    def load_states(self):
        target = {
            "policy_model": self.policy_state,
            "prior_model": self.prior_state,
            "epoch": 0,
            "rng": jax.random.PRNGKey(0),
        }

        return self.ckpt_manager.restore(self.ckpt_manager.latest_step(), items=target)
