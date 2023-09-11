import jax
import optax
import os
import json
import wandb

import orbax.checkpoint
import jax.numpy as jnp
import numpy as np

from flax.training import orbax_utils
from tqdm import tqdm
from collections import defaultdict

from src.configs import ExperiorConfig
from src.models import get_policy, get_prior, BetaPrior
from src.loss import get_policy_loss, prior_loss
from src.rollout import policy_rollout
from src.commons import PRNGKey
from src.baselines import get_baselines
from src.eval import bayes_regret


class BayesRegretTrainer:
    def __init__(self, conf: ExperiorConfig):
        self.conf = conf
        self.policy_state, self.prior_state = None, None
        self._policy_state, self._prior_step = None, None

        self.ckpt_manager = None

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        prior_opt = optax.adamw(learning_rate=self.conf.trainer.prior_lr)
        self.prior_state = get_prior(self.conf).create_state(
            prior_key, prior_opt, self.conf
        )

        policy_opt = optax.adamw(learning_rate=self.conf.trainer.policy_lr)
        self.policy_state = get_policy(self.conf).create_state(
            policy_key, policy_opt, self.conf
        )

        policy_loss = get_policy_loss(self.conf)

        if not self.conf.test_run:
            ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
                save_interval_steps=self.conf.save_every_steps,
                keep_period=self.conf.keep_every_steps,
            )

            self.ckpt_manager = orbax.checkpoint.CheckpointManager(
                self.conf.ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), ckpt_options
            )

        @jax.jit
        def _prior_step(key, policy_state, prior_state, batch):
            def policy_fn(k, t, a, r):
                return policy_state.apply_fn(
                    {"params": policy_state.params}, k, t, a, r
                )

            # shape of policy_log_p: (n_samples, horizon, num_actions)
            a, _, _ = policy_rollout(policy_fn, key, batch, self.conf.policy.horizon)

            def prior_loss_fn(prior_params):
                out = {}
                prior_log_p = prior_state.apply_fn({"params": prior_params}, batch)
                loss = prior_loss(a, batch, prior_log_p)
                out["loss"] = loss
                out["log_probs"] = prior_log_p.mean()

                # log prior_params
                # TODO maybe cleaner way to do this
                if issubclass(get_prior(self.conf), BetaPrior):
                    alpha_beta = jax.tree_map(lambda x: jnp.mean(x), prior_params)
                    out["alpha"] = alpha_beta["alphas_sq"]
                    out["beta"] = alpha_beta["betas_sq"]

                return loss, out

            (loss, out), prior_grads = jax.value_and_grad(prior_loss_fn, has_aux=True)(
                prior_state.params
            )

            prior_state = prior_state.apply_gradients(grads=prior_grads)

            return prior_state, out

        @jax.jit
        def _policy_step(key, policy_state, prior_state, batch):
            def policy_loss_fn(policy_params):
                def policy_fn(k, t, a, r):
                    return policy_state.apply_fn({"params": policy_params}, k, t, a, r)

                # shape of policy_log_p: (n_samples, horizon, num_actions)
                a, r, policy_log_p = policy_rollout(
                    policy_fn, key, batch, self.conf.policy.horizon
                )

                loss = policy_loss(a, r, policy_log_p, batch)
                out = {}
                out["loss"] = loss
                out["reward"] = r.mean()
                out["log_probs"] = policy_log_p.mean()

                return loss, out

            (loss, out), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_state.params)

            policy_state = policy_state.apply_gradients(grads=policy_grads)

            return policy_state, out

        self._policy_step = _policy_step
        self._prior_step = _prior_step

    def sample_envs(self, rng: PRNGKey):
        mu_vectors = self.prior_state.apply_fn(
            {"params": self.prior_state.params},
            rng_key=rng,
            size=self.conf.trainer.monte_carlo_samples,
            method="sample",
        )

        return jax.lax.stop_gradient(mu_vectors)

    def train_step(self, rng: PRNGKey, objective: str):
        assert objective in ["policy", "prior"]
        step_func = self._policy_step if objective == "policy" else self._prior_step

        b_size = self.conf.trainer.batch_size
        n_batches = self.conf.trainer.monte_carlo_samples // b_size
        rng, key = jax.random.split(rng)
        mu_vectors = self.sample_envs(key)

        output = defaultdict(list)
        for i in range(n_batches):
            batch = mu_vectors[i * b_size : (i + 1) * b_size]
            rng, key = jax.random.split(rng)
            state, aux = step_func(key, self.policy_state, self.prior_state, batch)
            if objective == "policy":
                self.policy_state = state
            else:
                self.prior_state = state
            for k, v in aux.items():
                output[k].append(v)

        return {f"{objective}/{k}": np.mean(v) for k, v in output.items()}

    def train(self, rng: PRNGKey):
        if self.conf.wandb.resume:
            ckpt = self.load_states()
            self.policy_state = ckpt["policy_model"]
            self.prior_state = ckpt["prior_model"]
            current_epoch = ckpt["epoch"] + 1
            rng = ckpt["rng"]
        else:
            current_epoch = 1

        prior_bar = tqdm(
            range(current_epoch, self.conf.trainer.epochs + 1), desc="Prior"
        )
        policy_bar = tqdm(
            range(current_epoch, self.conf.trainer.epochs + 1), desc="Policy"
        )

        for epoch in prior_bar:
            for i in range(1, self.conf.trainer.prior_steps + 1):
                rng, key = jax.random.split(rng)
                aux = self.train_step(key, "prior")
                prior_bar.set_postfix_str(
                    " ".join(["{}: {: .4f}".format(k, v) for (k, v) in aux.items()])
                )
                if not self.conf.test_run:
                    aux["prior/step"] = (epoch - 1) * self.conf.trainer.prior_steps + i
                    wandb.log(aux)
            prior_bar.update(1)

            for j in range(1, self.conf.trainer.policy_steps + 1):
                rng, key = jax.random.split(rng)
                aux = self.train_step(key, "policy")
                policy_bar.set_postfix_str(
                    " ".join(["{}: {: .4f}".format(k, v) for (k, v) in aux.items()])
                )
                if not self.conf.test_run:
                    aux["policy/step"] = (
                        epoch - 1
                    ) * self.conf.trainer.policy_steps + j
                    wandb.log(aux)
            policy_bar.update(1)

            if not self.conf.test_run:
                wandb.log({"epoch": epoch})
                self.save_states(epoch, rng)

    def save_metrics(self, rng: PRNGKey):
        save_path = os.path.join(self.conf.out_dir, "metrics.json")

        def policy_fn(key, t, a, r):
            return self.policy_state.apply_fn(
                {"params": self.policy_state.params}, key, t, a, r
            )

        models = get_baselines(self.conf)
        models["ours"] = policy_fn

        metrics = {}

        for name, model in models.items():
            rng, key = jax.random.split(rng)
            regret = bayes_regret(
                key,
                model,
                self.conf.prior.num_actions,
                self.conf.trainer.max_horizon,
                self.conf.trainer.monte_carlo_samples,
            )
            metrics[name] = regret.tolist()
            wandb.log({f"policy/{name}_regret": regret[-1]})

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

    def load_states(self, step=None):
        target = {
            "policy_model": self.policy_state,
            "prior_model": self.prior_state,
            "epoch": 0,
            "rng": jax.random.PRNGKey(0),
        }

        step = step or self.ckpt_manager.latest_step()

        return self.ckpt_manager.restore(step, items=target)
