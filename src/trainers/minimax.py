import jax
import optax
import wandb

import jax.numpy as jnp
from tqdm import tqdm

from src.configs import ExperiorConfig
from src.experts import Expert
from src.losses import get_policy_loss, prior_max_loss
from src.rollout import policy_rollout
from src.models import get_policy, get_prior
from src.commons import PRNGKey
from .trainer import Trainer


class MiniMaxTrainer(Trainer):
    def __init__(self, conf: ExperiorConfig, expert: Expert):
        super().__init__(conf, expert)

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        prior_opt = optax.adamw(learning_rate=self.conf.trainer.prior_trainer.lr)
        self.prior_state = get_prior(self.conf.prior.name).create_state(
            prior_key, prior_opt, self.conf.prior
        )

        policy_opt = optax.adamw(learning_rate=self.conf.trainer.policy_trainer.lr)
        self.policy_state = get_policy(self.conf.policy.name).create_state(
            policy_key, policy_opt, self.conf.policy
        )
        policy_loss = get_policy_loss(self.conf.trainer.policy_trainer.grad_est)

        @jax.jit
        def _prior_step(key, policy_state, prior_state, batch):
            def policy_fn(k, t, a, r):
                return policy_state.apply_fn(
                    {"params": policy_state.params}, k, t, a, r
                )

            # shape of policy_log_p: (n_samples, horizon, num_actions)
            a, _, _ = policy_rollout(
                policy_fn, key, batch, self.conf.trainer.train_horizon
            )

            def prior_loss_fn(prior_params):
                out = {}
                prior_log_p = prior_state.apply_fn({"params": prior_params}, batch)
                loss = prior_max_loss(a, batch, prior_log_p)
                out["loss"] = loss
                out["log_probs"] = prior_log_p.mean()

                # log prior_params
                if self.conf.prior.name == "beta":
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
                    policy_fn, key, batch, self.conf.trainer.train_horizon
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

    def train(self, rng: PRNGKey):
        if self.conf.wandb.resume:
            ckpt = self.load_states()
            self.policy_state = ckpt["policy_model"]
            self.prior_state = ckpt["prior_model"]
            current_epoch = ckpt["epoch"] + 1
            rng = ckpt["rng"]
        else:
            current_epoch = 1

        prior_conf = self.conf.trainer.prior_trainer
        policy_conf = self.conf.trainer.policy_trainer

        prior_bar = tqdm(range(current_epoch, prior_conf.epochs + 1), desc="Prior")
        policy_bar = tqdm(range(current_epoch, policy_conf.epochs + 1), desc="Policy")

        for epoch in policy_bar:
            for i in range(1, prior_conf.steps + 1):
                rng, key = jax.random.split(rng)
                aux = self.train_step(key, "prior")
                prior_bar.set_postfix_str(
                    " ".join(["{}: {: .4f}".format(k, v) for (k, v) in aux.items()])
                )
                if not self.conf.test_run:
                    aux["prior/step"] = (epoch - 1) * prior_conf.steps + i
                    wandb.log(aux)
            prior_bar.update(1)

            for j in range(1, policy_conf.steps + 1):
                rng, key = jax.random.split(rng)
                aux = self.train_step(key, "policy")
                policy_bar.set_postfix_str(
                    " ".join(["{}: {: .4f}".format(k, v) for (k, v) in aux.items()])
                )
                if not self.conf.test_run:
                    aux["policy/step"] = (epoch - 1) * policy_conf.steps + j
                    wandb.log(aux)
            policy_bar.update(1)

            if not self.conf.test_run:
                wandb.log({"epoch": epoch})
                self.save_states(epoch, rng)
