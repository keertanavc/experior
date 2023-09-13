import jax
import optax
import wandb

import jax.numpy as jnp
from tqdm import tqdm

from src.configs import ExperiorConfig
from src.losses import get_policy_loss
from src.rollout import policy_rollout
from src.models import get_policy, get_prior
from src.commons import PRNGKey
from .trainer import Trainer


class MaxEntTrainer(Trainer):
    def __init__(self, conf: ExperiorConfig):
        super().__init__(conf)

    def initialize(self, rng: PRNGKey):
        policy_key, prior_key = jax.random.split(rng)

        # TODO read the expert policy here
        self.expert_policy = (
            jnp.ones((self.conf.prior.num_actions,)) / self.conf.prior.num_actions
        )

        prior_opt = optax.adamw(learning_rate=self.conf.trainer.prior_trainer.lr)
        self.prior_state = get_prior(self.conf).create_state(
            prior_key, prior_opt, self.conf, self.expert_policy
        )

        policy_opt = optax.adamw(learning_rate=self.conf.trainer.policy_trainer.lr)
        self.policy_state = get_policy(self.conf).create_state(
            policy_key, policy_opt, self.conf
        )
        policy_loss = get_policy_loss(self.conf)

        @jax.jit
        def _prior_step(key, policy_state, prior_state, batch):
            def prior_loss_fn(prior_params):
                out = {}
                unnormal_prob = prior_state.apply_fn({"params": prior_params}, batch)
                loss = jnp.log(unnormal_prob.mean())
                out["loss"] = loss
                out["unnormal_probs"] = unnormal_prob.mean()
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

                density = prior_state.apply_fn({"params": prior_state.params}, batch)

                # shape of policy_log_p: (n_samples, horizon, num_actions)
                a, r, policy_log_p = policy_rollout(
                    policy_fn, key, batch, self.conf.trainer.train_horizon
                )

                loss = policy_loss(a, r, policy_log_p, batch, density)
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

        # first train the prior
        for epoch in prior_bar:
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

            if not self.conf.test_run:
                wandb.log({"prior/epoch": epoch})
                self.save_states(epoch, rng)

        for epoch in policy_bar:
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
                wandb.log({"policy/epoch": epoch})
                self.save_states(epoch, rng)
