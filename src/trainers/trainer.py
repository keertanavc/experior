import jax
import os
import json
import wandb

import orbax.checkpoint
import numpy as np

from flax.training import orbax_utils
from collections import defaultdict
from pprint import pprint

from src.configs import ExperiorConfig, BetaTSPolicyConfig
from src.commons import PRNGKey
from src.experts import Expert, SyntheticExpert
from src.baselines import BernoulliTS
from src.models import BetaTSPolicy
from src.eval import bayes_regret


class Trainer:
    def __init__(self, conf: ExperiorConfig, expert: Expert):
        self.conf = conf
        self.policy_state, self.prior_state = None, None
        self.expert = expert

        # training steps - take key, policy_state, prior_state, batch of mu vectors
        # return updated state and logs
        self._policy_step, self._prior_step = None, None

        if not self.conf.test_run:
            ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
                save_interval_steps=self.conf.save_every_steps,
                keep_period=self.conf.keep_every_steps,
            )

            self.ckpt_manager = orbax.checkpoint.CheckpointManager(
                self.conf.ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), ckpt_options
            )
        else:
            self.ckpt_manager = None

    def initialize(self, rng: PRNGKey):
        # define training steps here
        raise NotImplementedError

    def train(self, rng: PRNGKey):
        raise NotImplementedError

    def train_step(self, rng: PRNGKey, objective: str):
        assert objective in ["policy", "prior"]
        step_func = self._policy_step if objective == "policy" else self._prior_step
        trainer_conf = (
            self.conf.trainer.policy_trainer
            if objective == "policy"
            else self.conf.trainer.prior_trainer
        )

        b_size = trainer_conf.batch_size
        n_batches = trainer_conf.mc_samples // b_size
        rng, key = jax.random.split(rng)
        mu_vectors = self._sample_envs(key, trainer_conf.mc_samples)

        output = defaultdict(list)
        for i in range(n_batches):
            batch = mu_vectors[i * b_size : (i + 1) * b_size]
            rng, key = jax.random.split(rng)
            state, logs = step_func(key, self.policy_state, self.prior_state, batch)
            if objective == "policy":
                self.policy_state = state
            else:
                self.prior_state = state
            for k, v in logs.items():
                output[k].append(v)

        return {f"{objective}/{k}": np.mean(v) for k, v in output.items()}

    def _sample_envs(self, rng: PRNGKey, size):
        mu_vectors = self.prior_state.apply_fn(
            {"params": self.prior_state.params},
            rng_key=rng,
            size=size,
            method="sample",
        )

        return jax.lax.stop_gradient(mu_vectors)

    def save_metrics(self, rng: PRNGKey):
        # Save uniform and expert prior regret
        save_path = os.path.join(self.conf.out_dir, "metrics.json")

        def policy_fn(key, t, a, r):
            return self.policy_state.apply_fn(
                {"params": self.policy_state.params}, key, t, a, r
            )

        models = {"BernoulliTS": BernoulliTS(self.conf.prior.num_actions)}
        models["ours"] = policy_fn

        priors = {"uniform": None}
        if isinstance(self.expert, SyntheticExpert):
            priors["expert"] = lambda key, size: self.expert.prior_state.apply_fn(
                {"params": self.expert.prior_state.params},
                rng_key=key,
                size=size,
                method="sample",
            )
            rng, key = jax.random.split(rng)

            # define a TS model w.r.t. expert prior
            conf = BetaTSPolicyConfig(
                num_actions=self.conf.prior.num_actions, prior=self.conf.expert.prior
            )
            beta_ts_state = BetaTSPolicy.create_state(key, None, conf)
            models[
                "BernoulliTS_TruePrior"
            ] = lambda key, t, a, r: beta_ts_state.apply_fn(
                {"params": beta_ts_state.params}, key, t, a, r
            )

        metrics = {}
        final_regrets = {}

        for name, model in models.items():
            metrics[name] = {}
            for p, prior_fn in priors.items():
                rng, key = jax.random.split(rng)
                regret = bayes_regret(
                    key,
                    model,
                    self.conf.prior.num_actions,
                    self.conf.trainer.test_horizon,
                    self.conf.trainer.policy_trainer.mc_samples,
                    prior_fn=prior_fn,
                )
                metrics[name][p] = regret.tolist()
                final_regrets[f"{name}_{p}"] = float(regret[-1])
                if not self.conf.test_run:
                    wandb.log({f"policy/{name}_{p}_regret": regret[-1]})

        if not self.conf.test_run:
            with open(save_path, "w") as fp:
                json.dump(metrics, fp, indent=2)
        else:
            pprint(final_regrets)

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
        """Loads states of the trainer.

        Returns:
            dict: with keys `policy_model`, `prior_model`, `epoch`, and `rng`.
        """
        target = {
            "policy_model": self.policy_state,
            "prior_model": self.prior_state,
            "epoch": 0,
            "rng": jax.random.PRNGKey(0),
        }

        step = step or self.ckpt_manager.latest_step()

        return self.ckpt_manager.restore(step, items=target)
