from pydantic import BaseModel, validator
from typing import Any, Optional, Union, Callable, List

import jax.numpy as jnp
from jax.random import uniform, normal

from src.commons import PRNGKey

VAR_BASELINES = ["opt"]
GRAD_ESTIMATORS = ["reinforce", "binforce"]
TRAINERS = ["MaxEnt", "minimax", "mle"]
REF_DISTS = [uniform, normal]


##################### Prior Configs #####################


class BetaPriorConfig(BaseModel):
    name: str = "beta"
    num_actions: int
    init_alpha: Optional[Union[float, List[float]]]
    init_beta: Optional[Union[float, List[float]]]
    epsilon: float = 1e-3

    @validator("init_alpha")
    def validate_init_alpha(cls, v, values, field, config):
        if v is not None:
            if isinstance(v, float):
                v = [v] * values["num_actions"]
            assert (
                len(v) == values["num_actions"]
            ), "init_alpha must be a list of length num_actions"
        return v

    @validator("init_beta")
    def validate_init_beta(cls, v, values, field, config):
        if v is not None:
            if isinstance(v, float):
                v = [v] * values["num_actions"]
            assert (
                len(v) == values["num_actions"]
            ), "init_beta must be a list of length num_actions"
        return v


class MaxEntPriorConfig(BaseModel):
    name: str = "MaxEnt"
    num_actions: int
    ref_dist: Callable

    # validate ref_dist
    @validator("ref_dist")
    def validate_ref_dist(cls, v):
        if v is not None and v.func not in REF_DISTS:
            raise ValueError(f"ref dist {v} not found in config")

        return v


##################### Policy Configs #####################


class TransformerPolicyConfig(BaseModel):
    name: str = "transformer"
    num_actions: Optional[int]
    horizon: Optional[int]
    n_blocks: int
    h_dim: int
    num_heads: int
    drop_p: float = 0.0
    dtype: Any = jnp.float32


class SoftElimPolicyConfig(BaseModel):
    name: str = "softelim"
    num_actions: Optional[int]


class BetaTSPolicyConfig(BaseModel):
    name: str = "beta_ts"
    num_actions: Optional[int]
    prior: BetaPriorConfig


##################### Expert Configs #####################


class SyntheticExpertConfig(BaseModel):
    name: str = "synthetic"
    prior: BetaPriorConfig
    mc_samples: int = 1000  # number of samples from the prior to estimate the policy


##################### Trainer Configs #####################


class GradEstimatorConfig(BaseModel):
    name: str
    var_baseline: Optional[str]

    # validate baseline
    @validator("var_baseline")
    def validate_baseline(cls, v):
        if v is not None and v not in VAR_BASELINES:
            raise ValueError(f"var baseline {v} not found in config")
        return v

    # validate grad estimator
    @validator("name")
    def validate_grad_estimator(cls, v):
        if v not in GRAD_ESTIMATORS:
            raise ValueError(f"grad estimator {v} not found in config")
        return v


class ModelTrainerConfig(BaseModel):
    lr: float
    mc_samples: int
    epochs: int
    batch_size: int
    grad_est: Optional[GradEstimatorConfig]
    steps: int = 1  # steps per epoch (zero means fixed model - no training)


class TrainerConfig(BaseModel):
    name: str
    policy_trainer: ModelTrainerConfig
    prior_trainer: ModelTrainerConfig
    test_horizon: int
    train_horizon: int

    @validator("name")
    def validate_name(cls, v):
        if v not in TRAINERS:
            raise ValueError(f"trainer {v} not found in config")
        return v

    @validator("prior_trainer")
    def validate_trainer(cls, prior_trainer_val, values, field, config):
        policy_trainer_val = values["policy_trainer"]
        if values["name"] == "minimax":
            assert (
                prior_trainer_val.epochs == policy_trainer_val.epochs
            ), "Minimax trainer must have same number of epochs for both policy and prior"

        return prior_trainer_val


##################### Other Configs #####################


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]
    run_name: Optional[str]
    run_id: Optional[str]
    resume: bool = False
    # TODO handle log every steps for different time scales


class ExperiorConfig(BaseModel):
    prior: Union[MaxEntPriorConfig, BetaPriorConfig]
    trainer: TrainerConfig
    policy: Union[TransformerPolicyConfig, BetaTSPolicyConfig, SoftElimPolicyConfig]
    expert: SyntheticExpertConfig
    seed: int
    test_run: bool
    wandb: WandbConfig
    out_dir: str
    save_every_steps: int
    keep_every_steps: int
    ckpt_dir: Optional[str]

    # validate trainer
    @validator("trainer")
    def validate_trainer(cls, trainer_val, values, field, config):
        prior = values["prior"]
        if trainer_val.name == "MaxEnt":
            assert prior.name == "MaxEnt", "MaxEnt trainer must have MaxEnt prior"

        return trainer_val

    # validate policy
    @validator("policy")
    def validate_policy(cls, policy_val, values, field, config):
        prior = values["prior"]
        trainer = values["trainer"]
        if policy_val.name == "transformer":
            max_horizon = max(trainer.test_horizon, trainer.train_horizon)
            if policy_val.horizon:
                assert (
                    policy_val.horizon >= max_horizon
                ), "transformer policy horizon must be greater than train/test horizon"
            else:
                policy_val.horizon = max_horizon

        if policy_val.num_actions:
            assert (
                policy_val.num_actions == prior.num_actions
            ), "policy and prior must have same number of actions"
        else:
            policy_val.num_actions = prior.num_actions

        if trainer.name == "mle":
            assert policy_val.name == "beta_ts", "MLE trainer must have beta_ts policy"

        return policy_val

    # validate expert
    @validator("expert")
    def validate_expert(cls, expert_val, values, field, config):
        prior = values["prior"]
        if expert_val.name == "synthetic":
            assert (
                expert_val.prior.num_actions == prior.num_actions
            ), "expert and prior must have same number of actions"

        return expert_val
