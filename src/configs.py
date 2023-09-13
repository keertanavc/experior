from pydantic import BaseModel, validator
from typing import Any, Optional, Union, Callable

import jax.numpy as jnp
from jax.random import uniform, normal

from src.commons import PRNGKey

VAR_BASELINES = ["opt"]
GRAD_ESTIMATORS = ["reinforce", "binforce"]
TRAINERS = ["MaxEnt", "minimax"]
REF_DISTS = [uniform, normal]


##################### Policy Configs #####################


class TransformerPolicyConfig(BaseModel):
    name: str = "transformer"
    n_blocks: int
    h_dim: int
    num_heads: int
    drop_p: float = 0.0
    dtype: Any = jnp.float32


class SoftElimPolicyConfig(BaseModel):
    name: str = "softelim"


##################### Prior Configs #####################


class BetaPriorConfig(BaseModel):
    name: str
    num_actions: int
    init_alpha: Optional[float]
    init_beta: Optional[float]
    epsilon: float = 1e-3


class MaxEntPriorConfig(BaseModel):
    name: str
    num_actions: int
    ref_dist: Callable

    # validate ref_dist
    @validator("ref_dist")
    def validate_ref_dist(cls, v):
        if v is not None and v.func not in REF_DISTS:
            raise ValueError(f"ref dist {v} not found in config")

        return v


##################### Other Configs #####################


class GradEstimatorConfig(BaseModel):
    name: str = "reinforce"
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


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]
    run_name: Optional[str]
    run_id: Optional[str]
    resume: bool = False
    # TODO handle log every steps for different time scales


class ExperiorConfig(BaseModel):
    policy: Union[TransformerPolicyConfig, SoftElimPolicyConfig]
    prior: Union[MaxEntPriorConfig, BetaPriorConfig]
    trainer: TrainerConfig
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
