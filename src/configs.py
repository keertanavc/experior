from pydantic import BaseModel, validator
from typing import Any, Optional, Union

import jax.numpy as jnp

VAR_BASELINES = ["opt"]
GRAD_ESTIMATORS = ["reinforce", "binforce"]


class TransformerPolicyConfig(BaseModel):
    name: str = "transformer"
    horizon: int
    n_blocks: int
    h_dim: int
    num_heads: int
    drop_p: float = 0.0
    dtype: Any = jnp.float32


class SoftElimPolicyConfig(BaseModel):
    name: str = "softelim"
    horizon: int


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]
    log_every_steps: int
    run_name: Optional[str]
    run_id: Optional[str]
    resume: bool = False


class BetaPriorConfig(BaseModel):
    name: str = "beta"
    num_actions: int
    init_alpha: Optional[float]
    init_beta: Optional[float]
    epsilon: float = 1e-3


class PolicyGradEstimatorConfig(BaseModel):
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


class TrainerConfig(BaseModel):
    policy_lr: float
    prior_lr: float
    monte_carlo_samples: int
    epochs: int
    batch_size: int
    max_horizon: int
    policy_grad: PolicyGradEstimatorConfig


class ExperiorConfig(BaseModel):
    policy: Union[TransformerPolicyConfig, SoftElimPolicyConfig]
    prior: BetaPriorConfig
    trainer: TrainerConfig
    fix_prior: bool
    seed: int
    test_run: bool
    wandb: WandbConfig
    out_dir: str
    save_every_steps: int
    keep_every_steps: int
    ckpt_dir: Optional[str]
