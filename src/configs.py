from pydantic import BaseModel, validator
from typing import Any, Optional

import jax.numpy as jnp

class TransformerPolicyConfig(BaseModel):
    horizon: int
    num_actions: int
    n_blocks: int
    h_dim: int
    num_heads: int
    drop_p: float
    dtype: Any = jnp.float32


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]
    log_every_steps: int
    run_name: Optional[str]
    run_id: Optional[str]
    resume: bool = False


class BetaPriorConfig(BaseModel):
    num_actions: int
    init_alpha: Optional[float]
    init_beta: Optional[float]


class TrainerConfig(BaseModel):
    policy_lr: float
    prior_lr: float
    monte_carlo_samples: int
    epochs: int
    batch_size: int


class ExperiorConfig(BaseModel):
    policy: TransformerPolicyConfig
    prior: BetaPriorConfig
    train: TrainerConfig
    fix_prior: bool
    seed: int
    test_run: bool
    wandb: WandbConfig
    out_dir: str
    save_every_steps: int
    keep_every_steps: int
    ckpt_dir: Optional[str]

    @validator('prior')
    def prior_validation(cls, prior_value, values, field, config):
        num_actions = values['policy'].num_actions

        assert prior_value.num_actions == num_actions, \
            f"Prior num_actions ({prior_value.num_actions}) must match policy num_actions ({num_actions})"
        return prior_value
