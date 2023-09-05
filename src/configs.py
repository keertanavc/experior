from pydantic import BaseModel
from typing import Any, Optional, Union

import jax.numpy as jnp


class TransformerPolicyConfig(BaseModel):
    horizon: int
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
    epsilon: float = 1e-3


class TrainerConfig(BaseModel):
    policy_lr: float
    prior_lr: float
    monte_carlo_samples: int
    epochs: int
    batch_size: int
    max_horizon: int


class ExperiorConfig(BaseModel):
    policy: Union[TransformerPolicyConfig, Any]
    prior: Union[BetaPriorConfig, Any]
    trainer: TrainerConfig
    fix_prior: bool
    seed: int
    test_run: bool
    wandb: WandbConfig
    out_dir: str
    save_every_steps: int
    keep_every_steps: int
    ckpt_dir: Optional[str]
