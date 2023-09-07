from pydantic import BaseModel
from typing import Any, Optional, Union

import jax.numpy as jnp


class TransformerPolicyConfig(BaseModel):
    name: str = "transformer"
    horizon: int
    n_blocks: int
    h_dim: int
    num_heads: int
    drop_p: float = 0.
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


class TrainerConfig(BaseModel):
    policy_lr: float
    prior_lr: float
    monte_carlo_samples: int
    epochs: int
    batch_size: int
    max_horizon: int


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
