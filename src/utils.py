from typing import Tuple
from src.configs import ExperiorConfig
from random_word import RandomWords

import jax.numpy as jnp
import jax

import os
import yaml
import wandb

from flax.training import train_state

from typing import Any
from omegaconf import OmegaConf

import optax

PRNGKey = Any
Params = Any
Variables = Any
OptState = optax.OptState

Shape = Tuple[int, ...]
Dtype = Any
Array = Any


# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
              hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


class TrainState(train_state.TrainState):
    stats: Variables
    init_stats: Variables
    init_params: Params

    def init_opt_state(self):  # Initializes the optimizer state. TODO make sure it's correct
        new_opt_state = self.tx.init(self.params)
        return self.replace(opt_state=new_opt_state)

    def init_param_state(self):
        return self.replace(stats=self.init_stats, params=self.init_params)


def init_run_dir(conf: ExperiorConfig) -> ExperiorConfig:
    # Handle preemption and resume
    run_name = conf.wandb.run_name
    resume = True
    if run_name is None:
        r = RandomWords()
        w1, w2 = r.get_random_word(), r.get_random_word()
        run_name = f"{w1}_{w2}"

    out_dir = os.path.join(conf.out_dir, run_name)

    config_yaml = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_yaml):
        with open(config_yaml) as fp:
            old_conf = ExperiorConfig(**yaml.safe_load(fp))
        run_id = old_conf.wandb.run_id
    else:
        run_id = wandb.util.generate_id()
        resume = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        resume = False
    elif not os.path.exists(os.path.join(out_dir, "state.pt")):
        resume = False

    conf.out_dir = out_dir
    conf.ckpt_dir = os.path.join(out_dir, "ckpt")
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    OmegaConf.save(conf.dict(), os.path.join(out_dir, "config.yaml"))

    return conf
