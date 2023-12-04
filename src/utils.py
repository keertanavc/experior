import jax
import jax.numpy as jnp
import os
import yaml
import wandb
import chex

from jax.scipy.special import xlogy
from random_word import RandomWords


from src.configs import ExperiorConfig


# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (
            hasattr(key_or_seed, "shape")
            and (not key_or_seed.shape)
            and hasattr(key_or_seed, "dtype")
            and key_or_seed.dtype == jnp.int32
        ):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


def init_run_dir(conf: ExperiorConfig) -> ExperiorConfig:
    # Handle preemption and resume
    run_name = conf.wandb.run_name
    resume = True
    if run_name is None:
        r = RandomWords()
        w1, w2 = r.get_random_word(), r.get_random_word()
        run_name = f"{w1}_{w2}"

    out_dir = os.path.join(conf.out_dir, run_name)
    ckpt_dir = os.path.join(out_dir, "ckpt")

    config_yaml = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_yaml):
        with open(config_yaml) as fp:
            old_conf = ExperiorConfig(**yaml.load(fp, Loader=yaml.Loader))
        run_id = old_conf.wandb.run_id
    else:
        run_id = wandb.util.generate_id()
        resume = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(ckpt_dir)
        resume = False
    elif not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        resume = False

    conf.out_dir = out_dir
    conf.ckpt_dir = ckpt_dir
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    with open(config_yaml, "w") as fp:
        yaml.dump(conf.dict(), fp, default_flow_style=False)

    return conf


# # TODO better document
# def get_policy_prior_from_run(
#     run_path, step=None, only_conf=False
# ) -> (Callable, Callable, Callable, ExperiorConfig):
#     with open(os.path.join(run_path, "config.yaml")) as fp:
#         conf = ExperiorConfig(**yaml.load(fp, Loader=yaml.Loader))
#     if only_conf:
#         return None, None, None, conf

#     trainer = Trainer(conf)
#     rng = PRNGSequence(0)
#     trainer.initialize(next(rng))
#     ckpt = trainer.load_states(step)
#     policy_state = ckpt["policy_model"]
#     prior_state = ckpt["prior_model"]

#     def policy_fn(key, t, a, r):
#         return policy_state.apply_fn({"params": policy_state.params}, key, t, a, r)

#     def prior_fn(key, size):
#         return prior_state.apply_fn(
#             {"params": prior_state.params},
#             rng_key=key,
#             size=size,
#             method="sample",
#         )

#     if conf.prior.name == "MaxEnt":
#         raise Exception("Fix the MaxEnt density function")  # TODO

#     return policy_fn, prior_fn, conf


def kl_safe(p, q, eps=1e-6):
    """Returns the KL divergence between two distributions.

    Args:
        p: The first distribution.
        q: The second distribution.
    """
    q = (q + eps) / q.shape[-1]
    return (xlogy(p, p) - xlogy(p, q)).sum(axis=-1)


def moving_average(data: jnp.array, window_size: int):
    """Smooth data by calculating the moving average over a specified window size."""
    return jnp.convolve(data, jnp.ones(window_size) / window_size, mode="valid")


def uniform_sample_ball(rng_key, size: int, d: int) -> chex.Array:
    """Samples uniformly from the unit ball in R^{d}.

    Args:
        rng_key: A JAX random key.
        size: The size of the sample.
        d: The dimension of the ball.

    Returns:
        A sample from the unit ball in R^{d}, shape (size, d).
    """
    key1, key2 = jax.random.split(rng_key)
    norm = jax.random.normal(key1, (size, d))
    sphere = norm / jnp.linalg.norm(norm, axis=-1, keepdims=True)

    # Generate random radii
    random_radii = jax.random.uniform(key2, (size, 1)) ** (1 / d)
    return sphere * random_radii
