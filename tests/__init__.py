from src.configs import (
    ExperiorConfig,
    TransformerPolicyConfig,
    BetaPriorConfig,
    TrainerConfig,
    WandbConfig,
    PolicyGradEstimatorConfig,
)
import jax.numpy as jnp

TEST_CONFIG = ExperiorConfig(
    policy=TransformerPolicyConfig(
        horizon=10,
        h_dim=64,
        dtype=jnp.float32,
        num_heads=4,
        drop_p=0.1,
        n_blocks=3,
    ),
    prior=BetaPriorConfig(num_actions=3, init_alpha=2, init_beta=1),
    trainer=TrainerConfig(
        policy_lr=1e-3,
        prior_lr=1e-3,
        monte_carlo_samples=10,
        epochs=10,
        batch_size=10,
        max_horizon=10,
        policy_grad=PolicyGradEstimatorConfig(),
    ),
    fix_prior=False,
    seed=42,
    test_run=True,
    wandb=WandbConfig(project="test", log_every_steps=1),
    out_dir="output/tests/",
    save_every_steps=100,
    keep_every_steps=100,
)
