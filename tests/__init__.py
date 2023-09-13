from src.configs import (
    ExperiorConfig,
    TransformerPolicyConfig,
    BetaPriorConfig,
    TrainerConfig,
    WandbConfig,
    GradEstimatorConfig,
    ModelTrainerConfig,
)
import jax.numpy as jnp

TEST_CONFIG = ExperiorConfig(
    policy=TransformerPolicyConfig(
        h_dim=64,
        dtype=jnp.float32,
        num_heads=4,
        drop_p=0.1,
        n_blocks=3,
    ),
    prior=BetaPriorConfig(name="beta", num_actions=3, init_alpha=2, init_beta=1),
    trainer=TrainerConfig(
        name="minimax",
        policy_trainer=ModelTrainerConfig(
            lr=1e-3,
            batch_size=32,
            epochs=10,
            mc_samples=32,
            grad_est=GradEstimatorConfig(name="reinforce"),
        ),
        prior_trainer=ModelTrainerConfig(
            lr=1e-3, batch_size=32, epochs=10, mc_samples=32
        ),
        test_horizon=20,
        train_horizon=10,
    ),
    seed=42,
    test_run=True,
    wandb=WandbConfig(project="test"),
    out_dir="output/tests/",
    save_every_steps=100,
    keep_every_steps=100,
)
