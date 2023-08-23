from src.trainer import BayesRegretTrainer
from src.utils import PRNGSequence
from src.configs import TransformerPolicyConfig, OptimizerConfig, GlobalConfig
from src.eval import uniform_bayes_regret

import tqdm
import jax.numpy as jnp
import time

import matplotlib.pyplot as plt

def train(policy_config: TransformerPolicyConfig, optimizer_config: OptimizerConfig, global_config: GlobalConfig):
    rng = PRNGSequence(global_config.seed)
    trainer = BayesRegretTrainer(policy_config, optimizer_config)
    trainer.initialize(next(rng))

    with tqdm.tqdm(total=global_config.epochs) as pbar:
        for _ in range(global_config.epochs):
            loss, aux = trainer.train_step(next(rng))
            log_str = ' '.join(['{}: {: .4f}'.format(k, v) for (k, v) in aux.items()])
            pbar.set_postfix_str(log_str)
            pbar.update(1)
    regret = uniform_bayes_regret(next(rng), trainer.policy_state, trainer.pc, n_envs=100)
    plt.plot(regret)
    plt.savefig('outputs/regret.png')
    plt.close()

if __name__ == '__main__':
    pc = TransformerPolicyConfig(horizon=100,
                                 h_dim=64,
                                 dtype=jnp.float32,
                                 num_heads=4,
                                 drop_p=0.1,
                                 n_blocks=2,
                                 num_actions=5)

    oc = OptimizerConfig(policy_lr=1e-3, prior_lr=1e-4, mc_samples=100)
    gc = GlobalConfig(epochs=600, seed=123)

    train(pc, oc, gc)
