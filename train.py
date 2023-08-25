import hydra
import wandb

from omegaconf import OmegaConf
from tqdm import tqdm

from src.trainer import BayesRegretTrainer
from src.utils import PRNGSequence, init_run_dir
from src.configs import ExperiorConfig

from pprint import pprint

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: ExperiorConfig):
    conf = ExperiorConfig(**OmegaConf.to_object(conf))

    if conf.test_run:
        conf.train.epochs = 100
        conf.train.monte_carlo_samples = 64
        conf.train.batch_size = 16
    else:
        conf = init_run_dir(conf)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.dict(),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread")
        )

    rng = PRNGSequence(conf.seed)
    trainer = BayesRegretTrainer(conf)
    trainer.initialize(next(rng))

    pbar = tqdm(range(conf.train.epochs))
    for epoch in pbar:
        mu_vectors = trainer.sample_envs(next(rng))
        b_size = conf.train.batch_size
        inner_steps = conf.train.monte_carlo_samples // b_size

        for i in range(inner_steps):
            batch = mu_vectors[i * b_size: (i + 1) * b_size]
            _, aux = trainer.train_step(next(rng), batch)
            log_str = ' '.join(['{}: {: .4f}'.format(k, v)
                                for (k, v) in aux.items()])
            pbar.set_postfix_str(log_str)
        pbar.update(1)

        # TODO log average metrics for the epoch
        if epoch % conf.wandb.log_every_steps == 0 and not conf.test_run:
            wandb.log(aux, step=epoch)

    if not conf.test_run:
        trainer.save_metrics(next(rng))


if __name__ == '__main__':
    main()
