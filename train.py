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
        pprint(conf.dict())
        conf.trainer.epochs = 100
        conf.trainer.monte_carlo_samples = 64
        conf.trainer.batch_size = 16
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
            settings=wandb.Settings(start_method="thread"),
        )

    rng = PRNGSequence(conf.seed)

    trainer = BayesRegretTrainer(conf)
    trainer.initialize(next(rng))
    trainer.train(next(rng))

    if not conf.test_run:
        trainer.save_metrics(next(rng))

    trainer.ckpt_manager.close()

if __name__ == "__main__":
    main()
